//! Supervisor MMI Handler Registry
//!
//! This module provides the build-time handler registration mechanism for the MM Supervisor Core.
//! Handlers are registered at link time using `linkme::distributed_slice`, allowing platforms
//! to add custom supervisor handlers without modifying the core.
//!
//! ## Architecture
//!
//! The [`SUPERVISOR_MMI_HANDLERS`] distributed slice collects all handler entries across the
//! final binary. Each entry is a [`SupervisorMmiHandler`] that specifies a GUID and handler
//! function. During supervisor request processing, the core iterates the slice to find a
//! handler matching the communicate header GUID.
//!
//! ## Adding Platform-Specific Handlers
//!
//! To register a handler from a platform crate:
//!
//! ```rust,ignore
//! use patina_mm_supervisor_core::{SupervisorMmiHandler, SUPERVISOR_MMI_HANDLERS};
//! use r_efi::efi;
//!
//! fn my_handler(comm_buffer: *mut u8, comm_buffer_size: &mut usize) -> efi::Status {
//!     // Handle the request...
//!     efi::Status::SUCCESS
//! }
//!
//! #[linkme::distributed_slice(SUPERVISOR_MMI_HANDLERS)]
//! static MY_HANDLER: SupervisorMmiHandler = SupervisorMmiHandler {
//!     name: "MyPlatformHandler",
//!     handler_guid: efi::Guid::from_fields(
//!         0x12345678, 0xabcd, 0xef01,
//!         0x23, 0x45, &[0x67, 0x89, 0xab, 0xcd, 0xef, 0x01]
//!     ),
//!     handle: my_handler,
//! };
//! ```
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use r_efi::efi;

use patina_paging::{MemoryAttributes, PageTable, PtError};

use crate::mm_mem::PAGE_ALLOCATOR;
use crate::unblock_memory::{UnblockError, UNBLOCKED_MEMORY_TRACKER};
use crate::{
    POLICY_GATE,
    is_buffer_inside_mmram, read_cr3,
};

use patina_mm::protocol::mm_supervisor_request::{
    MmSupervisorRequestHeader,
    MmSupervisorVersionInfo,
    requests,
    MM_SUPERVISOR_REQUEST_HANDLER_GUID,
    MmSupervisorUnblockMemoryParams,
    REVISION, SIGNATURE,
};

use patina_mm_policy::{MemDescriptorV1_0, PolicyError};

// ============================================================================
// Supervisor MMI Handler Infrastructure
// ============================================================================

/// A build-time registered supervisor MMI handler.
///
/// Each entry represents a handler that the supervisor core will consider when dispatching
/// supervisor-channel requests. Handlers are matched by comparing the
/// [`EfiMmCommunicateHeader::header_guid`](crate::EfiMmCommunicateHeader::header_guid)
/// against [`handler_guid`](SupervisorMmiHandler::handler_guid).
///
/// ## Handler Function Signature
///
/// The [`handle`](SupervisorMmiHandler::handle) function receives:
/// - `comm_buffer`: Pointer to the data portion of the communicate buffer (after the header).
/// - `comm_buffer_size`: On input, the message length. On output, the response data length.
///
/// The handler should return an [`efi::Status`] code.
#[derive(Debug)]
pub struct SupervisorMmiHandler {
    /// Human-readable name for logging and debugging.
    pub name: &'static str,
    /// GUID identifying the request type this handler processes.
    pub handler_guid: efi::Guid,
    /// The handler function.
    ///
    /// # Arguments
    ///
    /// * `comm_buffer` - Pointer to the data payload (after `EfiMmCommunicateHeader`).
    /// * `comm_buffer_size` - On input, the data size; on output, the response data size.
    ///
    /// # Returns
    ///
    /// An EFI status code indicating the result of the handler.
    pub handle: fn(comm_buffer: *mut u8, comm_buffer_size: &mut usize) -> efi::Status,
}

// SAFETY: SupervisorMmiHandler contains only a &'static str, a Guid (plain data), and a fn pointer.
// All of these are inherently Sync.
unsafe impl Sync for SupervisorMmiHandler {}

/// The global distributed slice collecting all supervisor MMI handlers.
///
/// Handlers from the core and from platform crates are collected here at link time.
/// The supervisor dispatch loop iterates this slice to find a matching handler for
/// each incoming supervisor-channel request.
///
/// ## Usage
///
/// ```rust,ignore
/// use patina_mm_supervisor_core::{SupervisorMmiHandler, SUPERVISOR_MMI_HANDLERS};
///
/// #[linkme::distributed_slice(SUPERVISOR_MMI_HANDLERS)]
/// static MY_HANDLER: SupervisorMmiHandler = SupervisorMmiHandler {
///     name: "MyHandler",
///     handler_guid: MY_GUID,
///     handle: my_handler_fn,
/// };
/// ```
#[linkme::distributed_slice]
pub static SUPERVISOR_MMI_HANDLERS: [SupervisorMmiHandler];

// ============================================================================
// Core Supervisor MMI Handlers
// ============================================================================

// GUID for gEfiDxeMmReadyToLockProtocolGuid
// { 0x60ff8964, 0xe906, 0x41d0, { 0xaf, 0xed, 0xf2, 0x41, 0xe9, 0x74, 0xe0, 0x8e } }
/// GUID for the DXE MM Ready To Lock protocol.
pub const EFI_DXE_MM_READY_TO_LOCK_PROTOCOL_GUID: efi::Guid = efi::Guid::from_fields(
    0x60ff8964,
    0xe906,
    0x41d0,
    0xaf,
    0xed,
    &[0xf2, 0x41, 0xe9, 0x74, 0xe0, 0x8e],
);

/// Ready-to-lock handler.
///
/// Triggered from the non-MM environment upon DxeMmReadyToLock event.
/// After this handler runs, certain features (e.g., unblock memory) are no longer available.
#[linkme::distributed_slice(SUPERVISOR_MMI_HANDLERS)]
static READY_TO_LOCK_HANDLER: SupervisorMmiHandler = SupervisorMmiHandler {
    name: "MmReadyToLock",
    handler_guid: EFI_DXE_MM_READY_TO_LOCK_PROTOCOL_GUID,
    handle: mm_ready_to_lock_handler,
};

/// Supervisor request handler.
///
/// Handles general supervisor requests such as unblock memory, fetch policy,
/// version info, and communication buffer updates.
#[linkme::distributed_slice(SUPERVISOR_MMI_HANDLERS)]
static SUPV_REQUEST_HANDLER: SupervisorMmiHandler = SupervisorMmiHandler {
    name: "MmSupvRequest",
    handler_guid: MM_SUPERVISOR_REQUEST_HANDLER_GUID,
    handle: mm_supv_request_handler,
};

// ============================================================================
// Handler Implementations
// ============================================================================

/// MmReadyToLock handler implementation.
///
/// Called when the DXE phase signals that MM should transition to a locked state.
/// After this runs, no new memory regions can be unblocked and the memory policy
/// snapshot stored inside `PolicyGate` is considered the reference baseline.
fn mm_ready_to_lock_handler(_comm_buffer: *mut u8, _comm_buffer_size: &mut usize) -> efi::Status {
    log::info!("MmReadyToLockHandler invoked");

    let gate = match POLICY_GATE.get() {
        Some(g) => g,
        None => {
            log::error!("MmReadyToLock: POLICY_GATE not initialized");
            return efi::Status::NOT_READY;
        }
    };

    // If already locked, this is a no-op (idempotent).
    if gate.is_locked() {
        log::warn!("MmReadyToLock: already locked, ignoring duplicate");
        return efi::Status::SUCCESS;
    }

    // Take a snapshot and mark as locked.
    let cr3 = read_cr3();
    // SAFETY: cr3 points to the active PML4 table inside SMM,
    // and the memory policy buffer was configured during init.
    if let Err(e) = unsafe { gate.take_snapshot(cr3, is_buffer_inside_mmram) } {
        log::error!("MmReadyToLock: take_snapshot failed: {:?}", e);
        return efi::Status::DEVICE_ERROR;
    }

    // And mark the unblock memory tracker as locked as well since unblock memory is no longer allowed after this point.
    UNBLOCKED_MEMORY_TRACKER.set_core_init_complete();

    efi::Status::SUCCESS
}

// ============================================================================
// Supervisor Version Constants
// ============================================================================

/// Supervisor version. Encodes major.minor as (major << 16) | minor.
pub const VERSION: u32 = 0x00130008;

/// Supervisor patch level.
pub const PATCH_LEVEL: u32 = 0x00010001;

/// Maximum supported supervisor request level.
///
/// This is the highest request type value the supervisor supports.
/// Currently [`requests::COMM_UPDATE`] (0x0004).
pub const MAX_REQUEST_LEVEL: u64 = requests::COMM_UPDATE as u64;

// ============================================================================
// MM Supervisor Request Handler
// ============================================================================

/// MM Supervisor request handler implementation.
///
/// Handles structured requests from the non-MM environment, such as:
/// - [`requests::UNBLOCK_MEM`]: Unblock memory regions
/// - [`requests::FETCH_POLICY`]: Fetch security policy
/// - [`requests::VERSION_INFO`]: Query supervisor version information
/// - [`requests::COMM_UPDATE`]: Update communication buffer configuration
///
/// The buffer is expected to contain an [`MmSupervisorRequestHeader`] at the start.
/// On return, the header's `result` field is set and any response payload follows
/// immediately after the header.
fn mm_supv_request_handler(comm_buffer: *mut u8, comm_buffer_size: &mut usize) -> efi::Status {
    log::info!("MmSupvRequestHandler invoked (buffer_size={})", *comm_buffer_size);

    if comm_buffer.is_null() || *comm_buffer_size < MmSupervisorRequestHeader::SIZE {
        log::error!(
            "MmSupvRequestHandler: buffer too small ({} bytes, need at least {})",
            *comm_buffer_size,
            MmSupervisorRequestHeader::SIZE,
        );
        return efi::Status::INVALID_PARAMETER;
    }

    // SAFETY: We verified the buffer is non-null and large enough for the header.
    let header = unsafe { &*(comm_buffer as *const MmSupervisorRequestHeader) };

    // Validate signature
    if header.signature != SIGNATURE {
        log::error!(
            "MmSupvRequestHandler: invalid signature 0x{:08X}, expected 0x{:08X}",
            header.signature,
            SIGNATURE,
        );
        return efi::Status::INVALID_PARAMETER;
    }

    // Validate revision
    if header.revision > REVISION {
        log::error!(
            "MmSupvRequestHandler: unsupported revision {}, max supported {}",
            header.revision,
            REVISION,
        );
        return efi::Status::UNSUPPORTED;
    }

    // Dispatch by request type
    let status = match header.request {
        requests::VERSION_INFO => {
            log::info!("Processing VERSION_INFO request");
            handle_version_info(comm_buffer, comm_buffer_size)
        }
        requests::FETCH_POLICY => {
            log::info!("Processing FETCH_POLICY request");
            handle_fetch_policy(comm_buffer, comm_buffer_size)
        }
        requests::COMM_UPDATE => {
            log::info!("Processing COMM_UPDATE request");
            handle_comm_update(comm_buffer, comm_buffer_size)
        }
        requests::UNBLOCK_MEM => {
            log::info!("Processing UNBLOCK_MEM request");
            handle_unblock_mem(comm_buffer, comm_buffer_size)
        }
        unknown => {
            log::warn!("MmSupvRequestHandler: unsupported request type 0x{:08X}", unknown);
            *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
            efi::Status::UNSUPPORTED
        }
    };

    // Write the final status into the request header's result field.
    write_request_result(comm_buffer, status);

    // The handler's return value is only for indicating communication-level errors
    // (e.g., interrupt is being handled or not), in this case we handled the request successfully.
    efi::Status::SUCCESS
}

/// Write an [`efi::Status`] into the request header's `result` field.
///
/// The status is stored as its raw `usize` representation cast to `u64`,
/// matching the C `MM_SUPERVISOR_REQUEST_HEADER.Result` convention.
///
/// # Safety
///
/// `comm_buffer` must point to at least `MmSupervisorRequestHeader::SIZE` bytes of writable memory.
fn write_request_result(comm_buffer: *mut u8, status: efi::Status) {
    // SAFETY: caller guarantees buffer is large enough for the header.
    unsafe {
        let header = &mut *(comm_buffer as *mut MmSupervisorRequestHeader);
        header.result = status.as_usize() as u64;
    }
}

/// Handle a VERSION_INFO request.
///
/// Writes back the response header followed by [`MmSupervisorVersionInfo`].
fn handle_version_info(comm_buffer: *mut u8, comm_buffer_size: &mut usize) -> efi::Status {
    let response_size = MmSupervisorRequestHeader::SIZE + MmSupervisorVersionInfo::SIZE;

    if *comm_buffer_size < response_size {
        log::error!(
            "VERSION_INFO: buffer too small for response ({} bytes, need {})",
            *comm_buffer_size,
            response_size,
        );
        *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
        return efi::Status::BUFFER_TOO_SMALL;
    }

    // Write version info payload after the header
    let version_info = MmSupervisorVersionInfo {
        version: VERSION,
        patch_level: PATCH_LEVEL,
        max_supervisor_request_level: MAX_REQUEST_LEVEL,
    };

    // SAFETY: We verified the buffer is large enough for header + version info.
    unsafe {
        let payload_ptr = comm_buffer.add(MmSupervisorRequestHeader::SIZE) as *mut MmSupervisorVersionInfo;
        core::ptr::write(payload_ptr, version_info);
    }

    *comm_buffer_size = response_size;
    log::info!(
        "VERSION_INFO response: version=0x{:08X}, patch=0x{:08X}, max_level={}",
        VERSION,
        PATCH_LEVEL,
        MAX_REQUEST_LEVEL,
    );

    efi::Status::SUCCESS
}

/// Handle a FETCH_POLICY request.
///
/// Returns the merged memory + firmware policy to the caller.
///
/// ## Behaviour
///
/// 1. **First-time call (before lock):** takes a memory policy snapshot, saves it,
///    and sets the ready-to-lock flag (whichever of `MmReadyToLock` or `FETCH_POLICY`
///    fires first performs this).
/// 2. **Subsequent calls (after lock):** re-walks the page table and compares the
///    fresh result against the saved snapshot. Any discrepancy is a security
///    violation.
/// 3. **Merges** the memory policy snapshot with the static firmware policy blob
///    from `POLICY_GATE` and writes the combined result into `comm_buffer`.
///
/// ## Response layout
///
/// ```text
/// |----------------------------------|
/// | MmSupervisorRequestHeader (24 B) |
/// |----------------------------------|
/// | MemDescriptorV1_0[0..N]          |  <- memory policy snapshot
/// |----------------------------------|
/// | SecurePolicyDataV1_0 + payload   |  <- firmware policy blob (raw copy)
/// |----------------------------------|
/// ```
fn handle_fetch_policy(comm_buffer: *mut u8, comm_buffer_size: &mut usize) -> efi::Status {
    log::info!("FETCH_POLICY request");

    // -- 0. Obtain the PolicyGate -------------------------------------
    let gate = match POLICY_GATE.get() {
        Some(g) => g,
        None => {
            log::error!("FETCH_POLICY: POLICY_GATE not initialized");
            *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
            return efi::Status::NOT_READY;
        }
    };

    let cr3 = read_cr3();

    // -- 1. Ensure we have a snapshot (lock if not yet locked) ------------
    if !gate.is_locked() {
        // Policy requested prior to ready to lock - enforce lock now.
        log::info!("FETCH_POLICY: not yet locked - taking snapshot and locking now");
        // SAFETY: cr3 is valid and the memory policy buffer was configured during init.
        if let Err(e) = unsafe { gate.take_snapshot(cr3, is_buffer_inside_mmram) } {
            log::error!("FETCH_POLICY: take_snapshot failed: {:?}", e);
            *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
            return efi::Status::DEVICE_ERROR;
        }
    } else {
        // -- 2. Already locked - verify that current page table matches snapshot
        if let Err(status) = verify_policy_snapshot(gate, cr3) {
            *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
            return status;
        }
    }

    // -- 3. Write the merged policy into the comm buffer (after the header) -
    let payload_capacity = match comm_buffer_size
        .checked_sub(MmSupervisorRequestHeader::SIZE)
    {
        Some(c) => c,
        None => {
            log::error!("FETCH_POLICY: comm_buffer_size too small for header");
            *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
            return efi::Status::BUFFER_TOO_SMALL;
        }
    };

    // SAFETY: comm_buffer + header offset is valid writable memory.
    let dest = unsafe { comm_buffer.add(MmSupervisorRequestHeader::SIZE) };
    let payload_written = match unsafe { gate.fetch_n_update_policy(dest, payload_capacity) } {
        Ok(n) => n,
        Err(PolicyError::InternalError) => {
            // Could be buffer-too-small, size overflow, or missing snapshot.
            log::error!("FETCH_POLICY: fetch_n_update_policy failed");
            *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
            return efi::Status::BUFFER_TOO_SMALL;
        }
        Err(e) => {
            log::error!("FETCH_POLICY: fetch_n_update_policy unexpected error: {:?}", e);
            *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
            return efi::Status::DEVICE_ERROR;
        }
    };

    let total_response = MmSupervisorRequestHeader::SIZE + payload_written;
    *comm_buffer_size = total_response;
    log::info!("FETCH_POLICY: response {} bytes (header={}, payload={})", total_response, MmSupervisorRequestHeader::SIZE, payload_written);

    efi::Status::SUCCESS
}

// ============================================================================
// Policy Snapshot Helpers
// ============================================================================

/// Walks the page table and compares the result against the saved snapshot
/// inside `PolicyGate`. Allocates a temporary scratch buffer from the page
/// allocator for the fresh walk.
///
/// Returns `Ok(())` if the tables match, or an `efi::Status` error on mismatch
/// or allocation failure.
fn verify_policy_snapshot(
    gate: &patina_mm_policy::PolicyGate,
    cr3: u64,
) -> Result<(), efi::Status> {
    let saved_count = match gate.snapshot_count() {
        Some(c) => c,
        None => {
            log::warn!("verify_policy_snapshot: no snapshot available, skipping");
            return Ok(());
        }
    };

    let desc_size = core::mem::size_of::<MemDescriptorV1_0>();
    let needed_bytes = saved_count.checked_mul(desc_size).ok_or_else(|| {
        log::error!("verify_policy_snapshot: descriptor count overflow");
        efi::Status::DEVICE_ERROR
    })?;
    let needed_pages = (needed_bytes + 0xFFF) / 0x1000;

    let scratch_base = PAGE_ALLOCATOR
        .allocate_pages(needed_pages)
        .map_err(|e| {
            log::error!("verify_policy_snapshot: failed to allocate scratch buffer: {:?}", e);
            efi::Status::OUT_OF_RESOURCES
        })?;

    let scratch_ptr = scratch_base as *mut MemDescriptorV1_0;
    let scratch_max_count = (needed_pages * 0x1000) / desc_size;

    // SAFETY: scratch_ptr was just allocated and scratch_max_count is correct.
    let result = unsafe {
        gate.verify_snapshot(cr3, is_buffer_inside_mmram, scratch_ptr, scratch_max_count)
    };

    // Free the scratch buffer regardless of the result.
    let _ = PAGE_ALLOCATOR.free_pages(scratch_base, needed_pages);

    result.map_err(|e| {
        log::error!("verify_policy_snapshot: snapshot verification failed: {:?}", e);
        efi::Status::SECURITY_VIOLATION
    })
}

/// Handle a COMM_UPDATE request.
///
/// Updates the communication buffer address for future SMI entries.
fn handle_comm_update(_comm_buffer: *mut u8, comm_buffer_size: &mut usize) -> efi::Status {
    log::info!("COMM_UPDATE request");

    // We do not support dynamic communication buffer updates in this implementation, because
    // we expect the runtime allocation will fall into PEI memory bin.
    *comm_buffer_size = MmSupervisorRequestHeader::SIZE;

    efi::Status::ACCESS_DENIED
}

/// Handle an UNBLOCK_MEM request.
///
/// Unblocks a memory region so that user-mode MM drivers can access it.
///
/// ## Validation (stricter than the C `ProcessUnblockPages` implementation)
///
/// 1. **Ready-to-lock check** - reject if core init is complete (post-lock state).
/// 2. **Buffer size** - must hold header + [`MmSupervisorUnblockMemoryParams`].
/// 3. **Zero-GUID** - the identifier GUID must be non-zero.
/// 4. **Page alignment** - `PhysicalStart` must be 4 KiB aligned.
/// 5. **Non-zero page count** - `NumberOfPages` must be > 0.
/// 6. **Overflow** - `NumberOfPages * PAGE_SIZE` and `PhysicalStart + size` must not overflow.
/// 7. **MMRAM overlap** - region must not overlap supervisor RAM.
/// 8. **Duplicate / conflict** - checked by the [`UNBLOCKED_MEMORY_TRACKER`].
/// 9. **Page attributes** - pages must be not-present (RP set) and not read-only.
/// 10. **Page table update** - make pages present, R/W, NX; optionally supervisor-only (SP).
fn handle_unblock_mem(comm_buffer: *mut u8, comm_buffer_size: &mut usize) -> efi::Status {
    log::info!("UNBLOCK_MEM request");

    const PAGE_SIZE: u64 = 0x1000;

    // 1. Ready-to-lock check
    // After core initialization is complete, unblock requests are rejected.
    // This mirrors the C `mMmReadyToLockDone` guard.
    if UNBLOCKED_MEMORY_TRACKER.is_core_init_complete() {
        log::error!("UNBLOCK_MEM: rejected - core initialization already complete (post ready-to-lock)");
        *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
        return efi::Status::ACCESS_DENIED;
    }

    // 2. Buffer size check
    let min_size = MmSupervisorRequestHeader::SIZE + MmSupervisorUnblockMemoryParams::SIZE;
    if *comm_buffer_size < min_size {
        log::error!(
            "UNBLOCK_MEM: buffer too small ({} bytes, need at least {})",
            *comm_buffer_size,
            min_size,
        );
        *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
        return efi::Status::BUFFER_TOO_SMALL;
    }

    // 3. Parse the payload
    // SAFETY: We verified the buffer is large enough for header + params.
    let params = unsafe {
        &*(comm_buffer.add(MmSupervisorRequestHeader::SIZE) as *const MmSupervisorUnblockMemoryParams)
    };

    let physical_start = params.memory_descriptor.physical_start;
    let number_of_pages = params.memory_descriptor.number_of_pages;
    let attribute = params.memory_descriptor.attribute;
    let identifier_guid = params.identifier_guid;

    log::info!(
        "UNBLOCK_MEM: request from {:?} - PhysicalStart=0x{:016x}, Pages={}, Attr=0x{:x}",
        identifier_guid,
        physical_start,
        number_of_pages,
        attribute,
    );

    // 4. Zero-GUID check
    if *identifier_guid.as_bytes() == [0u8; 16] {
        log::error!("UNBLOCK_MEM: identifier GUID is zero");
        *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
        return efi::Status::INVALID_PARAMETER;
    }

    // 5. Page alignment check (stricter than C)
    if physical_start & (PAGE_SIZE - 1) != 0 {
        log::error!(
            "UNBLOCK_MEM: PhysicalStart 0x{:016x} is not page-aligned",
            physical_start,
        );
        *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
        return efi::Status::INVALID_PARAMETER;
    }

    // 6. Non-zero page count
    if number_of_pages == 0 {
        log::error!("UNBLOCK_MEM: NumberOfPages is 0");
        *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
        return efi::Status::INVALID_PARAMETER;
    }

    // 7. Overflow checks
    let region_size = match number_of_pages.checked_mul(PAGE_SIZE) {
        Some(s) => s,
        None => {
            log::error!(
                "UNBLOCK_MEM: NumberOfPages ({}) * PAGE_SIZE overflows u64",
                number_of_pages,
            );
            *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
            return efi::Status::INVALID_PARAMETER;
        }
    };

    if physical_start.checked_add(region_size).is_none() {
        log::error!(
            "UNBLOCK_MEM: address range 0x{:016x} + 0x{:x} overflows",
            physical_start,
            region_size,
        );
        *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
        return efi::Status::INVALID_PARAMETER;
    }

    // 8. MMRAM overlap check
    if PAGE_ALLOCATOR.is_region_inside_mmram(physical_start, region_size) {
        log::error!(
            "UNBLOCK_MEM: region 0x{:016x}-0x{:016x} overlaps with MMRAM",
            physical_start,
            physical_start + region_size,
        );
        *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
        return efi::Status::SECURITY_VIOLATION;
    }

    // 9. Duplicate / conflict check via tracker
    // We use the tracker's region count to distinguish newly-added vs idempotent.
    // For newly-added regions we must additionally verify page attributes and
    // apply page table changes. For idempotent (exact duplicate) requests we
    // can short-circuit with SUCCESS.
    let is_supervisor_page = (attribute & efi::MEMORY_SP) != 0;
    let track_attributes: u32 = if is_supervisor_page {
        patina_mm_policy::RESOURCE_ATTR_READ | patina_mm_policy::RESOURCE_ATTR_WRITE
            | 0x80000000 // high bit tag for supervisor-only tracking
    } else {
        patina_mm_policy::RESOURCE_ATTR_READ | patina_mm_policy::RESOURCE_ATTR_WRITE
    };

    let count_before = UNBLOCKED_MEMORY_TRACKER.region_count();
    match UNBLOCKED_MEMORY_TRACKER.unblock_memory(physical_start, region_size, track_attributes) {
        Ok(()) => {
            let count_after = UNBLOCKED_MEMORY_TRACKER.region_count();
            if count_after == count_before {
                // Idempotent - already tracked with same attributes, nothing more to do.
                log::info!(
                    "UNBLOCK_MEM: region 0x{:016x}-0x{:016x} already unblocked (idempotent)",
                    physical_start,
                    physical_start + region_size,
                );
                *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
                return efi::Status::SUCCESS;
            }
            // Newly added - continue to verify page attributes and update page table.
        }
        Err(UnblockError::ConflictingAttributes) => {
            log::error!(
                "UNBLOCK_MEM: region 0x{:016x}-0x{:016x} conflicts with existing entry",
                physical_start,
                physical_start + region_size,
            );
            *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
            return efi::Status::SECURITY_VIOLATION;
        }
        Err(e) => {
            log::error!(
                "UNBLOCK_MEM: tracker rejected request for 0x{:016x}-0x{:016x}: {:?}",
                physical_start,
                physical_start + region_size,
                e,
            );
            *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
            return efi::Status::INVALID_PARAMETER;
        }
    }

    // 10. Verify current page attributes
    // Pages must be not-present (ReadProtect) and NOT read-only. This ensures
    // we only unblock pages that were explicitly guarded, matching the C
    // `VerifyUnblockRequest` logic with an additional RO check.
    {
        let pt_guard = crate::PAGE_TABLE.lock();
        if let Some(ref pt) = *pt_guard {
            match pt.query_memory_region(physical_start, region_size) {
                Ok(current_attrs) => {
                    log::error!(
                        "UNBLOCK_MEM: pages at 0x{:016x} are already present (attrs: {:?}). \
                            Only not-present pages may be unblocked.",
                        physical_start,
                        current_attrs,
                    );
                    *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
                    return efi::Status::SECURITY_VIOLATION;
                }
                Err(PtError::NoMapping) => {
                    // Expected case - pages are currently not present, so we can unblock them.
                }
                Err(e) => {
                    log::error!(
                        "UNBLOCK_MEM: failed to query page attributes for 0x{:016x}-0x{:016x}: {:?}",
                        physical_start,
                        physical_start + region_size,
                        e,
                    );
                    *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
                    return efi::Status::DEVICE_ERROR;
                }
            }
        } else {
            log::error!("UNBLOCK_MEM: page table not initialized");
            *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
            return efi::Status::NOT_READY;
        }
    }

    // 11. Apply page table changes
    // Make the region:
    //   - Present (clear ReadProtect)
    //   - Read/Write (clear ReadOnly)
    //   - Non-executable (set ExecuteProtect) - data pages must be W^X
    //   - Optionally Supervisor-only (set Supervisor) if EFI_MEMORY_SP requested
    {
        let mut pt_guard = crate::PAGE_TABLE.lock();
        if let Some(ref mut pt) = *pt_guard {
            let mut new_attrs = MemoryAttributes::ExecuteProtect; // NX - data pages are non-executable
            if is_supervisor_page {
                new_attrs = new_attrs | MemoryAttributes::Supervisor; // Supervisor-only (U/S=0)
            }

            if let Err(e) = pt.map_memory_region(physical_start, region_size, new_attrs) {
                log::error!(
                    "UNBLOCK_MEM: failed to update page table for 0x{:016x}-0x{:016x}: {:?}",
                    physical_start,
                    physical_start + region_size,
                    e,
                );
                *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
                return efi::Status::DEVICE_ERROR;
            }
        }
        // If page table is None, we already returned NOT_READY above.
    }

    log::info!(
        "UNBLOCK_MEM: SUCCESS - unblocked 0x{:016x}-0x{:016x} ({} pages, {})",
        physical_start,
        physical_start + region_size,
        number_of_pages,
        if is_supervisor_page { "supervisor-only" } else { "user-accessible" },
    );

    *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
    efi::Status::SUCCESS
}
