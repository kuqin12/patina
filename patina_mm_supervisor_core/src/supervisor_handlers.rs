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

use crate::request_handler::{
    MmSupervisorRequestHeader, MmSupervisorVersionInfo,
    requests, responses, SIGNATURE, REVISION,
};

use patina_mm::protocol::mm_supervisor_request::MM_SUPERVISOR_REQUEST_HANDLER_GUID;

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
/// After this runs, no new memory regions can be unblocked and certain MMI handlers
/// are unregistered.
fn mm_ready_to_lock_handler(_comm_buffer: *mut u8, _comm_buffer_size: &mut usize) -> efi::Status {
    log::info!("MmReadyToLockHandler invoked");

    // TODO: Implement the actual ready-to-lock logic, such as:
    // - Take a memory policy snapshot
    // - Mark the supervisor as locked (mMmReadyToLockDone = true)
    // - Unregister any handlers that should not survive past ready-to-lock

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
            log::debug!("Processing VERSION_INFO request");
            handle_version_info(comm_buffer, comm_buffer_size)
        }
        requests::FETCH_POLICY => {
            log::debug!("Processing FETCH_POLICY request");
            handle_fetch_policy(comm_buffer, comm_buffer_size)
        }
        requests::COMM_UPDATE => {
            log::debug!("Processing COMM_UPDATE request");
            handle_comm_update(comm_buffer, comm_buffer_size)
        }
        requests::UNBLOCK_MEM => {
            log::debug!("Processing UNBLOCK_MEM request");
            handle_unblock_mem(comm_buffer, comm_buffer_size)
        }
        unknown => {
            log::warn!("MmSupvRequestHandler: unsupported request type 0x{:08X}", unknown);
            // Write error result into the header
            write_request_result(comm_buffer, responses::ERROR);
            *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
            return efi::Status::UNSUPPORTED;
        }
    };

    status
}

/// Write a result value into the request header's `result` field.
///
/// # Safety
///
/// `comm_buffer` must point to at least `MmSupervisorRequestHeader::SIZE` bytes of writable memory.
fn write_request_result(comm_buffer: *mut u8, result: u64) {
    // SAFETY: caller guarantees buffer is large enough for the header.
    unsafe {
        let header = &mut *(comm_buffer as *mut MmSupervisorRequestHeader);
        header.result = result;
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
        write_request_result(comm_buffer, responses::ERROR);
        *comm_buffer_size = MmSupervisorRequestHeader::SIZE;
        return efi::Status::BUFFER_TOO_SMALL;
    }

    // Write success into the header
    write_request_result(comm_buffer, responses::SUCCESS);

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
/// Returns the current security policy to the caller.
fn handle_fetch_policy(comm_buffer: *mut u8, comm_buffer_size: &mut usize) -> efi::Status {
    log::info!("FETCH_POLICY request");

    // TODO: Return the actual memory protection policy from the policy engine.
    // For now, write success and indicate empty policy.
    write_request_result(comm_buffer, responses::SUCCESS);
    *comm_buffer_size = MmSupervisorRequestHeader::SIZE;

    efi::Status::UNSUPPORTED
}

/// Handle a COMM_UPDATE request.
///
/// Updates the communication buffer address for future SMI entries.
fn handle_comm_update(comm_buffer: *mut u8, comm_buffer_size: &mut usize) -> efi::Status {
    log::info!("COMM_UPDATE request");

    // TODO: Parse the new communication buffer descriptor from the payload,
    // validate it against SMRAM, and update the internal comm buffer config.
    write_request_result(comm_buffer, responses::SUCCESS);
    *comm_buffer_size = MmSupervisorRequestHeader::SIZE;

    efi::Status::UNSUPPORTED
}

/// Handle an UNBLOCK_MEM request.
///
/// Unblocks a memory region so that user-mode MM drivers can access it.
fn handle_unblock_mem(comm_buffer: *mut u8, comm_buffer_size: &mut usize) -> efi::Status {
    log::info!("UNBLOCK_MEM request");

    // TODO: Parse the memory region descriptor from the payload, validate
    // against the memory policy, and update page table permissions.
    write_request_result(comm_buffer, responses::SUCCESS);
    *comm_buffer_size = MmSupervisorRequestHeader::SIZE;

    efi::Status::UNSUPPORTED
}
