//! MM User Core
//!
//! A pure Rust implementation of the MM User Core for standalone MM mode environments.
//!
//! This crate provides the core functionality for a user-mode (Ring 3) MM module that is
//! invoked by the MM Supervisor Core via privilege demotion. It implements the equivalent
//! functionality of the C `StandaloneMmCore` — discovering drivers from HOBs, evaluating
//! dependency expressions, dispatching drivers, and managing MMI handlers.
//!
//! ## Architecture
//!
//! The user core is invoked by the supervisor with three command types:
//! - **StartUserCore**: One-time initialization. Walk HOBs to discover drivers and dispatch them.
//! - **UserRequest**: Runtime MMI dispatch. Parse the communication buffer and invoke registered handlers.
//! - **UserApProcedure**: Execute a procedure on behalf of an AP.
//!
//! ## Entry Protocol
//!
//! The supervisor calls the user core entry point with three arguments:
//! - `arg1` (`u64`): Command type (0 = StartUserCore, 1 = UserRequest, 2 = UserApProcedure)
//! - `arg2` (`u64`): Command-specific data pointer
//! - `arg3` (`u64`): Command-specific size or auxiliary data
//!
//! ## Memory Model
//!
//! This crate runs in Ring 3 (user mode). It does not have direct access to supervisor
//! resources. All supervisor services are accessed through syscalls.
//!
//! ## Example
//!
//! ```rust,ignore
//! use patina_mm_user_core::*;
//!
//! struct MyPlatform;
//!
//! impl PlatformInfo for MyPlatform {
//!     const MAX_HANDLERS: usize = 64;
//! }
//!
//! static USER_CORE: MmUserCore<MyPlatform> = MmUserCore::new();
//! ```
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!
#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]
#![cfg(target_arch = "x86_64")]

extern crate alloc;

pub mod config_table;
pub mod core_handlers;
pub mod mm_dispatcher;
pub mod mm_mem;
pub mod mm_services;
pub mod mmi;
pub mod protocol_db;

use core::{
    ffi::c_void,
    mem,
    num::NonZeroUsize,
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use patina::pi::hob::{Hob, PhaseHandoffInformationTable};
use r_efi::efi;
use spin::Once;

use crate::{
    mm_dispatcher::MmDispatcher,
    mmi::MmiDatabase,
    protocol_db::ProtocolDatabase,
};

use core::arch::global_asm;

global_asm!(include_str!("entry_point.asm"));

// =============================================================================
// GUIDs
// =============================================================================

/// GUID used in `MemoryAllocationModule` HOBs to identify MM Supervisor module allocations.
///
/// `gMmSupervisorHobMemoryAllocModuleGuid`
pub const MM_SUPERVISOR_HOB_MEMORY_ALLOC_MODULE_GUID: efi::Guid = efi::Guid::from_fields(
    0x3efafe72,
    0x3dbf,
    0x4341,
    0xad,
    0x04,
    &[0x1c, 0xb6, 0xe8, 0xb6, 0x8e, 0x5e],
);

/// GUID identifying the MM User Core module itself.
///
/// `gMmSupervisorUserGuid`
pub const MM_SUPERVISOR_USER_GUID: efi::Guid = efi::Guid::from_fields(
    0x30d1cc3f,
    0xc1db,
    0x41ed,
    0xb1,
    0x13,
    &[0xab, 0xce, 0x21, 0xb0, 0x2b, 0xce],
);

/// GUID identifying the MM Supervisor Core module (to be skipped during driver discovery).
///
/// `gMmSupervisorCoreGuid`
pub const MM_SUPERVISOR_CORE_GUID: efi::Guid = efi::Guid::from_fields(
    0x4e4c89dc,
    0xa452,
    0x4b6b,
    0xb1,
    0x83,
    &[0xf1, 0x6a, 0x2a, 0x22, 0x37, 0x33],
);

/// GUID for depex data HOBs paired with driver `MemoryAllocationModule` HOBs.
///
/// `gMmSupervisorDepexHobGuid`
pub const MM_SUPERVISOR_DEPEX_HOB_GUID: efi::Guid = efi::Guid::from_fields(
    0xb17f0049,
    0xaffd,
    0x4530,
    0xac,
    0xd6,
    &[0xe2, 0x45, 0xe1, 0x9d, 0xea, 0xf1],
);

/// Mirrors the MM_SUPV_DEPEX_HOB_DATA structure defined in the supervisor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DepexHobData {
    pub name: efi::Guid, // Protocol GUID
    pub depex_expression_size: u64,
    pub depex_expression: [u8; 0],
}

/// GUID for the MM communication buffer HOB.
///
/// `gMmCommBufferHobGuid`
pub use patina_internal_mm_common::MM_COMM_BUFFER_HOB_GUID;



// Re-export shared MM types from the common crate.
pub use patina_internal_mm_common::{
    EfiMmEntryContext, MmCommBufferStatus, EfiMmCommunicateHeader,
    MmCommonBufferHobData, UserCommandType,
};

// =============================================================================
// PlatformInfo Trait
// =============================================================================

/// Platform configuration trait for the MM User Core.
///
/// Platforms implement this trait to provide compile-time constants that
/// configure the user core's internal data structures.
pub trait PlatformInfo: 'static {
    /// Maximum number of MMI handlers that can be registered.
    const MAX_HANDLERS: usize;
}

// =============================================================================
// Communication Buffer Tracking
// =============================================================================

/// Base address of the user communication buffer (discovered from HOBs).
///
/// The supervisor rewrites the HOB's `physical_start` to point to the internal
/// (MMRAM-resident, user-accessible) copy of the communication buffer before
/// invoking `StartUserCore`.
static COMM_BUFFER_BASE: AtomicU64 = AtomicU64::new(0);

/// Size in bytes of the user communication buffer.
static COMM_BUFFER_SIZE: AtomicU64 = AtomicU64::new(0);

// =============================================================================
// MmUserCore
// =============================================================================

/// Static reference to the user core instance.
static __USER_CORE: Once<NonZeroUsize> = Once::new();

/// Useful for offline inspection (like debugging) to determine core version.
#[used]
static MM_USER_CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The MM User Core responsible for driver dispatch and MMI handling in user mode.
///
/// This struct is generic over [`PlatformInfo`], which provides platform-specific
/// compile-time constants. Create a static instance and call [`rust_main`](MmUserCore::rust_main)
/// from the binary entry point.
///
/// ## Example
///
/// ```rust,ignore
/// static USER_CORE: MmUserCore<MyPlatform> = MmUserCore::new();
///
/// #[unsafe(export_name = "efi_main")]
/// pub extern "efiapi" fn _start(arg1: u64, arg2: u64, arg3: u64) -> u64 {
///     USER_CORE.rust_main(arg1, arg2, arg3)
/// }
/// ```
pub struct MmUserCore<P: PlatformInfo> {
    /// The MMI handler database.
    pub mmi_db: MmiDatabase,
    /// The protocol/handle database (for depex evaluation and driver services).
    pub protocol_db: ProtocolDatabase,
    /// The driver dispatcher.
    pub dispatcher: MmDispatcher,
    /// Whether the core has completed initialization.
    initialized: AtomicBool,
    /// Phantom data for the platform type.
    _phantom: core::marker::PhantomData<P>,
}

// SAFETY: MmUserCore is designed to be shared across threads with proper synchronization.
unsafe impl<P: PlatformInfo> Send for MmUserCore<P> {}
unsafe impl<P: PlatformInfo> Sync for MmUserCore<P> {}

impl<P: PlatformInfo> MmUserCore<P> {
    /// Creates a new instance of the MM User Core.
    pub const fn new() -> Self {
        Self {
            mmi_db: MmiDatabase::new(),
            protocol_db: ProtocolDatabase::new(),
            dispatcher: MmDispatcher::new(),
            initialized: AtomicBool::new(false),
            _phantom: core::marker::PhantomData,
        }
    }

    /// Sets the static user core instance for global access.
    ///
    /// Returns true if the address was successfully stored, false if already set.
    #[must_use]
    fn set_instance(&'static self) -> bool {
        let physical_address = NonNull::from_ref(self).expose_provenance();
        &physical_address == __USER_CORE.call_once(|| physical_address)
    }

    /// Gets the static MM User Core instance for global access.
    #[allow(unused)]
    pub fn instance<'a>() -> &'a Self {
        // SAFETY: The pointer is guaranteed to be valid as set_instance ensures single initialization.
        unsafe {
            NonNull::<Self>::with_exposed_provenance(
                *__USER_CORE.get().expect("MM User Core is not initialized."),
            )
            .as_ref()
        }
    }

    /// Main entry point for the MM User Core.
    ///
    /// This is called by the supervisor via `invoke_demoted_routine`. The arguments
    /// correspond to the three parameters passed by the supervisor:
    ///
    /// - `arg1`: Command type ([`UserCommandType`])
    /// - `arg2`: Command-specific data pointer
    /// - `arg3`: Command-specific size or auxiliary data
    ///
    /// Returns 0 on success, or a non-zero status on failure.
    pub fn entry_point_worker(&'static self, op_code: u64, arg1: u64, arg2: u64) -> u64 {
        let command = match UserCommandType::try_from(op_code) {
            Ok(cmd) => cmd,
            Err(unknown) => {
                log::error!("Unknown command type: {}", unknown);
                return efi::Status::INVALID_PARAMETER.as_usize() as u64;
            }
        };

        match command {
            UserCommandType::StartUserCore => {
                self.handle_start_user_core(arg1 as *const c_void)
            }
            UserCommandType::UserRequest => {
                self.handle_user_request(arg1, arg2)
            }
            UserCommandType::UserApProcedure => {
                self.handle_user_ap_procedure(arg1, arg2)
            }
        }
    }

    /// Handle the `StartUserCore` command.
    ///
    /// This is called once during initialization. The supervisor passes the HOB list
    /// pointer as `arg2`. We:
    /// 1. Set the static instance
    /// 2. Walk HOBs to discover the communication buffer
    /// 3. Walk HOBs to discover MM drivers (MemoryAllocationModule HOBs)
    /// 4. Read paired depex GuidHobs for each driver
    /// 5. Evaluate dependency expressions and dispatch drivers in order
    fn handle_start_user_core(&'static self, hob_list: *const c_void) -> u64 {
        if !self.set_instance() {
            log::warn!("MM User Core instance was already set, skipping re-initialization.");
            return efi::Status::ALREADY_STARTED.as_usize() as u64;
        }

        if hob_list.is_null() {
            log::error!("HOB list pointer is null.");
            return efi::Status::INVALID_PARAMETER.as_usize() as u64;
        }

        log::info!("MM User Core v{} starting initialization...", env!("CARGO_PKG_VERSION"));

        // Enable the heap (syscall page allocator) before doing anything that
        // requires dynamic allocation (driver discovery, depex parsing, etc.).
        mm_mem::SYSCALL_PAGE_ALLOCATOR.set_initialized();

        // Parse the HOB list
        let hob_list_info = unsafe {
            match (hob_list as *const PhaseHandoffInformationTable).as_ref() {
                Some(info) => info,
                None => {
                    log::error!("Failed to read HOB list header.");
                    return efi::Status::INVALID_PARAMETER.as_usize() as u64;
                }
            }
        };

        let hob = Hob::Handoff(hob_list_info);

        // Discover communication buffer from HOBs
        self.discover_comm_buffer(&hob);

        // Initialize the MM System Table (heap-allocated, function pointers
        // route to the global protocol DB and MMI DB in mm_services).
        let mm_system_table = mm_services::init_mm_system_table();
        log::info!("MM System Table initialized at {:p}", mm_system_table);

        // Publish the HOB list as a configuration table entry so dispatched
        // drivers can locate it via the system table (mirrors the C
        // `MmInstallConfigurationTable(&gMmCoreMmst, &gEfiHobListGuid, ...)`
        // call in `InitializeMmHobList`).
        let status = config_table::GLOBAL_CONFIG_TABLE_DB.install_configuration_table(
            &patina::guids::HOB_LIST,
            hob_list as *mut c_void,
        );
        if status != efi::Status::SUCCESS {
            log::error!("Failed to install HOB list configuration table: {:?}", status);
        }

        // Discover and dispatch MM drivers from HOBs
        let dispatch_result = self.dispatcher.discover_and_dispatch_drivers(
            &hob,
            &self.mmi_db,
            &self.protocol_db,
            mm_system_table as *const _ as *const core::ffi::c_void,
        );

        match dispatch_result {
            Ok(count) => {
                log::info!("Successfully dispatched {} MM driver(s).", count);
            }
            Err(status) => {
                log::error!("Driver dispatch failed: {:?}", status);
                return status.as_usize() as u64;
            }
        }

        // Register core MMI handlers (lifecycle events like ready-to-lock,
        // end-of-DXE, exit-boot-services, etc.).  Matches the C ordering
        // where handlers are registered after `MmDispatchFvs()`.
        core_handlers::register_core_mmi_handlers();

        self.initialized.store(true, Ordering::Release);
        log::info!("MM User Core initialization complete.");

        efi::Status::SUCCESS.as_usize() as u64
    }

    /// Handle the `UserRequest` command (runtime MMI dispatch).
    ///
    /// The supervisor passes a pointer to a buffer containing:
    /// - `EfiMmEntryContext` (at offset 0)
    /// - `MmCommBufferStatus` (at offset `context_size`)
    ///
    /// For synchronous MMIs the supervisor has already copied the external
    /// communication buffer into an internal (user-accessible) region.  We:
    /// 1. Validate the buffer via the `MmIsCommBuffer` syscall
    /// 2. Parse the `EfiMmCommunicateHeader` to extract the handler GUID and data
    /// 3. Dispatch via `mmi_manage` with the GUID and data pointer
    ///
    /// Asynchronous MMIs (timer, etc.) are always dispatched as root-only
    /// (`mmi_manage(None, …)`).
    ///
    /// Mirrors the C `MmEntryPoint` flow in `StandaloneMmCore.c`.
    fn handle_user_request(&self, supv_to_user_buffer: u64, context_size: u64) -> u64 {
        if supv_to_user_buffer == 0 {
            log::error!("Supervisor-to-user buffer is null.");
            return efi::Status::INVALID_PARAMETER.as_usize() as u64;
        }

        // Read the EfiMmEntryContext
        let _entry_context = unsafe {
            core::ptr::read(supv_to_user_buffer as *const EfiMmEntryContext)
        };

        // Read MmCommBufferStatus (immediately after the context)
        let comm_status = unsafe {
            core::ptr::read(
                (supv_to_user_buffer as *const u8).add(context_size as usize)
                    as *const MmCommBufferStatus,
            )
        };

        // ---- Synchronous MMI dispatch ----
        let mut sync_status = efi::Status::NOT_FOUND;
        let mut return_buffer_size: u64 = 0;

        let comm_buffer_base = COMM_BUFFER_BASE.load(Ordering::Acquire);
        let comm_buffer_size = COMM_BUFFER_SIZE.load(Ordering::Acquire);

        if comm_buffer_base != 0 && comm_status.is_comm_buffer_valid != 0 {
            // Validate the communication buffer via a supervisor syscall.
            if !mm_mem::is_comm_buffer(comm_buffer_base, comm_buffer_size) {
                log::error!(
                    "MmIsCommBuffer rejected buffer at 0x{:x} size 0x{:x}",
                    comm_buffer_base,
                    comm_buffer_size
                );
            } else {
                sync_status = self.dispatch_synchronous_mmi(
                    comm_buffer_base,
                    comm_buffer_size,
                    &mut return_buffer_size,
                );
            }
        }

        // ---- Asynchronous MMI dispatch (always runs) ----
        mm_services::GLOBAL_MMI_DB.mmi_manage(
            None,
            core::ptr::null(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );

        // Write back the updated status to the supervisor-to-user buffer
        let updated_status = MmCommBufferStatus {
            is_comm_buffer_valid: 0,
            talk_to_supervisor: 0,
            return_status: if sync_status == efi::Status::SUCCESS {
                efi::Status::SUCCESS.as_usize() as u64
            } else {
                efi::Status::NOT_FOUND.as_usize() as u64
            },
            return_buffer_size,
        };

        unsafe {
            core::ptr::write(
                (supv_to_user_buffer as *mut u8).add(context_size as usize)
                    as *mut MmCommBufferStatus,
                updated_status,
            );
        }

        efi::Status::SUCCESS.as_usize() as u64
    }

    /// Parse the `EfiMmCommunicateHeader` from the communication buffer and
    /// dispatch the appropriate GUID-specific MMI handler.
    ///
    /// Returns the dispatch status and updates `return_buffer_size` with the
    /// total response size (header + data).
    fn dispatch_synchronous_mmi(
        &self,
        comm_buffer_base: u64,
        comm_buffer_size: u64,
        return_buffer_size: &mut u64,
    ) -> efi::Status {
        let buffer_size = comm_buffer_size as usize;

        // The buffer must be large enough for at least the communicate header.
        if buffer_size < EfiMmCommunicateHeader::HEADER_SIZE {
            log::error!(
                "Communication buffer too small for header: {} < {}",
                buffer_size,
                EfiMmCommunicateHeader::HEADER_SIZE
            );
            return efi::Status::BAD_BUFFER_SIZE;
        }

        // SAFETY: We verified the buffer is large enough for the header.
        let header = unsafe {
            core::ptr::read_unaligned(comm_buffer_base as *const EfiMmCommunicateHeader)
        };

        // Determine header layout: check for V3 signature first, then fall
        // back to the legacy `EfiMmCommunicateHeader`.
        let (comm_guid_ptr, comm_header_size, mut data_size) =
            if header.header_guid == patina::pi::protocols::communication3::COMMUNICATE_HEADER_V3_GUID {
                // V3 header
                let v3 = unsafe {
                    core::ptr::read_unaligned(
                        comm_buffer_base as *const patina::pi::protocols::communication3::EfiMmCommunicateHeader,
                    )
                };
                let header_size = mem::size_of::<patina::pi::protocols::communication3::EfiMmCommunicateHeader>();
                let total = v3.buffer_size as usize;
                if total > buffer_size {
                    log::error!(
                        "V3 buffer_size 0x{:x} exceeds available 0x{:x}",
                        total,
                        buffer_size
                    );
                    return efi::Status::BAD_BUFFER_SIZE;
                }
                // GUID to dispatch is `message_guid` in V3
                let guid_offset = core::mem::offset_of!(
                    patina::pi::protocols::communication3::EfiMmCommunicateHeader,
                    message_guid
                );
                let guid_ptr = (comm_buffer_base as *const u8).wrapping_add(guid_offset) as *const efi::Guid;
                (guid_ptr, header_size, total.saturating_sub(header_size))
            } else {
                // Legacy header
                let message_length = header.message_length as usize;
                let total = EfiMmCommunicateHeader::HEADER_SIZE + message_length;
                if total > buffer_size {
                    log::error!(
                        "Legacy message_length 0x{:x} exceeds available 0x{:x}",
                        message_length,
                        buffer_size.saturating_sub(EfiMmCommunicateHeader::HEADER_SIZE)
                    );
                    return efi::Status::BAD_BUFFER_SIZE;
                }
                // GUID to dispatch is `header_guid` in legacy
                let guid_ptr = comm_buffer_base as *const efi::Guid;
                (guid_ptr, EfiMmCommunicateHeader::HEADER_SIZE, message_length)
            };

        // Zero the remainder of the buffer past the message (matches C behaviour).
        let used = comm_header_size + data_size;
        if used < buffer_size {
            unsafe {
                core::ptr::write_bytes(
                    (comm_buffer_base as *mut u8).add(used),
                    0,
                    buffer_size - used,
                );
            }
        }

        // Dispatch the GUID-specific handler.
        let comm_data_ptr = unsafe {
            (comm_buffer_base as *mut u8).add(comm_header_size) as *mut c_void
        };

        let status = mm_services::GLOBAL_MMI_DB.mmi_manage(
            Some(unsafe { &*comm_guid_ptr }),
            core::ptr::null(),
            comm_data_ptr,
            &mut data_size as *mut usize,
        );

        *return_buffer_size = (data_size + comm_header_size) as u64;
        status
    }

    /// Handle the `UserApProcedure` command.
    ///
    /// The supervisor passes the procedure pointer and argument. We call the procedure
    /// directly since we're already in user mode.
    fn handle_user_ap_procedure(&self, procedure: u64, argument: u64) -> u64 {
        if procedure == 0 {
            log::error!("AP procedure pointer is null.");
            return efi::Status::INVALID_PARAMETER.as_usize() as u64;
        }

        log::trace!("Executing AP procedure at 0x{:016x} with arg 0x{:016x}", procedure, argument);

        // SAFETY: The supervisor has validated the procedure pointer before dispatching.
        // The procedure follows the EFI AP_PROCEDURE calling convention.
        type EfiApProcedure = unsafe extern "efiapi" fn(*mut c_void);
        let proc_fn: EfiApProcedure = unsafe { core::mem::transmute(procedure) };
        unsafe { proc_fn(argument as *mut c_void) };

        efi::Status::SUCCESS.as_usize() as u64
    }

    /// Discover the communication buffer address from HOBs and store it for
    /// later use in `handle_user_request`.
    ///
    /// The supervisor rewrites the HOB's `physical_start` field to point to
    /// the internal (user-accessible) copy of the buffer before invoking
    /// `StartUserCore`, so the address we read here is the one we should
    /// read from at runtime.
    fn discover_comm_buffer(&self, hob: &Hob<'_>) {
        for current_hob in hob {
            if let Hob::GuidHob(guid_hob, data) = current_hob {
                if guid_hob.name == MM_COMM_BUFFER_HOB_GUID {
                    if data.len() >= mem::size_of::<MmCommonBufferHobData>() {
                        let buffer_data =
                            unsafe { &*(data.as_ptr() as *const MmCommonBufferHobData) };
                        let physical_start =
                            unsafe { core::ptr::addr_of!(buffer_data.physical_start).read_unaligned() };
                        let number_of_pages =
                            unsafe { core::ptr::addr_of!(buffer_data.number_of_pages).read_unaligned() };

                        let buffer_size = number_of_pages.saturating_mul(4096);

                        COMM_BUFFER_BASE.store(physical_start, Ordering::Release);
                        COMM_BUFFER_SIZE.store(buffer_size, Ordering::Release);

                        log::info!(
                            "Found MM communication buffer: base=0x{:016x}, pages={}, size=0x{:x}",
                            physical_start,
                            number_of_pages,
                            buffer_size,
                        );
                        return;
                    }
                }
            }
        }

        log::warn!("No MM communication buffer HOB found — only root MMI handlers will be supported.");
    }
}
