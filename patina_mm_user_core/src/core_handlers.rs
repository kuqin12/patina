//! MM Core Internal MMI Handlers
//!
//! These are the MMI handlers registered by the MM Core itself to handle
//! lifecycle events forwarded from the DXE phase. They mirror the C
//! `mMmCoreMmiHandlers[]` table in `StandaloneMmCore.c`.
//!
//! Each handler is registered with [`MmiDatabase::register_internal_handler`]
//! during startup and dispatched when the supervisor forwards the corresponding
//! GUID-tagged MMI through the communication buffer.
//!
//! ## Lifecycle Events
//!
//! | GUID | Handler | Description |
//! |------|---------|-------------|
//! | `MM_DISPATCH_EVENT` | [`mm_driver_dispatch_handler`] | Re-triggers driver dispatch |
//! | `MM_DXE_READY_TO_LOCK_PROTOCOL` | [`mm_ready_to_lock_handler`] | Unregisters one-shot handlers, installs lock protocol |
//! | `MM_END_OF_PEI_PROTOCOL` | [`mm_end_of_pei_handler`] | Installs end-of-PEI protocol |
//! | `EVENT_GROUP_END_OF_DXE` | [`mm_end_of_dxe_handler`] | Installs end-of-DXE protocol |
//! | `EVENT_EXIT_BOOT_SERVICES` | [`mm_exit_boot_service_handler`] | Installs exit-boot-services protocol |
//! | `EVENT_READY_TO_BOOT` | [`mm_ready_to_boot_handler`] | Installs ready-to-boot protocol |
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use core::ffi::c_void;

use r_efi::efi;
use spin::Mutex;

use crate::mmi::InternalMmiHandler;
use patina::guids;

// =============================================================================
// Handler table definition
// =============================================================================

/// Description of a core MMI handler to be registered at startup.
struct CoreMmiHandler {
    /// The handler function (native Rust signature).
    handler: InternalMmiHandler,
    /// The GUID that triggers this handler.
    handler_type: &'static efi::Guid,
    /// Whether this handler should be unregistered during ready-to-lock.
    unregister_on_lock: bool,
}

/// Table of MMI handlers registered by the MM Core, mirroring the C `mMmCoreMmiHandlers[]`.
static CORE_MMI_HANDLERS: &[CoreMmiHandler] = &[
    CoreMmiHandler {
        handler: mm_driver_dispatch_handler,
        handler_type: &guids::MM_DISPATCH_EVENT,
        unregister_on_lock: false,
    },
    CoreMmiHandler {
        handler: mm_ready_to_lock_handler,
        handler_type: &guids::MM_DXE_READY_TO_LOCK_PROTOCOL,
        unregister_on_lock: true,
    },
    CoreMmiHandler {
        handler: mm_end_of_pei_handler,
        handler_type: &guids::MM_END_OF_PEI_PROTOCOL,
        unregister_on_lock: false,
    },
    CoreMmiHandler {
        handler: mm_end_of_dxe_handler,
        handler_type: &guids::EVENT_GROUP_END_OF_DXE,
        unregister_on_lock: false,
    },
    CoreMmiHandler {
        handler: mm_exit_boot_service_handler,
        handler_type: &guids::EVENT_EXIT_BOOT_SERVICES,
        unregister_on_lock: false,
    },
    CoreMmiHandler {
        handler: mm_ready_to_boot_handler,
        handler_type: &guids::EVENT_READY_TO_BOOT,
        unregister_on_lock: false,
    },
];

// =============================================================================
// Dispatch handle tracking
// =============================================================================

/// Newtype wrapper around `efi::Handle` so it can be stored in a `static Mutex`.
///
/// `efi::Handle` is `*mut c_void` which is `!Send`.  The dispatch handles are only
/// written by the BSP during single-threaded init and read during the ready-to-lock
/// handler (also on the BSP), so it is safe to share them.
#[derive(Clone, Copy)]
struct SendHandle(efi::Handle);
unsafe impl Send for SendHandle {}
unsafe impl Sync for SendHandle {}

impl SendHandle {
    const NULL: Self = Self(core::ptr::null_mut());
}

/// Dispatch handles returned from `register_internal_handler` for each core handler.
///
/// Index matches the `CORE_MMI_HANDLERS` table. Populated by [`register_core_mmi_handlers`].
static DISPATCH_HANDLES: Mutex<[SendHandle; 6]> = Mutex::new([SendHandle::NULL; 6]);

// =============================================================================
// Public API
// =============================================================================

/// Register all core MMI handlers with the global MMI database.
///
/// This must be called after driver dispatch (matching the C `StandaloneMmMain`
/// ordering where handlers are registered after `MmDispatchFvs`).
pub fn register_core_mmi_handlers() {
    let mut handles = DISPATCH_HANDLES.lock();

    for (i, entry) in CORE_MMI_HANDLERS.iter().enumerate() {
        match crate::mm_services::GLOBAL_MMI_DB.register_internal_handler(
            entry.handler,
            Some(entry.handler_type),
        ) {
            Ok(handle) => {
                handles[i] = SendHandle(handle);
                log::info!(
                    "Registered core MMI handler [{}] for {:?}",
                    i,
                    entry.handler_type,
                );
            }
            Err(status) => {
                log::error!(
                    "Failed to register core MMI handler [{}] for {:?}: {:?}",
                    i,
                    entry.handler_type,
                    status,
                );
            }
        }
    }
}

// =============================================================================
// Install-protocol helper
// =============================================================================

/// Install a protocol with a NULL interface on a new handle.
///
/// This mirrors the C pattern used in lifecycle handlers:
/// ```c
/// MmHandle = NULL;
/// Status = MmInstallProtocolInterface(&MmHandle, &guid, EFI_NATIVE_INTERFACE, NULL);
/// ```
fn install_lifecycle_protocol(guid: &efi::Guid) -> efi::Status {
    let (handle, pending_notifies) =
        match crate::mm_services::GLOBAL_PROTOCOL_DB.install_protocol(
            core::ptr::null_mut(), // NULL handle → allocate new
            guid,
            core::ptr::null_mut(), // NULL interface
        ) {
            Ok(result) => result,
            Err(status) => return status,
        };

    log::info!(
        "Installed lifecycle protocol {:?} on handle {:p}",
        guid,
        handle,
    );

    // Fire pending notify callbacks outside the protocol DB lock.
    for notify in pending_notifies {
        unsafe {
            (notify.function)(
                &notify.guid as *const efi::Guid,
                notify.interface,
                notify.handle,
            );
        }
    }

    efi::Status::SUCCESS
}

// =============================================================================
// Handler implementations
// =============================================================================

/// MM Driver Dispatch Handler.
///
/// Re-triggers driver dispatch for any previously discovered but not-yet-dispatched
/// drivers. Once dispatch completes, the handler unregisters itself (it is a
/// one-shot handler).
///
/// Corresponds to the C `MmDriverDispatchHandler`.
fn mm_driver_dispatch_handler(
    _handler_type: &efi::Guid,
    _comm_buffer: *mut c_void,
    _comm_buffer_size: *mut usize,
) -> efi::Status {
    log::info!("MmDriverDispatchHandler");

    // TODO: Re-dispatch any remaining undispatched drivers.
    // Currently all drivers are dispatched during StartUserCore, so this is a no-op.

    // Self-unregister (one-shot).
    let handles = DISPATCH_HANDLES.lock();
    let dispatch_handle = handles[0].0;
    drop(handles);

    if !dispatch_handle.is_null() {
        let _ = crate::mm_services::GLOBAL_MMI_DB.mmi_handler_unregister(dispatch_handle);
    }

    log::info!("MmDriverDispatchHandler done");

    efi::Status::SUCCESS
}

/// MM Ready To Lock Handler.
///
/// Called when `gEfiDxeMmReadyToLockProtocolGuid` MMI is received. This:
/// 1. Unregisters handlers marked with `unregister_on_lock` (including itself).
/// 2. Installs the `gEfiMmReadyToLockProtocolGuid` protocol to notify MM drivers.
///
/// Corresponds to the C `MmReadyToLockHandler`.
fn mm_ready_to_lock_handler(
    _handler_type: &efi::Guid,
    _comm_buffer: *mut c_void,
    _comm_buffer_size: *mut usize,
) -> efi::Status {
    log::info!("MmReadyToLockHandler");

    // Unregister handlers that are no longer needed after MM lock.
    let handles = DISPATCH_HANDLES.lock();
    for (i, entry) in CORE_MMI_HANDLERS.iter().enumerate() {
        if entry.unregister_on_lock && !handles[i].0.is_null() {
            let _ = crate::mm_services::GLOBAL_MMI_DB.mmi_handler_unregister(handles[i].0);
        }
    }
    drop(handles);

    // Install the MM Ready To Lock Protocol.
    let status = install_lifecycle_protocol(&guids::MM_READY_TO_LOCK_PROTOCOL);
    if status != efi::Status::SUCCESS {
        log::error!("Failed to install MM Ready To Lock Protocol: {:?}", status);
    }

    status
}

/// MM End of PEI Handler.
///
/// Installs the `gEfiMmEndOfPeiProtocol` protocol.
///
/// Corresponds to the C `MmEndOfPeiHandler`.
fn mm_end_of_pei_handler(
    _handler_type: &efi::Guid,
    _comm_buffer: *mut c_void,
    _comm_buffer_size: *mut usize,
) -> efi::Status {
    log::info!("MmEndOfPeiHandler");

    install_lifecycle_protocol(&guids::MM_END_OF_PEI_PROTOCOL)
}

/// MM End of DXE Handler.
///
/// Installs the `gEfiMmEndOfDxeProtocolGuid` protocol.
///
/// Corresponds to the C `MmEndOfDxeHandler`.
fn mm_end_of_dxe_handler(
    _handler_type: &efi::Guid,
    _comm_buffer: *mut c_void,
    _comm_buffer_size: *mut usize,
) -> efi::Status {
    log::info!("MmEndOfDxeHandler");

    install_lifecycle_protocol(&guids::MM_END_OF_DXE_PROTOCOL)
}

/// MM Exit Boot Service Handler.
///
/// Installs the `gEfiEventExitBootServicesGuid` protocol (once).
///
/// Corresponds to the C `MmExitBootServiceHandler`.
fn mm_exit_boot_service_handler(
    _handler_type: &efi::Guid,
    _comm_buffer: *mut c_void,
    _comm_buffer_size: *mut usize,
) -> efi::Status {
    static FIRED: spin::Once<()> = spin::Once::new();
    let mut status = efi::Status::SUCCESS;

    FIRED.call_once(|| {
        status = install_lifecycle_protocol(&guids::EVENT_EXIT_BOOT_SERVICES);
    });

    status
}

/// MM Ready To Boot Handler.
///
/// Installs the `gEfiEventReadyToBootGuid` protocol (once).
///
/// Corresponds to the C `MmReadyToBootHandler`.
fn mm_ready_to_boot_handler(
    _handler_type: &efi::Guid,
    _comm_buffer: *mut c_void,
    _comm_buffer_size: *mut usize,
) -> efi::Status {
    static FIRED: spin::Once<()> = spin::Once::new();
    let mut status = efi::Status::SUCCESS;

    FIRED.call_once(|| {
        status = install_lifecycle_protocol(&guids::EVENT_READY_TO_BOOT);
    });

    status
}
