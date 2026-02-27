//! MMI (Management Mode Interrupt) Handler Database
//!
//! This module manages the registration and dispatch of MMI handlers, following the
//! same patterns as the C `Mmi.c` in `StandaloneMmPkg/Core`.
//!
//! ## Handler Types
//!
//! - **Root handlers**: Registered with `handler_type = None`. Called on every MMI regardless
//!   of the communication buffer contents. Used for hardware-level interrupt sources.
//! - **GUID-specific handlers**: Registered with a specific GUID. Called only when an MMI
//!   communication targets that GUID.
//!
//! ## Dispatch Flow
//!
//! [`MmiDatabase::mmi_manage`] is the main dispatch entry point:
//! 1. If `handler_type` is `None`, iterate root handlers
//! 2. If `handler_type` is `Some(guid)`, find the `MmiEntry` for that GUID and iterate its handlers
//! 3. Each handler returns a status that determines whether dispatch continues
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use alloc::{vec, vec::Vec};
use core::ffi::c_void;

use r_efi::efi;
use spin::Mutex;

/// EFI_WARN_INTERRUPT_SOURCE_QUIESCED — PI spec warning status code.
/// Indicates an interrupt source was quiesced.
const WARN_INTERRUPT_SOURCE_QUIESCED: efi::Status = efi::Status::from_usize(3);

/// EFI_INTERRUPT_PENDING — PI spec status for pending interrupts.
const INTERRUPT_PENDING: efi::Status = efi::Status::from_usize(0x80000000 | 0x00000004);

// =============================================================================
// Handler Types
// =============================================================================

/// MMI handler entry point signature.
///
/// Re-exported from [`patina::mm_services::MmiHandlerEntryPoint`].
pub use patina::mm_services::MmiHandlerEntryPoint;

/// An MMI entry groups all handlers registered for a specific GUID.
struct MmiEntry {
    /// The handler type GUID.
    handler_type: efi::Guid,
    /// All handlers registered for this GUID.
    handlers: Vec<MmiHandler>,
}

/// A registered MMI handler.
struct MmiHandler {
    /// The handler function.
    handler: MmiHandlerEntryPoint,
    /// Whether this handler is marked for removal (deferred removal during dispatch).
    to_remove: bool,
}

// =============================================================================
// MMI Database
// =============================================================================

/// The MMI handler database.
///
/// Manages root handlers (called for all MMIs) and GUID-specific handlers.
/// Thread-safe via internal `Mutex`.
pub struct MmiDatabase {
    /// Internal state protected by a mutex.
    inner: Mutex<MmiDatabaseInner>,
}

struct MmiDatabaseInner {
    /// Root MMI handlers (called for every MMI, regardless of GUID).
    root_handlers: Vec<MmiHandler>,
    /// GUID-specific MMI entries.
    entries: Vec<MmiEntry>,
    /// Re-entrance depth counter for `mmi_manage`.
    manage_calling_depth: usize,
}

impl MmiDatabase {
    /// Creates a new empty `MmiDatabase`.
    pub const fn new() -> Self {
        Self {
            inner: Mutex::new(MmiDatabaseInner {
                root_handlers: Vec::new(),
                entries: Vec::new(),
                manage_calling_depth: 0,
            }),
        }
    }

    /// Register an MMI handler.
    ///
    /// If `handler_type` is `None`, the handler is registered as a root handler.
    /// If `handler_type` is `Some(guid)`, the handler is registered for that specific GUID.
    ///
    /// Returns `Ok(dispatch_handle)` on success, where `dispatch_handle` is an opaque handle
    /// that can be used to unregister the handler.
    pub fn mmi_handler_register(
        &self,
        handler: MmiHandlerEntryPoint,
        handler_type: Option<&efi::Guid>,
    ) -> Result<efi::Handle, efi::Status> {
        let mut inner = self.inner.lock();

        match handler_type {
            None => {
                // Root handler
                inner.root_handlers.push(MmiHandler {
                    handler,
                    to_remove: false,
                });
                // Return a pseudo-handle (pointer to the handler fn as handle)
                let handle = handler as *const () as efi::Handle;
                log::debug!("Registered root MMI handler: {:?}", handle);
                Ok(handle)
            }
            Some(guid) => {
                // Find or create the MMI entry for this GUID
                let entry = inner
                    .entries
                    .iter_mut()
                    .find(|e| e.handler_type == *guid);

                match entry {
                    Some(entry) => {
                        entry.handlers.push(MmiHandler {
                            handler,
                            to_remove: false,
                        });
                    }
                    None => {
                        inner.entries.push(MmiEntry {
                            handler_type: *guid,
                            handlers: vec![MmiHandler {
                                handler,
                                to_remove: false,
                            }],
                        });
                    }
                }

                let handle = handler as *const () as efi::Handle;
                log::debug!("Registered MMI handler for {:?}: {:?}", guid, handle);
                Ok(handle)
            }
        }
    }

    /// Unregister an MMI handler.
    ///
    /// Marks the handler for deferred removal. The actual removal happens after
    /// `mmi_manage` completes if we're currently inside a dispatch.
    pub fn mmi_handler_unregister(
        &self,
        dispatch_handle: efi::Handle,
    ) -> Result<(), efi::Status> {
        let mut inner = self.inner.lock();

        // Search root handlers
        for handler in inner.root_handlers.iter_mut() {
            if handler.handler as *const () as efi::Handle == dispatch_handle {
                handler.to_remove = true;
                log::debug!("Marked root MMI handler {:?} for removal.", dispatch_handle);
                return Ok(());
            }
        }

        // Search GUID-specific handlers
        for entry in inner.entries.iter_mut() {
            for handler in entry.handlers.iter_mut() {
                if handler.handler as *const () as efi::Handle == dispatch_handle {
                    handler.to_remove = true;
                    log::debug!(
                        "Marked MMI handler {:?} for removal (GUID: {:?}).",
                        dispatch_handle,
                        entry.handler_type,
                    );
                    return Ok(());
                }
            }
        }

        log::warn!("MMI handler {:?} not found for unregistration.", dispatch_handle);
        Err(efi::Status::NOT_FOUND)
    }

    /// Manage (dispatch) an MMI.
    ///
    /// This is the main dispatch function, equivalent to the C `MmiManage`.
    ///
    /// - If `handler_type` is `None`, root handlers are dispatched.
    /// - If `handler_type` is `Some(guid)`, the handlers for that GUID are dispatched.
    ///
    /// Returns:
    /// - `EFI_SUCCESS` if at least one handler returned success
    /// - `EFI_WARN_INTERRUPT_SOURCE_PENDING` if an interrupt source was processed but not quiesced
    /// - `EFI_INTERRUPT_PENDING` if a handler indicated the interrupt is still pending
    /// - `EFI_NOT_FOUND` if no handlers are registered for the given type
    pub fn mmi_manage(
        &self,
        handler_type: Option<&efi::Guid>,
        context: *const c_void,
        comm_buffer: *mut c_void,
        comm_buffer_size: *mut usize,
    ) -> efi::Status {
        let mut inner = self.inner.lock();
        inner.manage_calling_depth += 1;

        let mut return_status = efi::Status::NOT_FOUND;

        match handler_type {
            None => {
                // Root MMI handler dispatch
                return_status = Self::dispatch_handlers(
                    &inner.root_handlers,
                    context,
                    comm_buffer,
                    comm_buffer_size,
                    false, // root handlers don't short-circuit
                );
            }
            Some(guid) => {
                // GUID-specific handler dispatch
                if let Some(entry) = inner.entries.iter().find(|e| e.handler_type == *guid) {
                    return_status = Self::dispatch_handlers(
                        &entry.handlers,
                        context,
                        comm_buffer,
                        comm_buffer_size,
                        true, // GUID-specific handlers short-circuit on SUCCESS/INTERRUPT_PENDING
                    );
                }
            }
        }

        inner.manage_calling_depth -= 1;

        // Clean up handlers marked for removal when we're at the outermost dispatch level
        if inner.manage_calling_depth == 0 {
            Self::cleanup_removed_handlers(&mut inner);
        }

        return_status
    }

    /// Dispatch a list of handlers, following the MmiManage status protocol.
    fn dispatch_handlers(
        handlers: &[MmiHandler],
        context: *const c_void,
        comm_buffer: *mut c_void,
        comm_buffer_size: *mut usize,
        short_circuit: bool,
    ) -> efi::Status {
        let mut return_status = efi::Status::NOT_FOUND;

        for handler in handlers {
            if handler.to_remove {
                continue;
            }

            // SAFETY: The handler function pointer was validated at registration time.
            let status = unsafe {
                (handler.handler)(
                    handler.handler as *const () as efi::Handle,
                    context,
                    comm_buffer,
                    comm_buffer_size,
                )
            };

            match status {
                efi::Status::SUCCESS => {
                    return_status = efi::Status::SUCCESS;
                    if short_circuit {
                        break;
                    }
                }
                s if s == INTERRUPT_PENDING => {
                    if short_circuit {
                        return INTERRUPT_PENDING;
                    }
                    if return_status != efi::Status::SUCCESS {
                        return_status = status;
                    }
                }
                s if s == WARN_INTERRUPT_SOURCE_QUIESCED => {
                    return_status = efi::Status::SUCCESS;
                }
                _ => {
                    // Other statuses are ignored per PI spec
                }
            }
        }

        return_status
    }

    /// Remove handlers marked with `to_remove` and clean up empty entries.
    fn cleanup_removed_handlers(inner: &mut MmiDatabaseInner) {
        inner.root_handlers.retain(|h| !h.to_remove);

        inner.entries.retain_mut(|entry| {
            entry.handlers.retain(|h| !h.to_remove);
            !entry.handlers.is_empty()
        });
    }
}
