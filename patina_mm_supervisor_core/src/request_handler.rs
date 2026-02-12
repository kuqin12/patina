//! Request Handler Module
//!
//! This module provides the infrastructure for handling incoming MM supervisor requests.
//! Handlers can be registered to process specific types of requests.
//!
//! ## Memory Model
//!
//! This module does not perform heap allocation. Handlers are stored as static references
//! in fixed-size arrays with compile-time constants provided via const generics.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use core::sync::atomic::{AtomicUsize, Ordering};
use spin::Mutex;
use r_efi::efi;

// Re-export shared protocol types from patina_mm so consumers of this crate get them too.
pub use patina_mm::protocol::mm_supervisor_request::{
    self as mm_supv_protocol,
    MmSupervisorRequestHeader,
    MmSupervisorVersionInfo,
    requests,
    responses,
    SIGNATURE,
    REVISION,
};

/// The result of a request handler invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestResult {
    /// The request was handled successfully.
    Success,
    /// The handler does not handle this type of request.
    NotHandled,
    /// The request was handled but resulted in an error.
    Error(efi::Status),
    /// The request requires deferred processing.
    Deferred,
}

impl From<RequestResult> for efi::Status {
    fn from(result: RequestResult) -> Self {
        match result {
            RequestResult::Success => efi::Status::SUCCESS,
            RequestResult::NotHandled => efi::Status::NOT_FOUND,
            RequestResult::Error(status) => status,
            RequestResult::Deferred => efi::Status::NOT_READY,
        }
    }
}

/// Context information passed to request handlers.
#[derive(Debug)]
pub struct RequestContext {
    /// The GUID identifying the type of request.
    pub handler_guid: efi::Guid,
    /// Pointer to the communication buffer data.
    pub buffer: *mut u8,
    /// Size of the communication buffer.
    pub buffer_size: usize,
    /// Whether this request is from the supervisor channel (vs user channel).
    pub is_supervisor_request: bool,
    /// The CPU ID of the processor handling this request.
    pub cpu_id: u32,
}

impl RequestContext {
    /// Creates a new request context.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `buffer` points to valid memory of at least `buffer_size` bytes
    /// and that the memory remains valid for the lifetime of the context.
    pub unsafe fn new(
        handler_guid: efi::Guid,
        buffer: *mut u8,
        buffer_size: usize,
        is_supervisor_request: bool,
        cpu_id: u32,
    ) -> Self {
        Self {
            handler_guid,
            buffer,
            buffer_size,
            is_supervisor_request,
            cpu_id,
        }
    }

    /// Gets a slice view of the buffer.
    pub unsafe fn as_slice(&self) -> &[u8] {
        if self.buffer.is_null() || self.buffer_size == 0 {
            &[]
        } else {
            //
            // # Safety
            //
            // The caller must ensure the buffer is still valid.
            unsafe {
                core::slice::from_raw_parts(self.buffer, self.buffer_size)
            }
        }
    }

    /// Gets a mutable slice view of the buffer.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.buffer.is_null() || self.buffer_size == 0 {
            &mut []
        } else {
            //
            // # Safety
            //
            // The caller must ensure the buffer is still valid and no other references exist.
            unsafe {
                core::slice::from_raw_parts_mut(self.buffer, self.buffer_size)
            }
        }
    }
}

/// Trait for implementing MM request handlers.
///
/// Handlers are registered with the supervisor core and are invoked when matching
/// requests are received.
///
/// ## Example
///
/// ```rust,ignore
/// use patina_mm_supervisor_core::{RequestHandler, RequestContext, RequestResult};
///
/// struct MyHandler;
///
/// impl RequestHandler for MyHandler {
///     fn guid(&self) -> r_efi::efi::Guid {
///         r_efi::efi::Guid::from_fields(
///             0x12345678, 0x1234, 0x5678,
///             0x12, 0x34, &[0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0]
///         )
///     }
///
///     fn handle(&self, context: &mut RequestContext) -> RequestResult {
///         // Process the request
///         RequestResult::Success
///     }
/// }
///
/// // Create as static and register with supervisor
/// static MY_HANDLER: MyHandler = MyHandler;
/// ```
pub trait RequestHandler: Sync {
    /// Returns the GUID that identifies requests this handler processes.
    fn guid(&self) -> efi::Guid;

    /// Handles a request.
    ///
    /// # Arguments
    ///
    /// * `context` - The request context containing buffer and metadata.
    ///
    /// # Returns
    ///
    /// The result of handling the request.
    fn handle(&self, context: &mut RequestContext) -> RequestResult;

    /// Returns the name of this handler (for debugging/logging).
    fn name(&self) -> &'static str {
        "Unknown"
    }

    /// Returns whether this handler requires supervisor privilege.
    ///
    /// If true, the handler will only be invoked for supervisor-channel requests.
    fn requires_supervisor(&self) -> bool {
        false
    }
}

/// A slot for storing a handler reference.
struct HandlerSlot {
    /// The handler reference stored behind a small spinlock.
    handler: Mutex<Option<&'static dyn RequestHandler>>,
}

impl HandlerSlot {
    /// Creates a new empty handler slot.
    const fn new() -> Self {
        Self {
            handler: Mutex::new(None),
        }
    }

    /// Sets the handler in this slot.
    ///
    /// Returns true if successful, false if slot was already occupied.
    fn set(&self, handler: &'static dyn RequestHandler) -> bool {
        let mut guard = self.handler.lock();
        if guard.is_none() {
            *guard = Some(handler);
            true
        } else {
            false
        }
    }

    /// Gets the handler from this slot.
    fn get(&self) -> Option<&'static dyn RequestHandler> {
        *self.handler.lock()
    }

    /// Checks if this slot is empty.
    fn is_empty(&self) -> bool {
        self.handler.lock().is_none()
    }

    /// Clears the handler from this slot.
    ///
    /// Returns the handler that was in the slot, if any.
    fn clear(&self) -> Option<&'static dyn RequestHandler> {
        self.handler.lock().take()
    }
}

/// A dispatcher for routing requests to appropriate handlers.
///
/// Uses fixed-size arrays with const generic for maximum handler count.
///
/// ## Const Generic Parameters
///
/// * `MAX_HANDLERS` - The maximum number of handlers that can be registered.
pub struct RequestDispatcher<const MAX_HANDLERS: usize> {
    /// Handler slots - fixed size array.
    slots: [HandlerSlot; MAX_HANDLERS],
    /// Number of registered handlers.
    registered_count: AtomicUsize,
}

impl<const MAX_HANDLERS: usize> RequestDispatcher<MAX_HANDLERS> {
    /// Creates a new request dispatcher.
    ///
    /// This is a const fn and performs no heap allocation.
    pub const fn new() -> Self {
        Self {
            slots: [const { HandlerSlot::new() }; MAX_HANDLERS],
            registered_count: AtomicUsize::new(0),
        }
    }

    /// Registers a handler with the dispatcher.
    ///
    /// Returns `true` if the handler was registered, `false` if the handler table is full.
    pub fn register(&self, handler: &'static dyn RequestHandler) -> bool {
        // Find an empty slot
        for slot in &self.slots {
            if slot.set(handler) {
                self.registered_count.fetch_add(1, Ordering::SeqCst);
                log::info!(
                    "Registered request handler '{}' for GUID {:?}",
                    handler.name(),
                    handler.guid()
                );
                return true;
            }
        }

        log::warn!(
            "Failed to register handler '{}': handler table full (max {})",
            handler.name(),
            MAX_HANDLERS
        );
        false
    }

    /// Unregisters a handler by GUID.
    ///
    /// Returns `true` if a handler was removed.
    pub fn unregister(&self, guid: &efi::Guid) -> bool {
        for slot in &self.slots {
            if let Some(handler) = slot.get() {
                if &handler.guid() == guid {
                    slot.clear();
                    self.registered_count.fetch_sub(1, Ordering::SeqCst);
                    return true;
                }
            }
        }
        false
    }

    /// Dispatches a request to the appropriate handler(s).
    ///
    /// Returns the result of the first handler that processes the request.
    pub fn dispatch(&self, context: &mut RequestContext) -> RequestResult {
        for slot in &self.slots {
            if let Some(handler) = slot.get() {
                // Check if this handler matches the request GUID
                if handler.guid() != context.handler_guid {
                    continue;
                }

                // Check privilege requirements
                if handler.requires_supervisor() && !context.is_supervisor_request {
                    log::warn!(
                        "Handler '{}' requires supervisor privilege but request is from user channel",
                        handler.name()
                    );
                    continue;
                }

                log::trace!(
                    "Dispatching request to handler '{}' (GUID: {:?})",
                    handler.name(),
                    handler.guid()
                );

                let result = handler.handle(context);

                if result != RequestResult::NotHandled {
                    return result;
                }
            }
        }

        RequestResult::NotHandled
    }

    /// Gets the number of registered handlers.
    pub fn handler_count(&self) -> usize {
        self.registered_count.load(Ordering::SeqCst)
    }

    /// Gets the maximum number of handlers.
    pub const fn max_handlers(&self) -> usize {
        MAX_HANDLERS
    }

    /// Checks if a handler for the given GUID is registered.
    pub fn has_handler(&self, guid: &efi::Guid) -> bool {
        for slot in &self.slots {
            if let Some(handler) = slot.get() {
                if &handler.guid() == guid {
                    return true;
                }
            }
        }
        false
    }

    /// Iterates over registered handlers, calling the closure for each.
    pub fn for_each_handler<F: FnMut(&dyn RequestHandler)>(&self, mut f: F) {
        for slot in &self.slots {
            if let Some(handler) = slot.get() {
                f(handler);
            }
        }
    }
}

impl<const MAX_HANDLERS: usize> Default for RequestDispatcher<MAX_HANDLERS> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestHandler {
        guid: efi::Guid,
        name: &'static str,
    }

    impl RequestHandler for TestHandler {
        fn guid(&self) -> efi::Guid {
            self.guid
        }

        fn handle(&self, _context: &mut RequestContext) -> RequestResult {
            RequestResult::Success
        }

        fn name(&self) -> &'static str {
            self.name
        }
    }

    #[test]
    fn test_request_result_conversion() {
        assert_eq!(efi::Status::SUCCESS, RequestResult::Success.into());
        assert_eq!(efi::Status::NOT_FOUND, RequestResult::NotHandled.into());
    }

    #[test]
    fn test_dispatcher_is_const() {
        // Verify we can create a static dispatcher
        static _DISPATCHER: RequestDispatcher<8> = RequestDispatcher::new();
    }

    #[test]
    fn test_dispatcher_registration() {
        static HANDLER: TestHandler = TestHandler {
            guid: efi::Guid::from_fields(
                0x12345678, 0x1234, 0x5678,
                0x12, 0x34, &[0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0]
            ),
            name: "TestHandler",
        };

        let dispatcher: RequestDispatcher<4> = RequestDispatcher::new();
        assert!(dispatcher.register(&HANDLER));
        assert_eq!(dispatcher.handler_count(), 1);
        assert!(dispatcher.has_handler(&HANDLER.guid));
    }

    #[test]
    fn test_dispatcher_unregistration() {
        static HANDLER: TestHandler = TestHandler {
            guid: efi::Guid::from_fields(
                0x12345678, 0x1234, 0x5678,
                0x12, 0x34, &[0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0]
            ),
            name: "TestHandler",
        };

        let dispatcher: RequestDispatcher<4> = RequestDispatcher::new();
        dispatcher.register(&HANDLER);
        assert!(dispatcher.unregister(&HANDLER.guid));
        assert!(!dispatcher.has_handler(&HANDLER.guid));
        assert_eq!(dispatcher.handler_count(), 0);
    }

    #[test]
    fn test_dispatcher_max_handlers() {
        static HANDLER1: TestHandler = TestHandler {
            guid: efi::Guid::from_fields(0x1, 0, 0, 0, 0, &[0; 6]),
            name: "Handler1",
        };
        static HANDLER2: TestHandler = TestHandler {
            guid: efi::Guid::from_fields(0x2, 0, 0, 0, 0, &[0; 6]),
            name: "Handler2",
        };
        static HANDLER3: TestHandler = TestHandler {
            guid: efi::Guid::from_fields(0x3, 0, 0, 0, 0, &[0; 6]),
            name: "Handler3",
        };

        let dispatcher: RequestDispatcher<2> = RequestDispatcher::new();
        assert!(dispatcher.register(&HANDLER1));
        assert!(dispatcher.register(&HANDLER2));
        assert!(!dispatcher.register(&HANDLER3)); // Should fail - full
        assert_eq!(dispatcher.handler_count(), 2);
    }

    #[test]
    fn test_request_header_validation() {
        let valid_header = MmSupervisorRequestHeader {
            signature: SIGNATURE,
            revision: REVISION,
            request: requests::VERSION_INFO,
            reserved: 0,
            result: 0,
        };
        assert!(valid_header.is_valid());

        let invalid_header = MmSupervisorRequestHeader {
            signature: 0x12345678,
            revision: REVISION,
            request: requests::VERSION_INFO,
            reserved: 0,
            result: 0,
        };
        assert!(!invalid_header.is_valid());
    }
}
