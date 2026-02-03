//! MM Supervisor Core
//!
//! A pure Rust implementation of the MM Supervisor Core for standalone MM mode environments.
//!
//! This crate provides the core functionality for running a supervisor in MM (Management Mode)
//! that orchestrates incoming requests on the BSP while APs wait in a holding pen.
//!
//! ## Architecture
//!
//! The entry point is executed on all cores:
//! - **BSP**: Performs one-time initialization and enters the request serving loop
//! - **APs**: Enter a holding pen and poll mailboxes for commands from BSP
//!
//! ## Memory Model
//!
//! This is a core component that manages its own memory. It does **not** use heap allocation.
//! All structures use fixed-size arrays with compile-time constants provided via const generics.
//!
//! ## Example
//!
//! ```rust,ignore
//! use patina_mm_supervisor_core::*;
//!
//! struct MyPlatform;
//!
//! impl PlatformInfo for MyPlatform {
//!     type CpuInfo = Self;
//!     const MAX_CPU_COUNT: usize = 8;
//!     const MAX_HANDLERS: usize = 32;
//! }
//!
//! impl CpuInfo for MyPlatform {
//!     fn ap_poll_timeout_us() -> u64 { 1000 }
//! }
//!
//! static SUPERVISOR: MmSupervisorCore<MyPlatform> = MmSupervisorCore::new();
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
#![feature(coverage_attribute)]
#![feature(generic_const_exprs)]

mod cpu;
mod mailbox;
mod request_handler;

pub use cpu::{ApState, CpuInfo, CpuManager};
pub use mailbox::{ApCommand, ApMailbox, ApResponse, MailboxManager};
pub use request_handler::{RequestContext, RequestHandler, RequestResult, RequestDispatcher};

use core::{
    ffi::c_void,
    num::NonZeroUsize,
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicU32, Ordering},
};

use spin::Once;

/// A trait to be implemented by the platform to provide configuration values and types to be used
/// by the MM Supervisor Core.
///
/// ## Example
///
/// ```rust,ignore
/// use patina_mm_supervisor_core::*;
///
/// struct ExamplePlatform;
///
/// impl CpuInfo for ExamplePlatform {
///     fn ap_poll_timeout_us() -> u64 { 1000 }
/// }
///
/// impl PlatformInfo for ExamplePlatform {
///     type CpuInfo = Self;
///     const MAX_CPU_COUNT: usize = 8;
///     const MAX_HANDLERS: usize = 32;
/// }
/// ```
#[cfg_attr(test, mockall::automock(type CpuInfo = MockCpuInfo;))]
pub trait PlatformInfo: 'static {
    /// The platform's CPU information and configuration.
    type CpuInfo: CpuInfo;

    /// Maximum number of CPUs supported by the platform.
    /// This is used to size the CPU manager and mailbox arrays.
    const MAX_CPU_COUNT: usize;

    /// Maximum number of request handlers that can be registered.
    const MAX_HANDLERS: usize;
}

/// Static reference to the MM Supervisor Core instance.
///
/// This is set during the `entry_point` call and provides global access to the supervisor.
static __SUPERVISOR: Once<NonZeroUsize> = Once::new();

/// Counter for tracking CPU arrivals at the entry point.
static CPU_ARRIVAL_COUNT: AtomicU32 = AtomicU32::new(0);

/// Flag indicating that initialization is complete.
static BSP_INIT_COMPLETE: AtomicBool = AtomicBool::new(false);

/// The MM Supervisor Core responsible for managing the standalone MM environment.
///
/// This struct is generic over the [`PlatformInfo`] trait, which provides platform-specific
/// configuration including compile-time constants for array sizes.
///
/// The supervisor manages:
/// - BSP initialization and request handling
/// - AP management through the holding pen and mailbox system
/// - Request dispatching and response handling
///
/// ## Memory Model
///
/// This struct does not perform heap allocation. All internal structures use fixed-size
/// arrays based on the `MAX_CPU_COUNT` and `MAX_HANDLERS` constants from [`PlatformInfo`].
///
/// ## Usage
///
/// Create a static instance of the supervisor and call `entry_point` from all cores:
///
/// ```rust,ignore
/// use patina_mm_supervisor_core::*;
///
/// static SUPERVISOR: MmSupervisorCore<MyPlatform> = MmSupervisorCore::new();
///
/// #[no_mangle]
/// pub extern "efiapi" fn mm_entry(hob_list: *const c_void) -> ! {
///     SUPERVISOR.entry_point(hob_list)
/// }
/// ```
pub struct MmSupervisorCore<P: PlatformInfo>
where
    [(); P::MAX_CPU_COUNT]:,
    [(); P::MAX_HANDLERS]:,
{
    /// Manager for CPU-related operations.
    cpu_manager: CpuManager<{ P::MAX_CPU_COUNT }>,
    /// Manager for AP mailboxes.
    mailbox_manager: MailboxManager<{ P::MAX_CPU_COUNT }>,
    /// Request dispatcher for handling incoming requests.
    request_dispatcher: RequestDispatcher<{ P::MAX_HANDLERS }>,
    /// Flag indicating if the core has been initialized.
    initialized: AtomicBool,
    /// Phantom data for the platform type.
    _phantom: core::marker::PhantomData<P>,
}

// SAFETY: The MmSupervisorCore is designed to be shared across threads with proper synchronization.
unsafe impl<P: PlatformInfo> Send for MmSupervisorCore<P>
where
    [(); P::MAX_CPU_COUNT]:,
    [(); P::MAX_HANDLERS]:,
{
}
unsafe impl<P: PlatformInfo> Sync for MmSupervisorCore<P>
where
    [(); P::MAX_CPU_COUNT]:,
    [(); P::MAX_HANDLERS]:,
{
}

#[coverage(off)]
impl<P: PlatformInfo> MmSupervisorCore<P>
where
    [(); P::MAX_CPU_COUNT]:,
    [(); P::MAX_HANDLERS]:,
{
    /// Creates a new instance of the MM Supervisor Core.
    ///
    /// This is a const fn that performs no heap allocation.
    pub const fn new() -> Self {
        Self {
            cpu_manager: CpuManager::new(),
            mailbox_manager: MailboxManager::new(),
            request_dispatcher: RequestDispatcher::new(),
            initialized: AtomicBool::new(false),
            _phantom: core::marker::PhantomData,
        }
    }

    /// Sets the static supervisor instance for global access.
    ///
    /// Returns true if the address was successfully stored, false if already set.
    #[must_use]
    fn set_instance(&'static self) -> bool {
        let physical_address = NonNull::from_ref(self).expose_provenance();
        &physical_address == __SUPERVISOR.call_once(|| physical_address)
    }

    /// Gets the static MM Supervisor Core instance for global access.
    #[allow(unused)]
    pub(crate) fn instance<'a>() -> &'a Self {
        // SAFETY: The pointer is guaranteed to be valid as set_instance ensures single initialization.
        unsafe {
            NonNull::<Self>::with_exposed_provenance(
                *__SUPERVISOR.get().expect("MM Supervisor Core is not initialized."),
            )
            .as_ref()
        }
    }

    /// The entry point for the MM Supervisor Core.
    ///
    /// This function is called on all cores (BSP and APs). The BSP performs initialization
    /// and enters the request serving loop, while APs enter the holding pen.
    ///
    /// # Arguments
    ///
    /// * `hob_list` - Pointer to the HOB (Hand-Off Block) list passed from the pre-MM phase.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The supervisor instance was already set
    /// - The HOB list pointer is null
    pub fn entry_point(&'static self, hob_list: *const c_void) -> ! {
        // Get the current CPU's APIC ID to determine if we're BSP or AP
        let cpu_id = cpu::get_current_cpu_id();

        // Track CPU arrival
        let arrival_order = CPU_ARRIVAL_COUNT.fetch_add(1, Ordering::SeqCst);

        // The first CPU to arrive is considered the BSP
        let is_bsp = arrival_order == 0;

        if is_bsp {
            // BSP path: Initialize the supervisor
            assert!(self.set_instance(), "MM Supervisor Core instance was already set!");
            assert!(!hob_list.is_null(), "MM Supervisor Core requires a non-null HOB list pointer.");

            log::info!("MM Supervisor Core v{}", env!("CARGO_PKG_VERSION"));
            log::info!("BSP (CPU {}) starting initialization...", cpu_id);

            // Register BSP with CPU manager
            self.cpu_manager.register_cpu(cpu_id, true);

            // Perform platform-specific initialization
            self.bsp_init(hob_list);

            // Mark as initialized
            self.initialized.store(true, Ordering::Release);

            // Signal that initialization is complete
            BSP_INIT_COMPLETE.store(true, Ordering::Release);

            log::info!("BSP initialization complete, entering request serving loop...");

            // Enter the main request serving loop
            self.bsp_request_loop()
        } else {
            // AP path: Wait for BSP to complete initialization, then enter holding pen
            log::trace!("AP (CPU {}) waiting for BSP initialization...", cpu_id);

            // Spin until BSP completes initialization
            while !BSP_INIT_COMPLETE.load(Ordering::Acquire) {
                core::hint::spin_loop();
            }

            // Register this AP with the CPU manager
            self.cpu_manager.register_cpu(cpu_id, false);

            log::trace!("AP (CPU {}) entering holding pen...", cpu_id);

            // Enter the holding pen
            self.ap_holding_pen(cpu_id)
        }
    }

    /// BSP-specific initialization.
    ///
    /// This is called only on the BSP after basic setup is complete.
    fn bsp_init(&'static self, _hob_list: *const c_void) {
        log::trace!("BSP performing one-time initialization...");

        // TODO: Process HOB list for MM-specific configuration
        // TODO: Initialize memory services
        // TODO: Set up protocol database
        // TODO: Initialize request handler infrastructure

        log::trace!("BSP one-time initialization complete.");
    }

    /// The main request serving loop for the BSP.
    ///
    /// The BSP sits in this loop, processing incoming requests and dispatching
    /// work to APs as needed.
    fn bsp_request_loop(&'static self) -> ! {
        log::trace!("BSP entering request serving loop...");

        loop {
            // Check for incoming requests
            // In a real implementation, this would check the communication buffer
            // and dispatch handlers for incoming MM requests.

            // Process any pending work
            self.process_pending_requests();

            // Brief pause to avoid spinning too aggressively
            core::hint::spin_loop();
        }
    }

    /// Process pending requests from the communication buffer.
    fn process_pending_requests(&self) {
        // TODO: Check communication buffer for incoming requests
        // TODO: Dispatch to registered handlers via self.request_dispatcher
        // TODO: Optionally distribute work to APs via mailbox
    }

    /// The holding pen for APs.
    ///
    /// APs wait here, polling their mailbox for commands from the BSP.
    fn ap_holding_pen(&'static self, cpu_id: u32) -> ! {
        log::trace!("AP (CPU {}) in holding pen, polling mailbox...", cpu_id);

        loop {
            // Check mailbox for commands
            if let Some(command) = self.mailbox_manager.check_mailbox(cpu_id) {
                log::trace!("AP (CPU {}) received command: {:?}", cpu_id, command);

                // Execute the command
                let response = self.execute_ap_command(cpu_id, command);

                // Post the response
                self.mailbox_manager.post_response(cpu_id, response);
            }

            // Pause to avoid spinning too aggressively
            // In a production system, this might use MWAIT or HLT
            for _ in 0..1000 {
                core::hint::spin_loop();
            }
        }
    }

    /// Execute a command received by an AP.
    fn execute_ap_command(&self, cpu_id: u32, command: ApCommand) -> ApResponse {
        match command {
            ApCommand::Nop => {
                log::trace!("AP (CPU {}) executing NOP", cpu_id);
                ApResponse::Success
            }
            ApCommand::Execute { handler_id, context } => {
                log::trace!("AP (CPU {}) executing handler {}", cpu_id, handler_id);
                // TODO: Look up and execute the handler
                let _ = context;
                ApResponse::Success
            }
            ApCommand::Shutdown => {
                log::trace!("AP (CPU {}) received shutdown command", cpu_id);
                ApResponse::Success
            }
        }
    }

    /// Register a request handler with the supervisor.
    ///
    /// Handlers are invoked when matching requests are received.
    ///
    /// Returns `true` if the handler was registered, `false` if the handler table is full.
    pub fn register_handler(&self, handler: &'static dyn RequestHandler) -> bool {
        self.request_dispatcher.register(handler)
    }

    /// Get the CPU manager.
    pub fn cpu_manager(&self) -> &CpuManager<{ P::MAX_CPU_COUNT }> {
        &self.cpu_manager
    }

    /// Get the mailbox manager.
    pub fn mailbox_manager(&self) -> &MailboxManager<{ P::MAX_CPU_COUNT }> {
        &self.mailbox_manager
    }

    /// Get the request dispatcher.
    pub fn request_dispatcher(&self) -> &RequestDispatcher<{ P::MAX_HANDLERS }> {
        &self.request_dispatcher
    }

    /// Send a command to a specific AP.
    ///
    /// Returns `Ok(())` if the command was successfully posted to the mailbox,
    /// or `Err(())` if the AP is not available or the mailbox is full.
    pub fn send_ap_command(&self, cpu_id: u32, command: ApCommand) -> Result<(), ()> {
        self.mailbox_manager.send_command(cpu_id, command)
    }

    /// Wait for a response from a specific AP.
    ///
    /// Returns the response from the AP, or `None` if timeout or error.
    pub fn wait_ap_response(&self, cpu_id: u32, timeout_us: u64) -> Option<ApResponse> {
        self.mailbox_manager.wait_response(cpu_id, timeout_us)
    }

    /// Check if the supervisor has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }
}

impl<P: PlatformInfo> Default for MmSupervisorCore<P>
where
    [(); P::MAX_CPU_COUNT]:,
    [(); P::MAX_HANDLERS]:,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlatform;

    impl CpuInfo for TestPlatform {}

    impl PlatformInfo for TestPlatform {
        type CpuInfo = Self;
        const MAX_CPU_COUNT: usize = 4;
        const MAX_HANDLERS: usize = 8;
    }

    #[test]
    fn test_cpu_info_defaults() {
        assert_eq!(<TestPlatform as CpuInfo>::ap_poll_timeout_us(), 1000);
    }

    #[test]
    fn test_supervisor_creation() {
        let _supervisor: MmSupervisorCore<TestPlatform> = MmSupervisorCore::new();
        // Just verify it compiles and creates without panic
    }

    #[test]
    fn test_supervisor_is_const() {
        // Verify we can create a static instance (no heap allocation)
        static _SUPERVISOR: MmSupervisorCore<TestPlatform> = MmSupervisorCore::new();
    }
}
