//! Example MM Supervisor Binary for QEMU Q35
//!
//! This is an example platform binary that demonstrates how to build a PE/COFF
//! MM Supervisor using the `patina_mm_supervisor_core` crate.
//!
//! ## Building
//!
//! Build with cargo for the UEFI target:
//! ```bash
//! cargo build --release --target x86_64-unknown-uefi --bin example_mm_supervisor
//! ```
//!
//! ## Entry Point
//!
//! The MM Supervisor is handed off by the MM IPL (Initial Program Loader) after:
//! - Page tables are set up
//! - The supervisor image is loaded into MMRAM
//! - A HOB list is constructed with MMRAM ranges and other configuration
//!
//! The entry point `MmSupervisorMain` is called on ALL processors simultaneously.
//! The first processor to arrive becomes the BSP, others become APs.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!
#![cfg(all(target_os = "uefi", target_arch = "x86_64"))]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![no_std]
#![no_main]

use core::{ffi::c_void, panic::PanicInfo};
use core::sync::atomic::AtomicBool;
use patina_mm_supervisor_core::*;
// use the the uart from patina
use patina::{log::Format, serial::uart::Uart16550};
use patina_adv_logger::logger::AdvancedLogger;
use patina_stacktrace::StackTrace;

// =============================================================================
// Platform Configuration
// =============================================================================

/// Platform configuration for the example MM Supervisor.
struct ExamplePlatform;

impl CpuInfo for ExamplePlatform {
    /// Override the default AP polling timeout if needed.
    fn ap_poll_timeout_us() -> u64 {
        1000 // 1ms polling interval
    }
}

impl PlatformInfo for ExamplePlatform {
    type CpuInfo = Self;

    /// Maximum number of CPUs this platform supports.
    /// This should match your hardware/VM configuration.
    const MAX_CPU_COUNT: usize = 8;

    /// Maximum number of request handlers that can be registered.
    const MAX_HANDLERS: usize = 32;
}

/// Flag indicating that advanced logger initialization is complete.
static ADV_LOGGER_INIT_COMPLETE: AtomicBool = AtomicBool::new(false);

// =============================================================================
// Static Supervisor Instance
// =============================================================================

/// The static MM Supervisor Core instance.
///
/// This is instantiated at compile time with no heap allocation.
static SUPERVISOR: MmSupervisorCore<ExamplePlatform> = MmSupervisorCore::new();

static LOGGER: AdvancedLogger<Uart16550> = AdvancedLogger::new(
    Format::Standard,
    &[
        ("goblin", log::LevelFilter::Off),
        ("gcd_measure", log::LevelFilter::Off),
        ("allocations", log::LevelFilter::Off),
        ("efi_memory_map", log::LevelFilter::Off),
        ("mm_comm", log::LevelFilter::Off),
        ("sw_mmi", log::LevelFilter::Off),
        ("patina_performance", log::LevelFilter::Off),
    ],
    log::LevelFilter::Info,
    Uart16550::Io { base: 0x402 },
);

// =============================================================================
// Request Handlers (Examples)
// =============================================================================

/// Example: Version info request handler.
struct VersionInfoHandler;

impl RequestHandler for VersionInfoHandler {
    fn guid(&self) -> r_efi::efi::Guid {
        // MM Supervisor Request Handler GUID
        // This should match gMmSupervisorRequestHandlerGuid from the EDK2 headers
        r_efi::efi::Guid::from_fields(
            0x2e6b1cb5,
            0x7d56,
            0x40e9,
            0xa3,
            0x16,
            &[0x4a, 0x0e, 0xf4, 0xef, 0x85, 0x53],
        )
    }

    fn handle(&self, context: &mut RequestContext) -> RequestResult {
        // TODO: Parse the request header and return version info
        // This is where you'd implement the VERSION_INFO request handling
        let _ = context;
        RequestResult::Success
    }

    fn name(&self) -> &'static str {
        "VersionInfoHandler"
    }

    fn requires_supervisor(&self) -> bool {
        false // Can be called from user channel
    }
}

/// Static instance of the version info handler.
static VERSION_INFO_HANDLER: VersionInfoHandler = VersionInfoHandler;

// =============================================================================
// Panic Handler
// =============================================================================

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    log::error!("{}", info);

    if let Err(err) = unsafe { StackTrace::dump() } {
        log::error!("StackTrace: {}", err);
    }

    loop {}
}

// =============================================================================
// Entry Point
// =============================================================================

/// The MM Supervisor entry point.
///
/// This function is called by the MM IPL on ALL processors after the supervisor
/// image has been loaded into MMRAM and page tables have been configured.
///
/// # Arguments
///
/// * `hob_list` - Pointer to the HOB (Hand-Off Block) list containing:
///   - MMRAM ranges
///   - Memory allocation information
///   - Platform configuration
///   - FV (Firmware Volume) locations for MM drivers
///
/// # Entry Convention
///
/// - All processors enter this function simultaneously
/// - The first processor to arrive becomes the BSP
/// - Other processors become APs and enter the holding pen
/// - The function never returns (diverging `-> !`)
///
/// # Export Name
///
/// The export name `MmSupervisorMain` matches the EDK2 convention for
/// standalone MM supervisor entry points. The MM IPL looks for this symbol
/// when loading the supervisor.
#[unsafe(export_name = "rust_main")]
pub extern "efiapi" fn mm_supervisor_main(cpu_index: usize, hob_list: *const c_void) {
    // TODO: should not have it here because we will get back here everytime an MM call is made

    // Register platform-specific handlers before entering the main loop
    // Note: Only BSP will actually process these, but it's safe for APs to
    // call register_handler as well (they'll just fail to register duplicates)
    if !ADV_LOGGER_INIT_COMPLETE.swap(true, core::sync::atomic::Ordering::SeqCst) {
        log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Trace)).unwrap();
        // SAFETY: The physical_hob_list pointer is considered valid at this point as it's provided by the core
        // to the entry point.
        unsafe {
            LOGGER.init(hob_list).unwrap();
        }
    }

    // The entry_point handles BSP vs AP routing internally
    SUPERVISOR.entry_point(cpu_index, hob_list)
}

// =============================================================================
// Optional: Pre-BSP Initialization Hook
// =============================================================================

/// Optional early initialization that runs before the supervisor core starts.
///
/// This can be used to set up logging, debugging, or other early infrastructure.
/// Called by the entry point before `SUPERVISOR.entry_point()`.
#[allow(dead_code)]
fn early_init() {
    // Example: Initialize a serial logger
    // log::set_logger(&MY_LOGGER).ok();

    // Example: Register handlers
    // Note: Must be done before entry_point() or by the BSP during bsp_init()
    let _ = SUPERVISOR.register_handler(&VERSION_INFO_HANDLER);
}
