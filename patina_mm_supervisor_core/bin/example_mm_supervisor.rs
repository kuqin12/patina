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

/// ACPI PM Timer port on QEMU Q35 (from FADT X_PM_TIMER_BLOCK).
const PM_TIMER_PORT: u16 = 0x608;

impl CpuInfo for ExamplePlatform {
    /// Override the default AP polling timeout if needed.
    fn ap_poll_timeout_us() -> u64 {
        1000 // 1ms polling interval
    }

    fn perf_timer_frequency() -> Option<u64> {
        // SAFETY: On Q35 the PM Timer is always available at PM_TIMER_PORT.
        Some(unsafe {
            patina::component::service::timer::x86_64::calibrate_tsc_from_pm_timer(PM_TIMER_PORT)
        })
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

    // Initialize the advanced logger on the first CPU to arrive (BSP)
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
