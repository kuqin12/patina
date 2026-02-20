//! x86_64-specific timer calibration routines.
//!
//! Provides [`calibrate_tsc_from_pm_timer`] for platforms (e.g. QEMU Q35)
//! where CPUID leaves 0x15/0x16 do not report the TSC frequency.
//!
//! ## References
//!
//! - [ACPI PM Timer](https://uefi.org/specs/ACPI/6.5/04_ACPI_Hardware_Specification.html)
//! - [FADT Table Definition](https://uefi.org/htmlspecs/ACPI_Spec_6_4_html/05_ACPI_Software_Programming_Model/ACPI_Software_Programming_Model.html#fixed-acpi-description-table-fadt)
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

/// Standard ACPI PM Timer frequency: 3.579545 MHz.
const ACPI_PM_TIMER_FREQUENCY: u64 = 3_579_545;

/// Calibrates the TSC frequency by measuring TSC ticks over a known interval
/// of the ACPI Power Management Timer.
///
/// This is the standard approach for platforms (e.g. QEMU Q35) where CPUID
/// leaves 0x15/0x16 do not report the TSC frequency.
///
/// The function measures approximately 50 ms of PM Timer ticks and derives
/// the TSC frequency from the ratio of TSC delta to elapsed wall-clock time.
///
/// # Safety
///
/// The caller must ensure that `pm_timer_port` is a valid I/O port for the
/// ACPI PM Timer (e.g. `0x608` on Q35).  Reading from an invalid port is
/// undefined behavior.
pub unsafe fn calibrate_tsc_from_pm_timer(pm_timer_port: u16) -> u64 {
    use core::arch::x86_64 as arch;

    const MAX_WAIT_CYCLES: usize = 1_000_000;

    // Wait for a PM timer edge to avoid partial intervals.
    let mut start_pm = unsafe { read_pm_timer(pm_timer_port) };
    let mut cycles_left = MAX_WAIT_CYCLES;
    loop {
        let next = unsafe { read_pm_timer(pm_timer_port) };
        if next != start_pm {
            start_pm = next;
            break;
        }
        cycles_left -= 1;
        if cycles_left == 0 {
            log::warn!("PM timer calibration: timeout waiting for edge");
            break;
        }
    }

    // Record starting TSC.
    // SAFETY: `_rdtsc` reads the timestamp counter — no memory safety implications.
    let start_tsc = unsafe { arch::_rdtsc() };

    // Hz / 20 ≈ 50 ms worth of PM timer ticks.
    const TARGET_INTERVAL_DIVISOR: u64 = 20;
    let target_ticks = (ACPI_PM_TIMER_FREQUENCY / TARGET_INTERVAL_DIVISOR) as u32;

    let mut end_pm;
    cycles_left = MAX_WAIT_CYCLES;
    loop {
        end_pm = unsafe { read_pm_timer(pm_timer_port) };
        if end_pm.wrapping_sub(start_pm) >= target_ticks {
            break;
        }
        cycles_left -= 1;
        if cycles_left == 0 {
            log::warn!("PM timer calibration: timeout waiting for target ticks");
            return ACPI_PM_TIMER_FREQUENCY; // best-effort fallback
        }
    }

    // Record ending TSC.
    let end_tsc = unsafe { arch::_rdtsc() };

    // Compute: frequency = delta_tsc / delta_time
    // where delta_time = delta_pm / ACPI_PM_TIMER_FREQUENCY (in seconds).
    let delta_pm = end_pm.wrapping_sub(start_pm) as u64;
    let delta_time_ns = (delta_pm * 1_000_000_000) / ACPI_PM_TIMER_FREQUENCY;
    let delta_tsc = end_tsc - start_tsc;

    (delta_tsc * 1_000_000_000) / delta_time_ns
}

/// Reads the 32-bit ACPI PM Timer value from `port`.
///
/// # Safety
///
/// `port` must be a valid I/O port address for the PM Timer.
unsafe fn read_pm_timer(port: u16) -> u32 {
    let value: u32;
    unsafe {
        core::arch::asm!(
            "in eax, dx",
            in("dx") port,
            out("eax") value,
            options(nomem, nostack, preserves_flags),
        );
    }
    value
}
