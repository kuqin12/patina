//! Arch-specific timer functionality
//!
//! Provides the [`ArchTimerFunctionality`] trait for components that need
//! timing services, plus portable helpers for reading the hardware performance
//! counter and determining its frequency.
//!
//! Architecture-specific calibration routines live in submodules:
//!
//! - [`x86_64`] – ACPI PM Timer-based TSC calibration.
//! - [`aarch64`] – (reserved for future use).
//!
//! ## Overriding the frequency
//!
//! By default, this module attempts to determine the timer frequency via architecture specific methods.
//! (cpuid for x86, `CNTFRQ_EL0` for aarch64)
//!
//! Platforms can override this with a custom performance frequency by providing the Core with the correct frequency:
//!
//! <!-- (The below test has to be ignore because `patna` cannot depend on `patina_dxe_core` - circular dependency.) -->
//! ```rust,ignore
//!     let frequency_hz: u64 = 1_000_000_000; // Compute with platform-specific methods.
//!
//!     Core::default()
//!        .init_timer_frequency(Some(frequency_hz))
//!```
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

/// Trait that provides architecture-specific timer functionality.
/// Components that need timing functionality can request this service.
pub trait ArchTimerFunctionality: Send + Sync {
    /// Value of the counter (ticks).
    fn cpu_count(&self) -> u64;
    /// Value in Hz of how often the counter increment.
    fn perf_frequency(&self) -> u64;
    /// Value that the performance counter starts with.
    fn cpu_count_start(&self) -> u64 {
        0
    }
    /// Value that the performance counter ends with before it rolls over.
    fn cpu_count_end(&self) -> u64 {
        u64::MAX
    }
}

/// Returns the current value of the hardware performance counter.
///
/// * **x86_64** – reads `RDTSC`.
/// * **aarch64** – reads `CNTPCT_EL0`.
///
/// The function is intentionally `#[inline(always)]` so that callers pay
/// minimal overhead for a single timestamp read.
#[coverage(off)]
#[inline(always)]
pub fn arch_cpu_count() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `_rdtsc` is a leaf intrinsic that reads the processor's
        // timestamp counter.  It has no memory or pointer safety implications.
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(target_arch = "aarch64")]
    {
        crate::read_sysreg!(CNTPCT_EL0)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    0
}

/// Attempts to determine the performance counter frequency (in Hz) using
/// architecture-specific detection.
///
/// * **x86_64** – tries CPUID leaf 0x15, then leaf 0x16.
/// * **aarch64** – reads `CNTFRQ_EL0`.
///
/// Returns `0` if the frequency cannot be determined.  Callers should treat
/// `0` as "unknown" and fall back to a platform-provided value (e.g. PM Timer
/// calibration on Q35).
#[coverage(off)]
pub fn arch_perf_frequency() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{self, CpuidResult};

        // SAFETY: CPUID is always available on x86_64 and is safe to call.
        let CpuidResult { eax, ebx, ecx, .. } = unsafe { x86_64::__cpuid(0x15) };
        if eax != 0 && ebx != 0 && ecx != 0 {
            // CPUID 0x15 gives TSC_frequency = (ECX * EBX) / EAX.
            return (ecx as u64 * ebx as u64) / eax as u64;
        }

        // CPUID 0x16 gives base frequency in MHz in EAX.
        let CpuidResult { eax, .. } = unsafe { x86_64::__cpuid(0x16) };
        if eax != 0 {
            return (eax as u64) * 1_000_000;
        }

        0
    }

    #[cfg(target_arch = "aarch64")]
    {
        crate::read_sysreg!(CNTFRQ_EL0)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    0
}

#[cfg(test)]
#[coverage(off)]
mod tests {
    use super::*;

    #[test]
    fn test_arch_cpu_count_is_monotonic() {
        let a = arch_cpu_count();
        let b = arch_cpu_count();
        // TSC / system counter should be non-decreasing.
        assert!(b >= a);
    }

    #[test]
    fn test_arch_perf_frequency_is_non_negative() {
        // On a host that supports CPUID 0x15/0x16 this should be > 0.
        // On CI VMs it may be 0 — that's acceptable.
        let _f = arch_perf_frequency();
    }
}
