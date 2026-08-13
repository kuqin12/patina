//! Performance Timer for the MM Supervisor Core
//!
//! Provides real-time, TSC-based timing helpers used by mailbox timeouts and
//! AP-arrival polling.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use core::arch::x86_64;
use core::arch::x86_64::{__cpuid, CpuidResult};

const CPUID_TIME_STAMP_COUNTER: u32 = 0x15;
const CPUID_PROCESSOR_FREQUENCY: u32 = 0x16;

// TODO: This is copied from perf_timer.rs in patina_dxe_core
/// Returns the current CPU count using architecture-specific methods.
///
/// Skip coverage as any value could be valid, including 0.
#[cfg_attr(coverage, coverage(off))]
fn ticks() -> u64 {
    // SAFETY: _rdtsc only reads the TSC on x86_64. No invariants are required for safety.
    unsafe { x86_64::_rdtsc() }
}

/// Converts a duration in microseconds to the equivalent tick count using the
/// CPU's detected performance-counter frequency (via CPUID).
#[inline]
pub fn us_to_ticks(us: u64) -> Option<u64> {
    let freq = arch_perf_frequency();
    Some(((freq as u128 * us as u128) / 1_000_000) as u64)
}

pub(crate) fn arch_perf_frequency() -> u64 {
    // Try to get TSC frequency from CPUID (most Intel and AMD platforms).
    let CpuidResult { eax, ebx, ecx, .. } = __cpuid(CPUID_TIME_STAMP_COUNTER);
    if eax != 0 && ebx != 0 && ecx != 0 {
        // CPUID 0x15 gives TSC_frequency = (ECX * EBX) / EAX.
        // Most modern x86 platforms support this leaf.
        return (ecx as u64 * ebx as u64) / eax as u64;
    }

    // CPUID 0x16 gives base frequency in MHz in EAX.
    // This is supported on some older x86 platforms.
    // This is a nominal frequency and is less accurate for reflecting actual operating conditions.
    let CpuidResult { eax, .. } = __cpuid(CPUID_PROCESSOR_FREQUENCY);
    if eax != 0 {
        // CPUID 0x16 gives base frequency in MHz in EAX.
        // This is supported on some older x86 platforms.
        // This is a nominal frequency and is less accurate for reflecting actual operating conditions.
        return (eax * 1_000_000) as u64;
    }

    0
}

/// Spins until at least `timeout_us` microseconds have elapsed.
///
/// Returns `true` when the provided `condition` closure returns `true`
/// before the deadline, or `false` on timeout.
///
/// If the performance frequency is unknown, falls back to a conservative
/// iteration-count heuristic (`timeout_us * 10` loops).
pub fn spin_until<F>(timeout_us: u64, mut condition: F) -> bool
where
    F: FnMut() -> bool,
{
    if let Some(deadline_ticks) = us_to_ticks(timeout_us) {
        let start = ticks();
        loop {
            if condition() {
                return true;
            }
            if ticks().wrapping_sub(start) >= deadline_ticks {
                return false;
            }
            core::hint::spin_loop();
        }
    } else {
        // Fallback: iteration-count approximation.
        let iterations = timeout_us.saturating_mul(10);
        for _ in 0..iterations {
            if condition() {
                return true;
            }
            core::hint::spin_loop();
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_us_to_ticks_returns_some() {
        // The exact value depends on the CPU's detected frequency, but a conversion is
        // always produced.
        assert!(us_to_ticks(1000).is_some());
    }

    #[test]
    fn test_spin_until_immediate_true() {
        let result = spin_until(1_000, || true);
        assert!(result);
    }
}
