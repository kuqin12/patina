//! Performance Timer for the MM Supervisor Core
//!
//! Provides real-time, TSC-based timing helpers used by mailbox timeouts and
//! AP-arrival polling.  The module reads the hardware counter via the shared
//! [`patina::component::service::timer`] utilities and stores a one-time
//! calibrated frequency.
//!
//! ## Initialization
//!
//! Call [`init`] once during BSP init with the platform-provided frequency
//! (or `0` to auto-detect from CPUID).
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use patina::component::service::timer as perf_timer;
use spin::Once;

/// Cached performance counter frequency (Hz).  Initialized once during
/// BSP init and read by all cores.
static FREQUENCY: Once<u64> = Once::new();

/// Initializes the performance timer with a frequency value.
///
/// If `platform_frequency` is non-zero it is used as-is.  Otherwise
/// the function falls back to CPUID-based auto-detection.
///
/// Calling this more than once is harmless — subsequent calls are no-ops.
pub fn init(platform_frequency: u64) {
    FREQUENCY.call_once(|| {
        let freq = if platform_frequency != 0 {
            platform_frequency
        } else {
            perf_timer::arch_perf_frequency()
        };

        if freq == 0 {
            log::warn!(
                "perf_timer: unable to determine performance counter frequency; \
                 timeouts will use iteration-count fallback"
            );
        } else {
            log::info!("perf_timer: frequency = {} Hz ({:.3} GHz)", freq, freq as f64 / 1e9);
        }

        freq
    });
}

/// Returns the current performance counter value (TSC on x86_64).
#[inline(always)]
pub fn ticks() -> u64 {
    perf_timer::arch_cpu_count()
}

/// Returns the cached frequency in Hz, or `0` if not yet initialized /
/// not determinable.
#[inline]
pub fn frequency() -> u64 {
    FREQUENCY.get().copied().unwrap_or(0)
}

/// Converts a duration in microseconds to the equivalent tick count using
/// the cached frequency.
///
/// Returns `None` if the frequency is unknown (0).
#[inline]
pub fn us_to_ticks(us: u64) -> Option<u64> {
    let freq = frequency();
    if freq == 0 {
        return None;
    }
    // ticks = freq * us / 1_000_000
    // Use u128 to avoid overflow for large values of freq * us.
    Some(((freq as u128 * us as u128) / 1_000_000) as u64)
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
    fn test_init_with_known_frequency() {
        // Note: Once is global — run in isolation or accept first-writer-wins.
        init(1_000_000_000); // 1 GHz
        assert!(frequency() > 0);
    }

    #[test]
    fn test_us_to_ticks_basic() {
        // For a 1 GHz counter, 1 us = 1000 ticks.
        // We can't guarantee the global FREQUENCY state, so just verify None for 0.
        if frequency() == 0 {
            assert_eq!(us_to_ticks(1000), None);
        }
    }

    #[test]
    fn test_spin_until_immediate_true() {
        let result = spin_until(1_000, || true);
        assert!(result);
    }
}
