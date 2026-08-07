//! Counting Rendezvous Semaphores for the MM Supervisor Core
//!
//! Provides lightweight, allocation-free counting semaphore primitives built on
//! a single [`AtomicU32`]. These are used to coordinate the BSP/AP SMI exit
//! barrier: the BSP signals a release count and APs consume it, and vice versa
//! for exit acknowledgements.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use core::sync::atomic::{AtomicU32, Ordering};

/// Signals a counting rendezvous semaphore (atomic increment).
pub(crate) fn sem_signal(sem: &AtomicU32) {
    sem.fetch_add(1, Ordering::AcqRel);
}

/// Blocks (spins, no timer) until the semaphore is positive, then consumes one count.
pub(crate) fn sem_wait(sem: &AtomicU32) {
    loop {
        let value = sem.load(Ordering::Acquire);
        if value != 0 && sem.compare_exchange_weak(value, value - 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            return;
        }
        core::hint::spin_loop();
    }
}

/// Consumes one count if the semaphore is positive. Non-blocking; returns whether a
/// count was taken.
pub(crate) fn sem_try_take(sem: &AtomicU32) -> bool {
    let mut value = sem.load(Ordering::Acquire);
    while value != 0 {
        match sem.compare_exchange_weak(value, value - 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(current) => value = current,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sem_signal_increments_count() {
        let sem = AtomicU32::new(0);
        sem_signal(&sem);
        assert_eq!(sem.load(Ordering::Acquire), 1);
        sem_signal(&sem);
        assert_eq!(sem.load(Ordering::Acquire), 2);
    }

    #[test]
    fn test_sem_try_take_returns_false_when_zero() {
        let sem = AtomicU32::new(0);
        assert!(!sem_try_take(&sem));
        assert_eq!(sem.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_sem_try_take_consumes_one_count() {
        let sem = AtomicU32::new(2);
        assert!(sem_try_take(&sem));
        assert_eq!(sem.load(Ordering::Acquire), 1);
        assert!(sem_try_take(&sem));
        assert_eq!(sem.load(Ordering::Acquire), 0);
        // Exhausted: no more counts to take.
        assert!(!sem_try_take(&sem));
    }

    #[test]
    fn test_sem_wait_consumes_available_count() {
        // With a positive count already available, `sem_wait` returns immediately and
        // consumes exactly one count without blocking.
        let sem = AtomicU32::new(0);
        sem_signal(&sem);
        sem_signal(&sem);
        sem_wait(&sem);
        assert_eq!(sem.load(Ordering::Acquire), 1);
        sem_wait(&sem);
        assert_eq!(sem.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_signal_wait_round_trip_is_balanced() {
        // Signaling N times and waiting N times leaves the semaphore balanced at zero.
        let sem = AtomicU32::new(0);
        for _ in 0..5 {
            sem_signal(&sem);
        }
        for _ in 0..5 {
            sem_wait(&sem);
        }
        assert_eq!(sem.load(Ordering::Acquire), 0);
        assert!(!sem_try_take(&sem));
    }
}
