//! CPU Management Module
//!
//! This module provides CPU identification and management for the MM Supervisor Core.
//! It handles BSP/AP detection, CPU registration, and state tracking.
//!
//! ## Memory Model
//!
//! This module does not perform heap allocation. All structures use fixed-size arrays
//! with compile-time constants provided via const generics.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use core::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use core::arch::{x86_64, x86_64::CpuidResult};

/// A trait to be implemented by the platform to provide CPU-related configuration.
///
/// ## Example
///
/// ```rust,ignore
/// use patina_mm_supervisor_core::CpuInfo;
///
/// struct ExamplePlatform;
///
/// impl CpuInfo for ExamplePlatform {
///     fn ap_poll_timeout_us() -> u64 { 500 }
/// }
/// ```
#[cfg_attr(test, mockall::automock)]
pub trait CpuInfo {
    /// Returns the timeout in microseconds for AP mailbox polling.
    ///
    /// By default, this returns 1000 (1ms) which is a reasonable polling interval.
    #[inline(always)]
    fn ap_poll_timeout_us() -> u64 {
        1000
    }

    /// Returns the performance counter frequency in Hz, if known by the platform.
    ///
    /// For example, on QEMU Q35 the platform can calibrate the TSC frequency
    /// from the ACPI PM Timer and return it here.
    ///
    /// If `None` is returned (the default), the supervisor will attempt
    /// auto-detection via CPUID.
    fn perf_timer_frequency() -> Option<u64> {
        None
    }
}

/// The state of an Application Processor (AP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ApState {
    /// The AP has not been registered yet.
    NotPresent = 0,
    /// The AP is in the holding pen, waiting for work.
    InHoldingPen = 1,
    /// The AP is currently executing a task.
    Busy = 2,
    /// The AP has been halted.
    Halted = 3,
}

impl From<u8> for ApState {
    fn from(value: u8) -> Self {
        match value {
            0 => ApState::NotPresent,
            1 => ApState::InHoldingPen,
            2 => ApState::Busy,
            3 => ApState::Halted,
            _ => ApState::NotPresent,
        }
    }
}

/// Information about a registered CPU stored in a fixed-size slot.
#[repr(C)]
struct CpuSlot {
    /// The CPU's APIC ID. u32::MAX means slot is unused.
    cpu_id: AtomicU32,
    /// Whether this CPU is the BSP (0 = AP, 1 = BSP).
    is_bsp: AtomicU8,
    /// Current state (for APs only).
    state: AtomicU8,
    /// Padding for alignment.
    _padding: [u8; 2],
}

impl CpuSlot {
    /// Creates a new empty CPU slot.
    const fn new() -> Self {
        Self {
            cpu_id: AtomicU32::new(u32::MAX),
            is_bsp: AtomicU8::new(0),
            state: AtomicU8::new(ApState::NotPresent as u8),
            _padding: [0; 2],
        }
    }

    /// Checks if this slot is in use.
    fn is_used(&self) -> bool {
        self.cpu_id.load(Ordering::Acquire) != u32::MAX
    }

    /// Gets the CPU ID if the slot is used.
    fn get_cpu_id(&self) -> Option<u32> {
        let id = self.cpu_id.load(Ordering::Acquire);
        if id == u32::MAX {
            None
        } else {
            Some(id)
        }
    }
}

/// Manager for CPU-related operations.
///
/// Tracks registered CPUs and their states using fixed-size arrays.
///
/// ## Const Generic Parameters
///
/// * `MAX_CPUS` - The maximum number of CPUs that can be registered.
pub struct CpuManager<const MAX_CPUS: usize> {
    /// CPU slots - fixed size array.
    slots: [CpuSlot; MAX_CPUS],
    /// Number of CPUs currently registered.
    registered_count: AtomicU32,
    /// The APIC ID of the BSP.
    bsp_id: AtomicU32,
}

impl<const MAX_CPUS: usize> CpuManager<MAX_CPUS> {
    /// Creates a new CPU manager.
    ///
    /// This is a const fn and performs no heap allocation.
    pub const fn new() -> Self {
        Self {
            slots: [const { CpuSlot::new() }; MAX_CPUS],
            registered_count: AtomicU32::new(0),
            bsp_id: AtomicU32::new(u32::MAX),
        }
    }

    /// Registers a CPU with the manager.
    ///
    /// # Arguments
    ///
    /// * `cpu_id` - The CPU's APIC ID.
    /// * `is_bsp` - Whether this CPU is the BSP.
    ///
    /// # Returns
    ///
    /// The index of the registered CPU, or `None` if max CPUs reached or already registered.
    pub fn register_cpu(&self, cpu_id: u32, is_bsp: bool) -> Option<usize> {
        // Check if already registered
        for slot in &self.slots {
            if slot.get_cpu_id() == Some(cpu_id) {
                log::warn!("CPU {} already registered", cpu_id);
                return None;
            }
        }

        // Find an empty slot
        for (index, slot) in self.slots.iter().enumerate() {
            // Try to claim this slot using compare-exchange
            let result = slot.cpu_id.compare_exchange(
                u32::MAX,
                cpu_id,
                Ordering::AcqRel,
                Ordering::Acquire,
            );

            if result.is_ok() {
                // Successfully claimed the slot
                slot.is_bsp.store(if is_bsp { 1 } else { 0 }, Ordering::Release);
                slot.state.store(
                    if is_bsp { ApState::Busy as u8 } else { ApState::InHoldingPen as u8 },
                    Ordering::Release,
                );

                self.registered_count.fetch_add(1, Ordering::SeqCst);

                if is_bsp {
                    self.bsp_id.store(cpu_id, Ordering::SeqCst);
                    log::info!("Registered BSP with APIC ID {}", cpu_id);
                } else {
                    log::trace!("Registered AP with APIC ID {} at index {}", cpu_id, index);
                }

                return Some(index);
            }
        }

        log::warn!("Maximum CPU count ({}) reached, cannot register CPU {}", MAX_CPUS, cpu_id);
        None
    }

    /// Gets the number of registered CPUs.
    pub fn registered_count(&self) -> usize {
        self.registered_count.load(Ordering::SeqCst) as usize
    }

    /// Gets the maximum number of CPUs supported.
    pub const fn max_cpus(&self) -> usize {
        MAX_CPUS
    }

    /// Gets the APIC ID of the BSP.
    pub fn bsp_id(&self) -> Option<u32> {
        let id = self.bsp_id.load(Ordering::SeqCst);
        if id == u32::MAX {
            None
        } else {
            Some(id)
        }
    }

    /// Checks if the given CPU ID is the BSP.
    pub fn is_bsp(&self, cpu_id: u32) -> bool {
        self.bsp_id() == Some(cpu_id)
    }

    /// Finds the slot index for a given CPU ID (APIC ID).
    fn find_slot(&self, cpu_id: u32) -> Option<usize> {
        for (index, slot) in self.slots.iter().enumerate() {
            if slot.get_cpu_id() == Some(cpu_id) {
                return Some(index);
            }
        }
        None
    }

    /// Finds the slot index for a given CPU ID (public wrapper).
    pub fn find_cpu_index(&self, cpu_id: u32) -> Option<usize> {
        self.find_slot(cpu_id)
    }

    /// Gets the APIC ID of the CPU at the given slot index.
    ///
    /// Returns `None` if the index is out of range or the slot is unused.
    pub fn get_cpu_id_by_index(&self, index: usize) -> Option<u32> {
        if index >= MAX_CPUS {
            return None;
        }
        self.slots[index].get_cpu_id()
    }

    /// Gets the AP state by slot index.
    ///
    /// Returns `None` if the index is out of range or the slot is unused.
    pub fn get_ap_state_by_index(&self, index: usize) -> Option<ApState> {
        if index >= MAX_CPUS {
            return None;
        }
        let slot = &self.slots[index];
        if slot.is_used() {
            Some(ApState::from(slot.state.load(Ordering::Acquire)))
        } else {
            None
        }
    }

    /// Gets the state of an AP.
    pub fn get_ap_state(&self, cpu_id: u32) -> Option<ApState> {
        let index = self.find_slot(cpu_id)?;
        Some(ApState::from(self.slots[index].state.load(Ordering::Acquire)))
    }

    /// Sets the state of an AP.
    pub fn set_ap_state(&self, cpu_id: u32, state: ApState) -> bool {
        let index = match self.find_slot(cpu_id) {
            Some(idx) => idx,
            None => return false,
        };

        let slot = &self.slots[index];

        // Don't allow changing BSP state
        if slot.is_bsp.load(Ordering::Acquire) != 0 {
            log::warn!("Attempted to change BSP state, ignoring");
            return false;
        }

        slot.state.store(state as u8, Ordering::Release);
        true
    }

    /// Iterates over all registered AP IDs.
    ///
    /// Calls the provided closure for each registered AP.
    pub fn for_each_ap<F: FnMut(u32)>(&self, mut f: F) {
        for slot in &self.slots {
            if let Some(cpu_id) = slot.get_cpu_id() {
                if slot.is_bsp.load(Ordering::Acquire) == 0 {
                    f(cpu_id);
                }
            }
        }
    }

    /// Counts APs in a specific state.
    pub fn count_aps_in_state(&self, state: ApState) -> usize {
        let mut count = 0;
        for slot in &self.slots {
            if slot.is_used()
                && slot.is_bsp.load(Ordering::Acquire) == 0
                && slot.state.load(Ordering::Acquire) == state as u8
            {
                count += 1;
            }
        }
        count
    }
}

impl<const MAX_CPUS: usize> Default for CpuManager<MAX_CPUS> {
    fn default() -> Self {
        Self::new()
    }
}

/// Gets the current CPU's APIC ID.
///
/// On x86_64, this reads the APIC ID from the Local APIC or CPUID.
#[cfg(target_arch = "x86_64")]
pub fn get_current_cpu_id() -> u32 {
    // Use CPUID to get the initial APIC ID
    // CPUID function 0x01, EBX[31:24] contains the initial APIC ID

    // SAFETY: CPUID is always available on x86_64 and reading it is safe.
    let CpuidResult { ebx, .. } = unsafe { x86_64::__cpuid(0x01) };
    let cpuid_result = (ebx >> 24) & 0xff;
    cpuid_result
}

/// Gets the current CPU's APIC ID (stub for non-x86_64).
#[cfg(not(target_arch = "x86_64"))]
pub fn get_current_cpu_id() -> u32 {
    0
}

// ============================================================================
// BSP Detection via IA32_APIC_BASE MSR
// ============================================================================

/// MSR index for IA32_APIC_BASE.
const IA32_APIC_BASE_MSR_INDEX: u32 = 0x1B;

/// BSP flag bit in IA32_APIC_BASE MSR (bit 8).
const IA32_APIC_BSP: u64 = 1 << 8;

/// MSR index for SMM Base Address (SMBASE).
pub const IA32_MSR_SMBASE: u32 = 0x9E;

/// Reads a Model-Specific Register (MSR) by index.
///
/// # Safety
///
/// The caller must ensure the MSR index is valid and readable on the current platform.
#[cfg(target_arch = "x86_64")]
pub unsafe fn read_msr(msr: u32) -> Result<u64, &'static str> {
    let lo: u32;
    let hi: u32;
    unsafe {
        core::arch::asm!(
            "rdmsr",
            in("ecx") msr,
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack),
        );
    }
    Ok(((hi as u64) << 32) | (lo as u64))
}

/// Reads a Model-Specific Register (stub for non-x86_64).
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn read_msr(_msr: u32) -> Result<u64, &'static str> {
    Err("rdmsr not supported on this architecture")
}

/// Writes a 64-bit value to a Model-Specific Register (MSR).
///
/// # Safety
///
/// The caller must ensure the MSR index is valid and writable on the current platform.
#[cfg(target_arch = "x86_64")]
pub unsafe fn write_msr(msr: u32, value: u64) -> Result<(), &'static str> {
    let lo = value as u32;
    let hi = (value >> 32) as u32;
    unsafe {
        core::arch::asm!(
            "wrmsr",
            in("ecx") msr,
            in("eax") lo,
            in("edx") hi,
            options(nomem, nostack),
        );
    }
    Ok(())
}

/// Writes a Model-Specific Register (stub for non-x86_64).
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn write_msr(_msr: u32, _value: u64) -> Result<(), &'static str> {
    Err("wrmsr not supported on this architecture")
}

/// Checks if the current processor is the Bootstrap Processor (BSP).
///
/// This reads the IA32_APIC_BASE MSR and checks the BSP flag (bit 8).
/// The BSP flag is set by hardware during reset and indicates which
/// processor is the bootstrap processor.
///
/// # Returns
///
/// `true` if this is the BSP, `false` if this is an AP.
#[cfg(target_arch = "x86_64")]
pub fn is_bsp() -> bool {
    // SAFETY: Reading the IA32_APIC_BASE MSR is safe on x86_64.
    let apic_base_lo: u32;
    let apic_base_hi: u32;
    unsafe {
        core::arch::asm!(
            "rdmsr",
            in("ecx") IA32_APIC_BASE_MSR_INDEX,
            out("eax") apic_base_lo,
            out("edx") apic_base_hi,
        );
    }

    let apic_base = ((apic_base_hi as u64) << 32) | (apic_base_lo as u64);
    (apic_base & IA32_APIC_BSP) != 0
}

/// Checks if the current processor is the BSP (stub for non-x86_64).
#[cfg(not(target_arch = "x86_64"))]
pub fn is_bsp() -> bool {
    true // Assume BSP on non-x86_64 platforms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_manager_creation() {
        let manager: CpuManager<4> = CpuManager::new();
        assert_eq!(manager.registered_count(), 0);
        assert!(manager.bsp_id().is_none());
        assert_eq!(manager.max_cpus(), 4);
    }

    #[test]
    fn test_cpu_manager_is_const() {
        // Verify we can create a static instance
        static _MANAGER: CpuManager<8> = CpuManager::new();
    }

    #[test]
    fn test_cpu_registration() {
        let manager: CpuManager<4> = CpuManager::new();

        // Register BSP
        let bsp_idx = manager.register_cpu(0, true);
        assert_eq!(bsp_idx, Some(0));
        assert_eq!(manager.bsp_id(), Some(0));
        assert!(manager.is_bsp(0));

        // Register APs
        let ap1_idx = manager.register_cpu(1, false);
        assert_eq!(ap1_idx, Some(1));
        assert!(!manager.is_bsp(1));

        let ap2_idx = manager.register_cpu(2, false);
        assert_eq!(ap2_idx, Some(2));

        assert_eq!(manager.registered_count(), 3);
    }

    #[test]
    fn test_duplicate_registration() {
        let manager: CpuManager<4> = CpuManager::new();

        assert!(manager.register_cpu(1, false).is_some());
        assert!(manager.register_cpu(1, false).is_none()); // Should fail - duplicate
    }

    #[test]
    fn test_ap_state_management() {
        let manager: CpuManager<4> = CpuManager::new();
        manager.register_cpu(0, true);
        manager.register_cpu(1, false);

        // Check initial state
        assert_eq!(manager.get_ap_state(1), Some(ApState::InHoldingPen));

        // Change state
        assert!(manager.set_ap_state(1, ApState::Busy));
        assert_eq!(manager.get_ap_state(1), Some(ApState::Busy));

        // Cannot change BSP state
        assert!(!manager.set_ap_state(0, ApState::Halted));
    }

    #[test]
    fn test_for_each_ap() {
        let manager: CpuManager<4> = CpuManager::new();
        manager.register_cpu(0, true);
        manager.register_cpu(1, false);
        manager.register_cpu(2, false);

        let mut ap_ids = [0u32; 4];
        let mut count = 0;
        manager.for_each_ap(|id| {
            if count < 4 {
                ap_ids[count] = id;
                count += 1;
            }
        });

        assert_eq!(count, 2);
        assert!(ap_ids[..count].contains(&1));
        assert!(ap_ids[..count].contains(&2));
    }

    #[test]
    fn test_max_cpu_limit() {
        let manager: CpuManager<2> = CpuManager::new();
        assert!(manager.register_cpu(0, true).is_some());
        assert!(manager.register_cpu(1, false).is_some());
        assert!(manager.register_cpu(2, false).is_none()); // Should fail
    }

    #[test]
    fn test_count_aps_in_state() {
        let manager: CpuManager<4> = CpuManager::new();
        manager.register_cpu(0, true);
        manager.register_cpu(1, false);
        manager.register_cpu(2, false);

        assert_eq!(manager.count_aps_in_state(ApState::InHoldingPen), 2);
        assert_eq!(manager.count_aps_in_state(ApState::Busy), 0);

        manager.set_ap_state(1, ApState::Busy);
        assert_eq!(manager.count_aps_in_state(ApState::InHoldingPen), 1);
        assert_eq!(manager.count_aps_in_state(ApState::Busy), 1);
    }
}
