//! Syscall Interface Setup
//!
//! This module handles the initialization and configuration of syscall/sysret MSRs
//! for privilege level transitions. It manages per-CPU storage for MSR values and
//! the syscall cache structure used during ring transitions.
//!
//! ## MSR Configuration
//!
//! - **MSR_IA32_STAR**: Contains segment selectors for syscall/sysret
//!   - Bits 47:32 = SYSRET CS and SS (LONG_CS_R3_PH << 16)
//!   - Bits 31:16 = SYSCALL CS and SS (LONG_CS_R0)
//!
//! - **MSR_IA32_LSTAR**: Contains the 64-bit RIP for syscall entry (SyscallCenter)
//!
//! - **MSR_IA32_EFER**: Extended Feature Enable Register
//!   - Bit 0 (SCE) must be set to enable syscall/sysret
//!
//! - **MSR_IA32_KERNEL_GS_BASE**: Used with swapgs to switch between user and kernel
//!   GS base addresses, allowing access to per-CPU data in the syscall handler.
//!

#![allow(unsafe_op_in_unsafe_fn)]

use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

use super::{
    MSR_IA32_STAR, MSR_IA32_LSTAR, MSR_IA32_EFER,
    MSR_IA32_GS_BASE, MSR_IA32_KERNEL_GS_BASE,
    LONG_CS_R0, LONG_CS_R3_PH, EFER_SCE,
    PrivilegeError,
};

// ============================================================================
// Error Types
// ============================================================================

/// Errors specific to syscall setup operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyscallSetupError {
    /// The syscall interface has not been initialized.
    NotInitialized,
    /// Already initialized.
    AlreadyInitialized,
    /// Invalid CPU index (exceeds configured CPU count).
    InvalidCpuIndex,
    /// Out of resources.
    OutOfResources,
    /// MSR stores are not ready.
    NotReady,
}

impl From<SyscallSetupError> for PrivilegeError {
    fn from(e: SyscallSetupError) -> Self {
        match e {
            SyscallSetupError::NotInitialized => PrivilegeError::NotInitialized,
            SyscallSetupError::AlreadyInitialized => PrivilegeError::AlreadyInitialized,
            SyscallSetupError::InvalidCpuIndex => PrivilegeError::InvalidCpuIndex,
            SyscallSetupError::OutOfResources => PrivilegeError::OutOfResources,
            SyscallSetupError::NotReady => PrivilegeError::NotReady,
        }
    }
}

// ============================================================================
// Syscall Cache Structure
// ============================================================================

/// Per-CPU syscall cache structure.
///
/// This structure is pointed to by MSR_IA32_KERNEL_GS_BASE and accessed via
/// the `gs:` segment prefix after `swapgs` in the syscall entry handler.
///
/// Layout must match the assembly code in SysCallEntry.nasm:
/// - Offset 0x00: MmSupvRsp (Ring 0 stack pointer)
/// - Offset 0x08: SavedUserRsp (Saved Ring 3 stack pointer)
/// - Offset 0x10: OsGsBasePtr (Original OS GS base)
/// - Offset 0x18: OsGsSwapBasePtr (Original OS kernel GS base)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SyscallCache {
    /// Ring 0 stack pointer to use on syscall entry.
    /// Assembly offset: MM_SUPV_RSP (0x00)
    pub mm_supv_rsp: u64,
    /// Saved Ring 3 stack pointer from syscall.
    /// Assembly offset: SAVED_USER_RSP (0x08)
    pub saved_user_rsp: u64,
    /// Original OS GS base pointer (from MSR_IA32_GS_BASE).
    pub os_gs_base_ptr: u64,
    /// Original OS kernel GS base pointer (from MSR_IA32_KERNEL_GS_BASE).
    pub os_gs_swap_base_ptr: u64,
}

// ============================================================================
// Per-CPU MSR Storage
// ============================================================================

/// Per-CPU storage for MSR values that need to be saved/restored.
#[derive(Debug, Clone, Copy, Default)]
struct CpuMsrStorage {
    /// Saved MSR_IA32_STAR value.
    msr_star: u64,
    /// Saved MSR_IA32_LSTAR value.
    msr_lstar: u64,
    /// Saved MSR_IA32_EFER value.
    msr_efer: u64,
    /// Syscall cache for this CPU.
    syscall_cache: SyscallCache,
    /// Whether this CPU's MSRs have been configured.
    configured: bool,
}

// ============================================================================
// Syscall Interface
// ============================================================================

/// Internal state for the syscall interface.
///
/// ## Const Generic Parameters
///
/// * `MAX_CPUS` - The maximum number of CPUs that can be supported.
struct SyscallInterfaceState<const MAX_CPUS: usize> {
    /// Per-CPU MSR storage.
    cpu_storage: [CpuMsrStorage; MAX_CPUS],
    /// Number of CPUs configured.
    num_cpus: usize,
    /// Syscall entry point address (SyscallCenter).
    syscall_entry_point: u64,
    /// CPL3 stack array base address.
    cpl3_stack_base: u64,
    /// Per-CPU stack size.
    stack_size: usize,
}

impl<const MAX_CPUS: usize> SyscallInterfaceState<MAX_CPUS> {
    const fn new() -> Self {
        Self {
            cpu_storage: [CpuMsrStorage {
                msr_star: 0,
                msr_lstar: 0,
                msr_efer: 0,
                syscall_cache: SyscallCache {
                    mm_supv_rsp: 0,
                    saved_user_rsp: 0,
                    os_gs_base_ptr: 0,
                    os_gs_swap_base_ptr: 0,
                },
                configured: false,
            }; MAX_CPUS],
            num_cpus: 0,
            syscall_entry_point: 0,
            cpl3_stack_base: 0,
            stack_size: 0,
        }
    }
}

/// Syscall interface manager.
///
/// Manages the syscall/sysret MSR configuration for all CPUs and provides
/// the infrastructure for Ring 0 ↔ Ring 3 transitions.
///
/// ## Const Generic Parameters
///
/// * `MAX_CPUS` - The maximum number of CPUs that can be supported.
///   This should match `PlatformInfo::MAX_CPU_COUNT`.
pub struct SyscallInterface<const MAX_CPUS: usize> {
    /// Whether the interface has been initialized.
    initialized: AtomicBool,
    /// Internal state protected by mutex.
    state: Mutex<SyscallInterfaceState<MAX_CPUS>>,
}

impl<const MAX_CPUS: usize> SyscallInterface<MAX_CPUS> {
    /// Creates a new syscall interface.
    pub const fn new() -> Self {
        Self {
            initialized: AtomicBool::new(false),
            state: Mutex::new(SyscallInterfaceState::new()),
        }
    }

    /// Returns the maximum number of CPUs this interface supports.
    pub const fn max_cpus(&self) -> usize {
        MAX_CPUS
    }

    /// Initializes the syscall interface.
    ///
    /// This should be called once during BSP initialization.
    ///
    /// # Arguments
    ///
    /// * `num_cpus` - Total number of CPUs to support (must be <= MAX_CPUS)
    /// * `syscall_entry_point` - Address of the syscall entry function (SyscallCenter)
    /// * `cpl3_stack_base` - Base address of the CPL3 stack array
    /// * `stack_size` - Per-CPU stack size
    ///
    /// # Returns
    ///
    /// `Ok(())` if initialization succeeded, error otherwise.
    pub fn init(
        &self,
        num_cpus: usize,
        syscall_entry_point: u64,
        cpl3_stack_base: u64,
        stack_size: usize,
    ) -> Result<(), SyscallSetupError> {
        // Check if already initialized
        if self.initialized.swap(true, Ordering::SeqCst) {
            return Err(SyscallSetupError::AlreadyInitialized);
        }

        if num_cpus == 0 || num_cpus > MAX_CPUS {
            self.initialized.store(false, Ordering::SeqCst);
            return Err(SyscallSetupError::InvalidCpuIndex);
        }

        let mut state = self.state.lock();
        state.num_cpus = num_cpus;
        state.syscall_entry_point = syscall_entry_point;
        state.cpl3_stack_base = cpl3_stack_base;
        state.stack_size = stack_size;

        log::info!(
            "SyscallInterface<{}> initialized: {} CPUs, entry=0x{:016x}, cpl3_stack=0x{:016x}, stack_size=0x{:x}",
            MAX_CPUS,
            num_cpus,
            syscall_entry_point,
            cpl3_stack_base,
            stack_size
        );

        Ok(())
    }

    /// Checks if the syscall interface is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    /// Gets the CPL3 stack pointer for a specific CPU.
    ///
    /// The stack pointer is calculated as:
    /// `cpl3_stack_base + stack_size * (cpu_index + 1) - sizeof(usize)`
    ///
    /// This gives the top of the stack for the CPU (stacks grow downward).
    pub fn get_cpl3_stack(&self, cpu_index: usize) -> Result<u64, SyscallSetupError> {
        if !self.is_initialized() {
            return Err(SyscallSetupError::NotInitialized);
        }

        let state = self.state.lock();
        if cpu_index >= state.num_cpus {
            return Err(SyscallSetupError::InvalidCpuIndex);
        }

        // Calculate stack top: base + size * (index + 1) - sizeof(usize)
        let stack_top = state.cpl3_stack_base
            .wrapping_add((state.stack_size as u64) * ((cpu_index as u64) + 1))
            .wrapping_sub(core::mem::size_of::<usize>() as u64);

        Ok(stack_top)
    }

    /// Sets up the syscall MSRs for a specific CPU.
    ///
    /// This configures:
    /// - MSR_IA32_STAR with segment selectors
    /// - MSR_IA32_LSTAR with syscall entry point
    /// - MSR_IA32_EFER with SCE bit enabled
    /// - MSR_IA32_GS_BASE cleared
    /// - MSR_IA32_KERNEL_GS_BASE with pointer to syscall cache
    ///
    /// # Safety
    ///
    /// This function modifies MSRs and must be called on the target CPU.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn setup_cpl0_msr_star(&self, cpu_index: usize) -> Result<(), SyscallSetupError> {
        if !self.is_initialized() {
            return Err(SyscallSetupError::NotInitialized);
        }

        let mut state = self.state.lock();
        if cpu_index >= state.num_cpus {
            return Err(SyscallSetupError::InvalidCpuIndex);
        }

        // Extract syscall_entry before taking mutable borrow of storage
        let syscall_entry = state.syscall_entry_point;
        let storage: &mut CpuMsrStorage = &mut state.cpu_storage[cpu_index];

        // Save current MSR values
        storage.msr_star = Self::read_msr(MSR_IA32_STAR);
        storage.msr_lstar = Self::read_msr(MSR_IA32_LSTAR);
        storage.msr_efer = Self::read_msr(MSR_IA32_EFER);

        // Configure MSR_IA32_STAR
        // Low 32 bits: preserved (EIP for 32-bit syscall, not used in 64-bit)
        // Bits 47:32: SYSRET CS and SS base (Ring 3) - (LONG_CS_R3_PH << 16)
        // Bits 63:48: SYSCALL CS and SS base (Ring 0) - LONG_CS_R0
        let star_low = storage.msr_star & 0xFFFF_FFFF;
        let star_high = ((LONG_CS_R3_PH as u64) << 16) | (LONG_CS_R0 as u64);
        let new_star = (star_high << 32) | star_low;
        Self::write_msr(MSR_IA32_STAR, new_star);

        // Configure MSR_IA32_LSTAR with syscall entry point
        Self::write_msr(MSR_IA32_LSTAR, syscall_entry);

        // Enable SCE (System Call Enable) in EFER
        let new_efer = storage.msr_efer | EFER_SCE;
        Self::write_msr(MSR_IA32_EFER, new_efer);

        // Save original GS base and clear it
        storage.syscall_cache.os_gs_base_ptr = Self::read_msr(MSR_IA32_GS_BASE);
        Self::write_msr(MSR_IA32_GS_BASE, 0);

        // Save original kernel GS base and set it to point to our syscall cache
        storage.syscall_cache.os_gs_swap_base_ptr = Self::read_msr(MSR_IA32_KERNEL_GS_BASE);
        let cache_ptr = &storage.syscall_cache as *const SyscallCache as u64;
        Self::write_msr(MSR_IA32_KERNEL_GS_BASE, cache_ptr);

        storage.configured = true;

        log::debug!(
            "CPU {} syscall MSRs configured: STAR=0x{:016x}, LSTAR=0x{:016x}, EFER=0x{:016x}",
            cpu_index,
            new_star,
            syscall_entry,
            new_efer
        );

        Ok(())
    }

    /// Restores the original MSR values for a specific CPU.
    ///
    /// # Safety
    ///
    /// This function modifies MSRs and must be called on the target CPU.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn restore_cpl0_msr_star(&self, cpu_index: usize) -> Result<(), SyscallSetupError> {
        if !self.is_initialized() {
            return Err(SyscallSetupError::NotInitialized);
        }

        let state = self.state.lock();
        if cpu_index >= state.num_cpus {
            return Err(SyscallSetupError::InvalidCpuIndex);
        }

        let storage = &state.cpu_storage[cpu_index];
        if !storage.configured {
            return Err(SyscallSetupError::NotReady);
        }

        // Restore all MSRs to their original values
        Self::write_msr(MSR_IA32_LSTAR, storage.msr_lstar);
        Self::write_msr(MSR_IA32_STAR, storage.msr_star);
        Self::write_msr(MSR_IA32_EFER, storage.msr_efer);
        Self::write_msr(MSR_IA32_GS_BASE, storage.syscall_cache.os_gs_base_ptr);
        Self::write_msr(MSR_IA32_KERNEL_GS_BASE, storage.syscall_cache.os_gs_swap_base_ptr);

        log::debug!("CPU {} syscall MSRs restored", cpu_index);

        Ok(())
    }

    /// Updates the Ring 0 stack pointer in the syscall cache for a CPU.
    ///
    /// This is called to set the stack pointer that will be used when
    /// entering Ring 0 via syscall.
    pub fn update_cpl0_stack_ptr(&self, cpu_index: usize, cpl0_stack_ptr: u64) -> Result<(), SyscallSetupError> {
        if !self.is_initialized() {
            return Err(SyscallSetupError::NotInitialized);
        }

        if cpl0_stack_ptr == 0 {
            return Err(SyscallSetupError::NotReady);
        }

        let mut state = self.state.lock();
        if cpu_index >= state.num_cpus {
            return Err(SyscallSetupError::InvalidCpuIndex);
        }

        state.cpu_storage[cpu_index].syscall_cache.mm_supv_rsp = cpl0_stack_ptr;

        Ok(())
    }

    /// Gets the syscall cache for a specific CPU.
    pub fn get_syscall_cache(&self, cpu_index: usize) -> Result<SyscallCache, SyscallSetupError> {
        if !self.is_initialized() {
            return Err(SyscallSetupError::NotInitialized);
        }

        let state = self.state.lock();
        if cpu_index >= state.num_cpus {
            return Err(SyscallSetupError::InvalidCpuIndex);
        }

        Ok(state.cpu_storage[cpu_index].syscall_cache)
    }

    /// Initializes syscall MSRs for a specific CPU.
    ///
    /// This is a convenience wrapper around `setup_cpl0_msr_star` that handles
    /// the common case of per-core initialization.
    ///
    /// # Arguments
    ///
    /// * `cpu_index` - Index of the CPU to initialize
    ///
    /// # Returns
    ///
    /// `Ok(())` if the CPU was successfully initialized, error otherwise.
    ///
    /// # Safety
    ///
    /// This function is safe to call but internally uses unsafe operations
    /// to configure MSRs. It must be called on the target CPU.
    pub fn init_for_cpu(&self, cpu_index: usize) -> Result<(), SyscallSetupError> {
        if !self.is_initialized() {
            // If the interface hasn't been initialized yet, just return OK
            // This can happen if BSP init hasn't set up the interface yet
            log::trace!("SyscallInterface not yet initialized, skipping CPU {} init", cpu_index);
            return Ok(());
        }

        // SAFETY: This must be called on the target CPU.
        // The caller (per_core_init) ensures this is called on each CPU for itself.
        unsafe {
            self.setup_cpl0_msr_star(cpu_index)
        }
    }

    // ========================================================================
    // MSR Helper Functions
    // ========================================================================

    /// Reads an MSR.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn read_msr(msr: u32) -> u64 {
        let (low, high): (u32, u32);
        core::arch::asm!(
            "rdmsr",
            in("ecx") msr,
            out("eax") low,
            out("edx") high,
            options(nomem, nostack, preserves_flags)
        );
        ((high as u64) << 32) | (low as u64)
    }

    /// Writes an MSR.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn write_msr(msr: u32, value: u64) {
        let low = value as u32;
        let high = (value >> 32) as u32;
        core::arch::asm!(
            "wrmsr",
            in("ecx") msr,
            in("eax") low,
            in("edx") high,
            options(nomem, nostack, preserves_flags)
        );
    }
}

impl<const MAX_CPUS: usize> Default for SyscallInterface<MAX_CPUS> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syscall_cache_layout() {
        // Verify the layout matches assembly expectations
        use core::mem::offset_of;
        assert_eq!(offset_of!(SyscallCache, mm_supv_rsp), 0x00);
        assert_eq!(offset_of!(SyscallCache, saved_user_rsp), 0x08);
        assert_eq!(offset_of!(SyscallCache, os_gs_base_ptr), 0x10);
        assert_eq!(offset_of!(SyscallCache, os_gs_swap_base_ptr), 0x18);
    }

    #[test]
    fn test_cpl3_stack_calculation() {
        let interface: SyscallInterface<8> = SyscallInterface::new();
        
        // Initialize with known values: num_cpus=4, entry=0x1000, cpl3_stack_base=0x10000, stack_size=0x4000
        interface.init(4, 0x1000, 0x10000, 0x4000).unwrap();
        
        // CPU 0: base + 0x4000 * 1 - 8 = 0x10000 + 0x4000 - 8 = 0x13FF8
        assert_eq!(interface.get_cpl3_stack(0).unwrap(), 0x13FF8);
        
        // CPU 1: base + 0x4000 * 2 - 8 = 0x10000 + 0x8000 - 8 = 0x17FF8
        assert_eq!(interface.get_cpl3_stack(1).unwrap(), 0x17FF8);
    }

    #[test]
    fn test_init_twice_fails() {
        let interface: SyscallInterface<8> = SyscallInterface::new();
        assert!(interface.init(4, 0x1000, 0x10000, 0x4000).is_ok());
        assert_eq!(
            interface.init(4, 0x1000, 0x10000, 0x4000),
            Err(SyscallSetupError::AlreadyInitialized)
        );
    }

    #[test]
    fn test_invalid_cpu_index() {
        let interface: SyscallInterface<8> = SyscallInterface::new();
        interface.init(4, 0x1000, 0x10000, 0x4000).unwrap();
        
        assert_eq!(
            interface.get_cpl3_stack(4),
            Err(SyscallSetupError::InvalidCpuIndex)
        );
        assert_eq!(
            interface.get_cpl3_stack(100),
            Err(SyscallSetupError::InvalidCpuIndex)
        );
    }

    #[test]
    fn test_max_cpus_exceeded() {
        let interface: SyscallInterface<4> = SyscallInterface::new();
        // Try to init with more CPUs than the const generic allows
        assert_eq!(
            interface.init(8, 0x1000, 0x10000, 0x4000),
            Err(SyscallSetupError::InvalidCpuIndex)
        );
    }

    #[test]
    fn test_max_cpus_accessor() {
        let interface: SyscallInterface<16> = SyscallInterface::new();
        assert_eq!(interface.max_cpus(), 16);
    }
}
