//! Call Gate and TSS Management
//!
//! This module manages call gates and Task State Segment (TSS) descriptors
//! for privilege level transitions. Call gates provide an alternative mechanism
//! (besides syscall/sysret) for Ring 3 code to transition back to Ring 0.
//!
//! ## Call Gate Usage
//!
//! 1. When invoking a demoted routine, the supervisor sets up a call gate
//!    pointing to the return address.
//!
//! 2. The demoted routine in Ring 3 can return to Ring 0 by doing a far call
//!    to the call gate selector.
//!
//! 3. The CPU automatically transitions to Ring 0 and jumps to the address
//!    in the call gate descriptor.
//!
//! ## TSS Usage
//!
//! The TSS is used to specify the Ring 0 stack pointer (RSP0) that the CPU
//! will load when transitioning from Ring 3 to Ring 0 via an interrupt or
//! call gate.
//!

#![allow(unsafe_op_in_unsafe_fn)]

use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

use super::{
    CALL_GATE_OFFSET, TSS_SEL_OFFSET, TSS_DESC_OFFSET,
    LONG_CS_R0, LONG_CS_R3, LONG_DS_R0, LONG_DS_R3,
    PrivilegeError, PrivilegeResult,
};

// ============================================================================
// Segment Selectors
// ============================================================================

/// Collection of segment selectors used for privilege transitions.
#[derive(Debug, Clone, Copy)]
pub struct SegmentSelectors {
    /// Ring 0 code segment selector.
    pub cs_r0: u16,
    /// Ring 0 data segment selector.
    pub ds_r0: u16,
    /// Ring 3 code segment selector.
    pub cs_r3: u16,
    /// Ring 3 data segment selector.
    pub ds_r3: u16,
    /// Call gate selector.
    pub call_gate: u16,
    /// TSS selector.
    pub tss: u16,
}

impl Default for SegmentSelectors {
    fn default() -> Self {
        Self {
            cs_r0: LONG_CS_R0,
            ds_r0: LONG_DS_R0,
            cs_r3: LONG_CS_R3,
            ds_r3: LONG_DS_R3,
            call_gate: CALL_GATE_OFFSET,
            tss: TSS_SEL_OFFSET,
        }
    }
}

// ============================================================================
// GDT Descriptor Structures
// ============================================================================

/// 64-bit Call Gate Descriptor.
///
/// A call gate allows privilege level transitions through a far call instruction.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CallGateDescriptor {
    /// Offset bits 15:0
    pub offset_low: u16,
    /// Target code segment selector
    pub selector: u16,
    /// Reserved (must be 0) and IST (bits 2:0)
    pub ist: u8,
    /// Type (0xC = 64-bit call gate) and DPL
    pub type_attr: u8,
    /// Offset bits 31:16
    pub offset_mid: u16,
    /// Offset bits 63:32
    pub offset_high: u32,
    /// Reserved (must be 0)
    pub reserved: u32,
}

impl CallGateDescriptor {
    /// Creates a new call gate descriptor.
    ///
    /// # Arguments
    ///
    /// * `target_offset` - The target code address
    /// * `target_selector` - The target code segment selector
    /// * `dpl` - Descriptor Privilege Level (0-3)
    pub fn new(target_offset: u64, target_selector: u16, dpl: u8) -> Self {
        Self {
            offset_low: (target_offset & 0xFFFF) as u16,
            selector: target_selector,
            ist: 0,
            // Type = 0xC (64-bit call gate), P = 1, DPL in bits 6:5
            type_attr: 0x8C | ((dpl & 0x3) << 5),
            offset_mid: ((target_offset >> 16) & 0xFFFF) as u16,
            offset_high: ((target_offset >> 32) & 0xFFFFFFFF) as u32,
            reserved: 0,
        }
    }

    /// Gets the target offset from the descriptor.
    pub fn get_offset(&self) -> u64 {
        (self.offset_low as u64)
            | ((self.offset_mid as u64) << 16)
            | ((self.offset_high as u64) << 32)
    }

    /// Sets the target offset in the descriptor.
    pub fn set_offset(&mut self, offset: u64) {
        self.offset_low = (offset & 0xFFFF) as u16;
        self.offset_mid = ((offset >> 16) & 0xFFFF) as u16;
        self.offset_high = ((offset >> 32) & 0xFFFFFFFF) as u32;
    }
}

/// 64-bit TSS Descriptor (16 bytes in 64-bit mode).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TssDescriptor {
    /// Limit bits 15:0
    pub limit_low: u16,
    /// Base bits 15:0
    pub base_low: u16,
    /// Base bits 23:16
    pub base_mid_low: u8,
    /// Type and attributes
    pub type_attr: u8,
    /// Limit bits 19:16 and flags
    pub limit_flags: u8,
    /// Base bits 31:24
    pub base_mid_high: u8,
    /// Base bits 63:32
    pub base_high: u32,
    /// Reserved
    pub reserved: u32,
}

impl TssDescriptor {
    /// Creates a new TSS descriptor.
    ///
    /// # Arguments
    ///
    /// * `base` - Base address of the TSS
    /// * `limit` - Size of the TSS minus 1
    pub fn new(base: u64, limit: u32) -> Self {
        Self {
            limit_low: (limit & 0xFFFF) as u16,
            base_low: (base & 0xFFFF) as u16,
            base_mid_low: ((base >> 16) & 0xFF) as u8,
            // Type = 0x9 (64-bit TSS available), P = 1
            type_attr: 0x89,
            limit_flags: ((limit >> 16) & 0x0F) as u8,
            base_mid_high: ((base >> 24) & 0xFF) as u8,
            base_high: ((base >> 32) & 0xFFFFFFFF) as u32,
            reserved: 0,
        }
    }

    /// Gets the base address from the descriptor.
    pub fn get_base(&self) -> u64 {
        (self.base_low as u64)
            | ((self.base_mid_low as u64) << 16)
            | ((self.base_mid_high as u64) << 24)
            | ((self.base_high as u64) << 32)
    }

    /// Sets the base address in the descriptor.
    pub fn set_base(&mut self, base: u64) {
        self.base_low = (base & 0xFFFF) as u16;
        self.base_mid_low = ((base >> 16) & 0xFF) as u8;
        self.base_mid_high = ((base >> 24) & 0xFF) as u8;
        self.base_high = ((base >> 32) & 0xFFFFFFFF) as u32;
    }
}

/// 64-bit Task State Segment.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TaskStateSegment {
    /// Reserved
    pub reserved0: u32,
    /// RSP for privilege level 0
    pub rsp0: u64,
    /// RSP for privilege level 1
    pub rsp1: u64,
    /// RSP for privilege level 2
    pub rsp2: u64,
    /// Reserved
    pub reserved1: u64,
    /// Interrupt Stack Table pointers (IST1-IST7)
    pub ist: [u64; 7],
    /// Reserved
    pub reserved2: u64,
    /// Reserved
    pub reserved3: u16,
    /// I/O Map Base Address
    pub iomap_base: u16,
}

impl TaskStateSegment {
    /// Creates a new TSS with the specified RSP0.
    pub fn new(rsp0: u64) -> Self {
        Self {
            rsp0,
            iomap_base: core::mem::size_of::<Self>() as u16,
            ..Default::default()
        }
    }
}

// ============================================================================
// GDT Register
// ============================================================================

/// GDTR (GDT Register) structure.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GdtRegister {
    /// Size of the GDT minus 1
    pub limit: u16,
    /// Linear address of the GDT
    pub base: u64,
}

// ============================================================================
// Call Gate Manager
// ============================================================================

/// State for the call gate manager.
struct CallGateState {
    /// GDT base address per CPU.
    gdt_bases: [u64; 256],
    /// GDT step size (distance between per-CPU GDTs).
    gdt_step_size: usize,
    /// Number of CPUs.
    num_cpus: usize,
}

impl CallGateState {
    const fn new() -> Self {
        Self {
            gdt_bases: [0; 256],
            gdt_step_size: 0,
            num_cpus: 0,
        }
    }
}

/// Manages call gates and TSS descriptors for privilege transitions.
pub struct CallGateManager {
    /// Whether the manager has been initialized.
    initialized: AtomicBool,
    /// Internal state.
    state: Mutex<CallGateState>,
}

impl CallGateManager {
    /// Creates a new call gate manager.
    pub const fn new() -> Self {
        Self {
            initialized: AtomicBool::new(false),
            state: Mutex::new(CallGateState::new()),
        }
    }

    /// Initializes the call gate manager.
    ///
    /// # Arguments
    ///
    /// * `gdt_buffer` - Base address of the GDT buffer
    /// * `gdt_step_size` - Step size between per-CPU GDTs
    /// * `num_cpus` - Number of CPUs
    pub fn init(
        &self,
        gdt_buffer: u64,
        gdt_step_size: usize,
        num_cpus: usize,
    ) -> PrivilegeResult<()> {
        if self.initialized.swap(true, Ordering::SeqCst) {
            return Err(PrivilegeError::AlreadyInitialized);
        }

        let mut state = self.state.lock();
        state.gdt_step_size = gdt_step_size;
        state.num_cpus = num_cpus;

        // Calculate GDT base for each CPU
        for i in 0..num_cpus {
            state.gdt_bases[i] = gdt_buffer + (gdt_step_size as u64) * (i as u64);
        }

        log::info!(
            "CallGateManager initialized: {} CPUs, gdt_buffer=0x{:016x}, step=0x{:x}",
            num_cpus,
            gdt_buffer,
            gdt_step_size
        );

        Ok(())
    }

    /// Gets the GDT base for the current CPU.
    ///
    /// # Safety
    ///
    /// This reads the GDTR register.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn get_current_gdt_base(&self) -> u64 {
        let mut gdtr = GdtRegister::default();
        core::arch::asm!(
            "sgdt [{}]",
            in(reg) &mut gdtr,
            options(nostack, preserves_flags)
        );
        gdtr.base
    }

    /// Sets up the call gate for returning from a demoted routine.
    ///
    /// # Arguments
    ///
    /// * `return_pointer` - Address to jump to when the call gate is invoked
    ///
    /// # Safety
    ///
    /// This modifies the GDT.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn setup_call_gate(
        &self,
        return_pointer: u64
    ) -> PrivilegeResult<()> {
        let state = self.state.lock();

        let gdt_base = self.get_current_gdt_base();
        let call_gate_addr = gdt_base + CALL_GATE_OFFSET as u64;

        // TODO: Clear GDT read-only protection before writing
        // SmmClearGdtReadOnlyForThisProcessor()

        let call_gate = call_gate_addr as *mut CallGateDescriptor;

        // Update the call gate offset
        let mut desc = core::ptr::read_volatile(call_gate);
        desc.set_offset(return_pointer);
        desc.selector = LONG_CS_R0;
        // Type = 0xC (64-bit call gate), P = 1, DPL = 3 (Ring 3 can call)
        desc.type_attr = 0xEC;
        core::ptr::write_volatile(call_gate, desc);

        // TODO: Restore GDT read-only protection
        // SmmSetGdtReadOnlyForThisProcessor()

        log::trace!("Call gate set to 0x{:016x}", return_pointer);

        Ok(())
    }

    /// Sets up the TSS descriptor with the Ring 0 stack pointer.
    ///
    /// # Arguments
    ///
    /// * `cpl0_stack_ptr` - Ring 0 stack pointer to use on privilege transitions
    ///
    /// # Safety
    ///
    /// This modifies the GDT and TSS.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn setup_tss_descriptor(
        &self,
        cpl0_stack_ptr: u64,
    ) -> PrivilegeResult<()> {
        let gdt_base = self.get_current_gdt_base();
        let tss_desc_addr = gdt_base + TSS_SEL_OFFSET as u64;
        let tss_addr = gdt_base + TSS_DESC_OFFSET as u64;

        // TODO: Clear GDT read-only protection before writing

        let tss_desc = tss_desc_addr as *mut TssDescriptor;
        let tss = tss_addr as *mut TaskStateSegment;

        // Update TSS descriptor to point to the TSS
        let mut desc = core::ptr::read_volatile(tss_desc);
        desc.set_base(tss_addr);
        core::ptr::write_volatile(tss_desc, desc);

        // Update RSP0 in the TSS
        let mut tss_data = core::ptr::read_volatile(tss);
        tss_data.rsp0 = cpl0_stack_ptr;
        core::ptr::write_volatile(tss, tss_data);

        // TODO: Restore GDT read-only protection

        log::trace!("TSS RSP0 set to 0x{:016x}", cpl0_stack_ptr);

        Ok(())
    }

    /// Loads the TSS into TR register.
    ///
    /// # Safety
    ///
    /// This modifies the TR register.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn load_tss(&self) -> PrivilegeResult<()> {
        core::arch::asm!(
            "ltr {:x}",
            in(reg) TSS_SEL_OFFSET,
            options(nostack, preserves_flags)
        );
        Ok(())
    }
}

// ============================================================================
// Global Instance
// ============================================================================

/// Global call gate manager instance.
pub static CALL_GATE_MANAGER: CallGateManager = CallGateManager::new();

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_gate_descriptor_offset() {
        let mut desc = CallGateDescriptor::new(0x12345678_9ABCDEF0, LONG_CS_R0, 3);
        assert_eq!(desc.get_offset(), 0x12345678_9ABCDEF0);
        
        desc.set_offset(0xFEDCBA98_76543210);
        assert_eq!(desc.get_offset(), 0xFEDCBA98_76543210);
    }

    #[test]
    fn test_tss_descriptor_base() {
        let mut desc = TssDescriptor::new(0x12345678_9ABCDEF0, 0x1000);
        assert_eq!(desc.get_base(), 0x12345678_9ABCDEF0);
        
        desc.set_base(0xFEDCBA98_76543210);
        assert_eq!(desc.get_base(), 0xFEDCBA98_76543210);
    }

    #[test]
    fn test_segment_selectors_default() {
        let selectors = SegmentSelectors::default();
        assert_eq!(selectors.cs_r0, LONG_CS_R0);
        assert_eq!(selectors.ds_r0, LONG_DS_R0);
        assert_eq!(selectors.cs_r3, LONG_CS_R3);
        assert_eq!(selectors.ds_r3, LONG_DS_R3);
    }
}
