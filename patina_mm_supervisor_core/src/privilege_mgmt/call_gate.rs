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

use core::arch::{global_asm, asm};
use x86_64::{VirtAddr, structures::tss::TaskStateSegment};
use super::{
    CALL_GATE_OFFSET, TSS_SEL_OFFSET, TSS_DESC_OFFSET,
    LONG_CS_R0,
};

global_asm!(include_str!("call_gate_transfer.asm"));

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
    /// Sets the base address in the descriptor.
    pub fn set_base(&mut self, base: u64) {
        self.base_low = (base & 0xFFFF) as u16;
        self.base_mid_low = ((base >> 16) & 0xFF) as u8;
        self.base_mid_high = ((base >> 24) & 0xFF) as u8;
        self.base_high = ((base >> 32) & 0xFFFFFFFF) as u32;
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
// Standalone Functions
// ============================================================================

/// Gets the current GDT base address by reading the GDTR register.
/// # Safety
/// This function is safe to call as it only reads the GDTR register and does not modify
/// any state. However, it is marked unsafe because it uses inline assembly.
#[cfg(target_arch = "x86_64")]
pub unsafe fn get_current_gdt_base() -> u64 {
    // Get current GDT base
    let mut gdtr = GdtRegister::default();
    core::arch::asm!(
        "sgdt [{}]",
        in(reg) &mut gdtr,
        options(nostack, preserves_flags)
    );
    let gdt_base = gdtr.base;
    gdt_base
}

/// Sets up the call gate for returning from a demoted routine.
/// This function is called from assembly code (InvokeDemotedRoutine).
///
/// # Arguments
///
/// * `return_pointer` - Address to jump to when the call gate is invoked
///
/// # Safety
///
/// This modifies the GDT.
#[unsafe(no_mangle)]
#[cfg(target_arch = "x86_64")]
pub unsafe extern "efiapi" fn setup_call_gate(
    return_pointer: u64,
    cpl0_stack_ptr: u64,
) {
    // Get current GDT base
    let gdt_base = get_current_gdt_base();

    let call_gate_addr = gdt_base + CALL_GATE_OFFSET as u64;

    let tss_desc_addr = gdt_base + TSS_SEL_OFFSET as u64;
    let tss_addr = gdt_base + TSS_DESC_OFFSET as u64;

    // Mask page protection on GDT to allow writing to the call gate descriptor
    let mut cr4: u64;
    // SAFETY: This is safe because we are temporarily disabling page protection
    // on the GDT to update the call gate descriptor, which is necessary for the
    // call gate setup. We will restore protections after updating.
    unsafe {
        asm!("mov {}, cr4", out(reg) cr4);
        asm!("mov cr4, {}", in(reg) cr4 & !(1 << 7)); // Clear PGE to disable page protection on GDT
    };

    // Now program the call gate descriptor for the return address
    let call_gate = call_gate_addr as *mut CallGateDescriptor;

    // Update the call gate offset
    let mut desc = core::ptr::read_volatile(call_gate);
    desc.set_offset(return_pointer);
    desc.selector = LONG_CS_R0;
    // Type = 0xC (64-bit call gate), P = 1, DPL = 3 (Ring 3 can call)
    desc.type_attr = 0xEC;
    core::ptr::write_volatile(call_gate, desc);


    // Then program the TSS descriptor to point to the TSS (which contains the stack pointer for Ring 0)
    let tss_desc = tss_desc_addr as *mut TssDescriptor;
    let tss = tss_addr as *mut TaskStateSegment;

    // Update TSS descriptor to point to the TSS
    let mut desc = core::ptr::read_volatile(tss_desc);
    desc.set_base(tss_addr);
    core::ptr::write_volatile(tss_desc, desc);

    // Update RSP0 in the TSS
    let mut tss_data = core::ptr::read_volatile(tss);
    tss_data.privilege_stack_table[0] = VirtAddr::new(cpl0_stack_ptr);
    core::ptr::write_volatile(tss, tss_data);

    // Restore GDT read-only protection
    unsafe {
        asm!("mov cr4, {}", in(reg) cr4);
    }

    log::trace!("Call gate set to 0x{:016x}, CPL0 stack pointer set to 0x{:016x}", return_pointer, cpl0_stack_ptr);
}

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

}
