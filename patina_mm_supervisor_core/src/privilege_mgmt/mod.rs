//! Privilege Management for MM Supervisor Core
//!
//! This module manages the privilege level transitions between Ring 0 (supervisor)
//! and Ring 3 (user) in the MM environment. It provides:
//!
//! - One-time initialization of syscall/sysret MSRs
//! - Demotion of code execution to Ring 3 via `InvokeDemotedRoutine`
//! - Handling of syscall requests from Ring 3 code
//! - Call gate and TSS descriptor management for privilege transitions
//!
//! ## Architecture
//!
//! The privilege management follows the x86_64 syscall/sysret model:
//!
//! 1. **Initialization**: Configure MSR_IA32_STAR, MSR_IA32_LSTAR, MSR_IA32_EFER
//!    to set up syscall entry points and segment selectors.
//!
//! 2. **Demotion**: Use `InvokeDemotedRoutine` to transition from Ring 0 to Ring 3.
//!    This sets up call gates for return and prepares the Ring 3 stack.
//!
//! 3. **Syscall Entry**: When Ring 3 code executes `syscall`, the CPU jumps to
//!    the address in MSR_IA32_LSTAR (our `SyscallCenter`), which dispatches
//!    to the appropriate handler.
//!
//! 4. **Return**: Ring 3 code returns via call gate or syscall dispatcher returns
//!    via `sysret`.
//!
//! ## Segment Layout (from SmiException.nasm)
//!
//! ```text
//! PROTECTED_DS      = 0x20
//! LONG_CS_R0        = 0x38  (Ring 0 code segment)
//! LONG_DS_R0        = 0x40  (Ring 0 data segment)
//! LONG_CS_R3_PH     = 0x4B  (Ring 3 code segment placeholder)
//! LONG_DS_R3        = 0x53  (Ring 3 data segment)
//! LONG_CS_R3        = 0x5B  (Ring 3 code segment)
//! CALL_GATE_OFFSET  = 0x60  (Call gate descriptor offset)
//! TSS_SEL_OFFSET    = 0x70  (TSS selector offset)
//! TSS_DESC_OFFSET   = 0x80  (TSS descriptor offset)
//! ```
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

mod syscall_setup;
mod syscall_dispatcher;
mod call_gate;

pub use syscall_setup::{
    SyscallInterface, SyscallCache, SyscallSetupError,
};
pub use syscall_dispatcher::{
    SyscallDispatcher, SyscallIndex, SyscallResult,
};
pub use call_gate::{
    CallGateManager, SegmentSelectors,
};

// ============================================================================
// Segment Selector Constants
// ============================================================================

/// Protected mode data segment selector.
pub const PROTECTED_DS: u16 = 0x20;

/// Long mode Ring 0 code segment selector.
pub const LONG_CS_R0: u16 = 0x38;

/// Long mode Ring 0 data segment selector.
pub const LONG_DS_R0: u16 = 0x40;

/// Long mode Ring 3 code segment placeholder (for STAR MSR).
pub const LONG_CS_R3_PH: u16 = 0x4B;

/// Long mode Ring 3 data segment selector.
pub const LONG_DS_R3: u16 = 0x53;

/// Long mode Ring 3 code segment selector.
pub const LONG_CS_R3: u16 = 0x5B;

/// Call gate descriptor offset in GDT.
pub const CALL_GATE_OFFSET: u16 = 0x60;

/// TSS selector offset in GDT.
pub const TSS_SEL_OFFSET: u16 = 0x70;

/// TSS descriptor offset in GDT.
pub const TSS_DESC_OFFSET: u16 = 0x80;

// ============================================================================
// MSR Definitions
// ============================================================================

/// MSR_IA32_STAR - System Call Target Address Register
/// Contains the segment selectors for syscall/sysret.
pub const MSR_IA32_STAR: u32 = 0xC000_0081;

/// MSR_IA32_LSTAR - Long Mode System Call Target Address Register
/// Contains the RIP for 64-bit syscall entry.
pub const MSR_IA32_LSTAR: u32 = 0xC000_0082;

/// MSR_IA32_CSTAR - Compatibility Mode System Call Target Address Register
/// Contains the RIP for compatibility mode syscall (not used in pure 64-bit).
pub const MSR_IA32_CSTAR: u32 = 0xC000_0083;

/// MSR_IA32_SFMASK - System Call Flag Mask
/// Contains flags to be cleared on syscall.
pub const MSR_IA32_SFMASK: u32 = 0xC000_0084;

/// MSR_IA32_EFER - Extended Feature Enable Register
pub const MSR_IA32_EFER: u32 = 0xC000_0080;

/// MSR_IA32_GS_BASE - GS Base Address Register
pub const MSR_IA32_GS_BASE: u32 = 0xC000_0101;

/// MSR_IA32_KERNEL_GS_BASE - Kernel GS Base Address Register (swapped on swapgs)
pub const MSR_IA32_KERNEL_GS_BASE: u32 = 0xC000_0102;

// ============================================================================
// EFER Bits
// ============================================================================

/// EFER.SCE - System Call Enable bit
pub const EFER_SCE: u64 = 1 << 0;

/// EFER.LME - Long Mode Enable bit
pub const EFER_LME: u64 = 1 << 8;

/// EFER.LMA - Long Mode Active bit
pub const EFER_LMA: u64 = 1 << 10;

/// EFER.NXE - No-Execute Enable bit
pub const EFER_NXE: u64 = 1 << 11;

// ============================================================================
// Common Types
// ============================================================================

/// Current privilege level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PrivilegeLevel {
    /// Ring 0 - Supervisor/Kernel mode
    Ring0 = 0,
    /// Ring 3 - User mode
    Ring3 = 3,
}

impl PrivilegeLevel {
    /// Returns the numeric value of the privilege level.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Creates a PrivilegeLevel from a numeric value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Ring0),
            3 => Some(Self::Ring3),
            _ => None,
        }
    }
}

/// Result type for privilege management operations.
pub type PrivilegeResult<T> = Result<T, PrivilegeError>;

/// Errors that can occur during privilege management operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivilegeError {
    /// The privilege management system has not been initialized.
    NotInitialized,
    /// Already initialized (cannot re-initialize).
    AlreadyInitialized,
    /// Invalid CPU index.
    InvalidCpuIndex,
    /// Out of resources (memory allocation failed).
    OutOfResources,
    /// The operation is not ready (missing prerequisite).
    NotReady,
    /// Invalid parameter provided.
    InvalidParameter,
    /// Security violation detected.
    SecurityViolation,
    /// The syscall index is not supported.
    UnsupportedSyscall,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privilege_level() {
        assert_eq!(PrivilegeLevel::Ring0.as_u8(), 0);
        assert_eq!(PrivilegeLevel::Ring3.as_u8(), 3);
        assert_eq!(PrivilegeLevel::from_u8(0), Some(PrivilegeLevel::Ring0));
        assert_eq!(PrivilegeLevel::from_u8(3), Some(PrivilegeLevel::Ring3));
        assert_eq!(PrivilegeLevel::from_u8(1), None);
        assert_eq!(PrivilegeLevel::from_u8(2), None);
    }
}
