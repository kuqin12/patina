//! AMD64 SMRAM Save State Map
//!
//! Register-to-offset lookup table for the AMD 64-bit SMRAM save state
//! layout (`AMD_SMRAM_SAVE_STATE_MAP64`).  The save state area starts at
//! `SMBASE + 0xFC00` (the address stored in `CpuSaveState[CpuIndex]`).
//! All offsets are relative to that base.
//!
//! Reference: AMD64 Architecture Programmer's Manual Vol 2, Table 10-2;
//! MdePkg `AmdSmramSaveStateMap.h`.
//!
//! ## Key Differences from Intel
//!
//! - Segment selectors are 2 bytes (UINT16) vs Intel's 4-byte fields.
//! - GDT, IDT, and LDT limits **are** supported (Intel returns `None`).
//! - CR4 is 8 bytes (Intel stores only 4 bytes).
//! - All 8-byte registers are stored contiguously (Intel splits DT bases).
//! - IO information uses `IO_DWord` at offset 0x2C0 with a different bit
//!   layout from Intel's `IOMisc`.
//! - AMD64 always operates in 64-bit mode during SMM, so the LMA
//!   pseudo-register always returns 64-bit without checking EFER.LMA.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0

use super::{
    MmSaveStateRegister, ParsedIoInfo, RegisterInfo, VendorConstants, IO_TYPE_INPUT,
    IO_TYPE_OUTPUT, IO_WIDTH_UINT8, IO_WIDTH_UINT16, IO_WIDTH_UINT32,
};

// ============================================================================
// Vendor Constants
// ============================================================================

/// AMD-specific offsets and behaviour constants.
pub static VENDOR_CONSTANTS: VendorConstants = VendorConstants {
    smmrevid_offset: 0x02FC,
    io_info_offset: 0x02C0,
    efer_offset: 0x02D0,
    rax_offset: 0x03F8,
    min_rev_id_io: 0x30064,
    lma_always_64: true,
};

// ============================================================================
// IO_DWord Bit Definitions (AMD-specific)
// ============================================================================

/// IO_DWord bit 0: direction (0 = WRITE/OUT, 1 = READ/IN).
const IO_DIRECTION_IN: u32 = 1;
//const IO_DIRECTION_OUT: u32 = 0;

/// IO_DWord bits \[5:4\]: data size encoding.
const IO_SIZE_BYTE: u32 = 0;
const IO_SIZE_WORD: u32 = 1;
//const IO_SIZE_RESERVED: u32 = 2;
const IO_SIZE_DWORD: u32 = 3;

// ============================================================================
// Register Map
// ============================================================================
//
// AMD save state struct layout (from SMBASE + 0xFC00):
//
//   +0x000 – 0x1FF: Reserved (padding).
//   +0x200 – 0x25F: Segment descriptors (ES, CS, SS, DS, FS, GS) — 16 bytes each.
//   +0x260 – 0x29F: System descriptors (GDTR, IDTR, LDTR, TR) — 16 bytes each.
//   +0x2A0 – 0x2BF: MSRs (KernelGsBase, STAR, LSTAR, CSTAR).
//   +0x2C0:         IO_DWord (4 bytes).
//   +0x2D0:         EFER (8 bytes).
//   +0x2FC:         SMMRevId (4 bytes).
//   +0x338 – 0x3FF: Registers (DR7, DR6, CR4, CR3, CR0, RFLAGS, RIP, R15..R8,
//                               RBP, RSP, RBX, RDI, RSI, RDX, RCX, RAX).
//
// Each segment descriptor is 16 bytes:
//   +0: Selector (UINT16)
//   +2: Attributes (UINT16)
//   +4: Limit (UINT32)
//   +8: BaseLoDword (UINT32)
//   +C: BaseHiDword (UINT32)

/// Looks up the AMD64 save state register info for a PI register.
///
/// Returns `None` for pseudo-registers (IO, LMA, ProcessorId) and for
/// `LdtInfo` (not supported in AMD's save state map).
pub fn register_info(reg: MmSaveStateRegister) -> Option<RegisterInfo> {
    match reg {
        // ================================================================
        // Descriptor Table Bases (8-byte, contiguous on AMD)
        // ================================================================
        MmSaveStateRegister::GdtBase => Some(RegisterInfo {
            lo_offset: 0x0268,
            hi_offset: 0x026C,
            native_width: 8,
        }),
        MmSaveStateRegister::IdtBase => Some(RegisterInfo {
            lo_offset: 0x0278,
            hi_offset: 0x027C,
            native_width: 8,
        }),
        // NOTE: The C reference code has a copy-paste bug where both lo and
        // hi point to `_LDTRBaseLoDword` (0x288).  The correct hi offset is
        // `_LDTRBaseHiDword` (0x28C).
        MmSaveStateRegister::LdtBase => Some(RegisterInfo {
            lo_offset: 0x0288,
            hi_offset: 0x028C,
            native_width: 8,
        }),

        // ================================================================
        // Descriptor Table Limits (supported on AMD, not on Intel)
        // ================================================================
        //
        // GDT and IDT limits are architecturally 16-bit, stored in UINT32
        // fields.  Only the lower 2 bytes are meaningful.
        MmSaveStateRegister::GdtLimit => Some(RegisterInfo {
            lo_offset: 0x0264,
            hi_offset: 0,
            native_width: 2,
        }),
        MmSaveStateRegister::IdtLimit => Some(RegisterInfo {
            lo_offset: 0x0274,
            hi_offset: 0,
            native_width: 2,
        }),
        // LDT limit is a system-segment limit (up to 32 bits in long mode).
        MmSaveStateRegister::LdtLimit => Some(RegisterInfo {
            lo_offset: 0x0284,
            hi_offset: 0,
            native_width: 4,
        }),

        // LdtInfo is not supported.
        MmSaveStateRegister::LdtInfo => None,

        // ================================================================
        // Segment Selectors (2-byte on AMD — UINT16 selectors only)
        // ================================================================
        MmSaveStateRegister::Es => Some(RegisterInfo {
            lo_offset: 0x0200,
            hi_offset: 0,
            native_width: 2,
        }),
        MmSaveStateRegister::Cs => Some(RegisterInfo {
            lo_offset: 0x0210,
            hi_offset: 0,
            native_width: 2,
        }),
        MmSaveStateRegister::Ss => Some(RegisterInfo {
            lo_offset: 0x0220,
            hi_offset: 0,
            native_width: 2,
        }),
        MmSaveStateRegister::Ds => Some(RegisterInfo {
            lo_offset: 0x0230,
            hi_offset: 0,
            native_width: 2,
        }),
        MmSaveStateRegister::Fs => Some(RegisterInfo {
            lo_offset: 0x0240,
            hi_offset: 0,
            native_width: 2,
        }),
        MmSaveStateRegister::Gs => Some(RegisterInfo {
            lo_offset: 0x0250,
            hi_offset: 0,
            native_width: 2,
        }),
        MmSaveStateRegister::LdtrSel => Some(RegisterInfo {
            lo_offset: 0x0280,
            hi_offset: 0,
            native_width: 2,
        }),
        MmSaveStateRegister::TrSel => Some(RegisterInfo {
            lo_offset: 0x0290,
            hi_offset: 0,
            native_width: 2,
        }),

        // ================================================================
        // Debug Registers (8-byte, contiguous)
        // ================================================================
        MmSaveStateRegister::Dr7 => Some(RegisterInfo {
            lo_offset: 0x0338,
            hi_offset: 0x033C,
            native_width: 8,
        }),
        MmSaveStateRegister::Dr6 => Some(RegisterInfo {
            lo_offset: 0x0340,
            hi_offset: 0x0344,
            native_width: 8,
        }),

        // ================================================================
        // Extended Registers R8–R15 (8-byte, contiguous)
        //
        // Stored in reverse order: R8 is at the highest address (0x3B8)
        // down to R15 at 0x380.
        // ================================================================
        MmSaveStateRegister::R8 => Some(RegisterInfo {
            lo_offset: 0x03B8,
            hi_offset: 0x03BC,
            native_width: 8,
        }),
        MmSaveStateRegister::R9 => Some(RegisterInfo {
            lo_offset: 0x03B0,
            hi_offset: 0x03B4,
            native_width: 8,
        }),
        MmSaveStateRegister::R10 => Some(RegisterInfo {
            lo_offset: 0x03A8,
            hi_offset: 0x03AC,
            native_width: 8,
        }),
        MmSaveStateRegister::R11 => Some(RegisterInfo {
            lo_offset: 0x03A0,
            hi_offset: 0x03A4,
            native_width: 8,
        }),
        MmSaveStateRegister::R12 => Some(RegisterInfo {
            lo_offset: 0x0398,
            hi_offset: 0x039C,
            native_width: 8,
        }),
        MmSaveStateRegister::R13 => Some(RegisterInfo {
            lo_offset: 0x0390,
            hi_offset: 0x0394,
            native_width: 8,
        }),
        MmSaveStateRegister::R14 => Some(RegisterInfo {
            lo_offset: 0x0388,
            hi_offset: 0x038C,
            native_width: 8,
        }),
        MmSaveStateRegister::R15 => Some(RegisterInfo {
            lo_offset: 0x0380,
            hi_offset: 0x0384,
            native_width: 8,
        }),

        // ================================================================
        // General-Purpose Registers (8-byte, contiguous)
        //
        // AMD order (highest to lowest):
        //   RAX, RCX, RDX, RSI, RDI, RBX, RSP, RBP
        // ================================================================
        MmSaveStateRegister::Rax => Some(RegisterInfo {
            lo_offset: 0x03F8,
            hi_offset: 0x03FC,
            native_width: 8,
        }),
        MmSaveStateRegister::Rbx => Some(RegisterInfo {
            lo_offset: 0x03D0,
            hi_offset: 0x03D4,
            native_width: 8,
        }),
        MmSaveStateRegister::Rcx => Some(RegisterInfo {
            lo_offset: 0x03F0,
            hi_offset: 0x03F4,
            native_width: 8,
        }),
        MmSaveStateRegister::Rdx => Some(RegisterInfo {
            lo_offset: 0x03E8,
            hi_offset: 0x03EC,
            native_width: 8,
        }),
        MmSaveStateRegister::Rsp => Some(RegisterInfo {
            lo_offset: 0x03C8,
            hi_offset: 0x03CC,
            native_width: 8,
        }),
        MmSaveStateRegister::Rbp => Some(RegisterInfo {
            lo_offset: 0x03C0,
            hi_offset: 0x03C4,
            native_width: 8,
        }),
        MmSaveStateRegister::Rsi => Some(RegisterInfo {
            lo_offset: 0x03E0,
            hi_offset: 0x03E4,
            native_width: 8,
        }),
        MmSaveStateRegister::Rdi => Some(RegisterInfo {
            lo_offset: 0x03D8,
            hi_offset: 0x03DC,
            native_width: 8,
        }),
        MmSaveStateRegister::Rip => Some(RegisterInfo {
            lo_offset: 0x0378,
            hi_offset: 0x037C,
            native_width: 8,
        }),

        // ================================================================
        // Flags and Control Registers
        // ================================================================
        MmSaveStateRegister::Rflags => Some(RegisterInfo {
            lo_offset: 0x0370,
            hi_offset: 0x0374,
            native_width: 8,
        }),
        MmSaveStateRegister::Cr0 => Some(RegisterInfo {
            lo_offset: 0x0358,
            hi_offset: 0x035C,
            native_width: 8,
        }),
        MmSaveStateRegister::Cr3 => Some(RegisterInfo {
            lo_offset: 0x0350,
            hi_offset: 0x0354,
            native_width: 8,
        }),
        // CR4 is 8 bytes on AMD (vs 4 bytes on Intel).
        MmSaveStateRegister::Cr4 => Some(RegisterInfo {
            lo_offset: 0x0348,
            hi_offset: 0x034C,
            native_width: 8,
        }),

        // Pseudo-registers are not in the architectural register map.
        MmSaveStateRegister::Io | MmSaveStateRegister::Lma | MmSaveStateRegister::ProcessorId => {
            None
        }
    }
}

// ============================================================================
// I/O Field Parsing (AMD IO_DWord)
// ============================================================================

/// Parses AMD's `IO_DWord` field from the SMRAM save state.
///
/// AMD `IO_DWord` bit layout:
/// - Bit 0:      Direction — 0 = WRITE (OUT), 1 = READ (IN).
/// - Bits \[3:1\]:  Reserved.
/// - Bits \[5:4\]:  Data size — 0 = byte, 1 = word, 3 = dword.
/// - Bits \[15:6\]: Reserved.
/// - Bits \[31:16\]: I/O port address.
///
/// Returns `None` if the data-size encoding is invalid (value 2 is reserved).
pub fn parse_io_field(io_field: u32) -> Option<ParsedIoInfo> {
    let direction = io_field & 1;
    let size_enc = (io_field >> 4) & 0x3;
    let port = (io_field >> 16) & 0xFFFF;

    let io_type = if direction == IO_DIRECTION_IN {
        IO_TYPE_INPUT
    } else {
        IO_TYPE_OUTPUT
    };

    let (io_width, byte_count) = match size_enc {
        IO_SIZE_BYTE => (IO_WIDTH_UINT8, 1usize),
        IO_SIZE_WORD => (IO_WIDTH_UINT16, 2usize),
        IO_SIZE_DWORD => (IO_WIDTH_UINT32, 4usize),
        _ => return None, // Reserved encoding.
    };

    Some(ParsedIoInfo {
        io_type,
        io_width,
        byte_count,
        io_port: port,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------
    // Register map tests
    // ----------------------------------------------------------------

    #[test]
    fn test_gpr_offsets() {
        let rax = register_info(MmSaveStateRegister::Rax).unwrap();
        assert_eq!(rax.lo_offset, 0x03F8);
        assert_eq!(rax.hi_offset, 0x03FC);
        assert_eq!(rax.native_width, 8);

        let rcx = register_info(MmSaveStateRegister::Rcx).unwrap();
        assert_eq!(rcx.lo_offset, 0x03F0);
        assert_eq!(rcx.hi_offset, 0x03F4);
    }

    #[test]
    fn test_segment_selectors_are_2_bytes() {
        let cs = register_info(MmSaveStateRegister::Cs).unwrap();
        assert_eq!(cs.lo_offset, 0x0210);
        assert_eq!(cs.native_width, 2);
        assert_eq!(cs.hi_offset, 0);
    }

    #[test]
    fn test_descriptor_table_bases_contiguous() {
        let gdt = register_info(MmSaveStateRegister::GdtBase).unwrap();
        assert_eq!(gdt.lo_offset, 0x0268);
        assert_eq!(gdt.hi_offset, 0x026C);
        assert_eq!(gdt.native_width, 8);
        // AMD uses contiguous lo/hi (unlike Intel's split layout).
        assert_eq!(gdt.hi_offset, gdt.lo_offset + 4);
    }

    #[test]
    fn test_limits_supported_on_amd() {
        let gdt_limit = register_info(MmSaveStateRegister::GdtLimit).unwrap();
        assert_eq!(gdt_limit.native_width, 2);
        assert_eq!(gdt_limit.lo_offset, 0x0264);

        let idt_limit = register_info(MmSaveStateRegister::IdtLimit).unwrap();
        assert_eq!(idt_limit.native_width, 2);
        assert_eq!(idt_limit.lo_offset, 0x0274);

        let ldt_limit = register_info(MmSaveStateRegister::LdtLimit).unwrap();
        assert_eq!(ldt_limit.native_width, 4);
        assert_eq!(ldt_limit.lo_offset, 0x0284);
    }

    #[test]
    fn test_cr4_is_8_bytes_on_amd() {
        let cr4 = register_info(MmSaveStateRegister::Cr4).unwrap();
        assert_eq!(cr4.native_width, 8);
        assert_eq!(cr4.lo_offset, 0x0348);
        assert_eq!(cr4.hi_offset, 0x034C);
    }

    #[test]
    fn test_ldt_info_unsupported() {
        assert!(register_info(MmSaveStateRegister::LdtInfo).is_none());
    }

    #[test]
    fn test_pseudo_registers_return_none() {
        assert!(register_info(MmSaveStateRegister::Io).is_none());
        assert!(register_info(MmSaveStateRegister::Lma).is_none());
        assert!(register_info(MmSaveStateRegister::ProcessorId).is_none());
    }

    #[test]
    fn test_register_coverage() {
        // All architectural registers except LdtInfo should be supported.
        let supported_regs = [
            MmSaveStateRegister::GdtBase,
            MmSaveStateRegister::IdtBase,
            MmSaveStateRegister::LdtBase,
            MmSaveStateRegister::GdtLimit,
            MmSaveStateRegister::IdtLimit,
            MmSaveStateRegister::LdtLimit,
            MmSaveStateRegister::Es,
            MmSaveStateRegister::Cs,
            MmSaveStateRegister::Ss,
            MmSaveStateRegister::Ds,
            MmSaveStateRegister::Fs,
            MmSaveStateRegister::Gs,
            MmSaveStateRegister::LdtrSel,
            MmSaveStateRegister::TrSel,
            MmSaveStateRegister::Dr7,
            MmSaveStateRegister::Dr6,
            MmSaveStateRegister::R8,
            MmSaveStateRegister::R9,
            MmSaveStateRegister::R10,
            MmSaveStateRegister::R11,
            MmSaveStateRegister::R12,
            MmSaveStateRegister::R13,
            MmSaveStateRegister::R14,
            MmSaveStateRegister::R15,
            MmSaveStateRegister::Rax,
            MmSaveStateRegister::Rbx,
            MmSaveStateRegister::Rcx,
            MmSaveStateRegister::Rdx,
            MmSaveStateRegister::Rsp,
            MmSaveStateRegister::Rbp,
            MmSaveStateRegister::Rsi,
            MmSaveStateRegister::Rdi,
            MmSaveStateRegister::Rip,
            MmSaveStateRegister::Rflags,
            MmSaveStateRegister::Cr0,
            MmSaveStateRegister::Cr3,
            MmSaveStateRegister::Cr4,
        ];

        for reg in &supported_regs {
            assert!(
                register_info(*reg).is_some(),
                "Missing AMD lookup for {:?}",
                reg
            );
        }
    }

    // ----------------------------------------------------------------
    // IO_DWord parsing tests
    // ----------------------------------------------------------------

    #[test]
    fn test_parse_io_field_in_byte() {
        // Direction=1 (IN), Size=0 (byte), Port=0x80
        let io_field: u32 = (0x0080 << 16) | (IO_SIZE_BYTE << 4) | IO_DIRECTION_IN;
        let parsed = parse_io_field(io_field).unwrap();
        assert_eq!(parsed.io_type, IO_TYPE_INPUT);
        assert_eq!(parsed.io_width, IO_WIDTH_UINT8);
        assert_eq!(parsed.byte_count, 1);
        assert_eq!(parsed.io_port, 0x80);
    }

    #[test]
    fn test_parse_io_field_out_dword() {
        // Direction=0 (OUT), Size=3 (dword), Port=0xCF8
        let io_field: u32 = (0x0CF8 << 16) | (IO_SIZE_DWORD << 4) | 0;
        let parsed = parse_io_field(io_field).unwrap();
        assert_eq!(parsed.io_type, IO_TYPE_OUTPUT);
        assert_eq!(parsed.io_width, IO_WIDTH_UINT32);
        assert_eq!(parsed.byte_count, 4);
        assert_eq!(parsed.io_port, 0x0CF8);
    }

    #[test]
    fn test_parse_io_field_in_word() {
        // Direction=1 (IN), Size=1 (word), Port=0x3F8
        let io_field: u32 = (0x03F8 << 16) | (IO_SIZE_WORD << 4) | IO_DIRECTION_IN;
        let parsed = parse_io_field(io_field).unwrap();
        assert_eq!(parsed.io_type, IO_TYPE_INPUT);
        assert_eq!(parsed.io_width, IO_WIDTH_UINT16);
        assert_eq!(parsed.byte_count, 2);
        assert_eq!(parsed.io_port, 0x03F8);
    }

    #[test]
    fn test_parse_io_field_reserved_size() {
        // Direction=0, Size=2 (reserved) → None
        let io_field: u32 = (0x0080 << 16) | (2 << 4) | 0;
        assert!(parse_io_field(io_field).is_none());
    }

    #[test]
    fn test_idt_base_hi_is_correct() {
        // Verify we use the correct hi offset (0x27C), not the buggy
        // C code value (0x278 = lo offset repeated).
        let idt = register_info(MmSaveStateRegister::IdtBase).unwrap();
        assert_eq!(idt.lo_offset, 0x0278);
        assert_eq!(idt.hi_offset, 0x027C);
        assert_ne!(idt.lo_offset, idt.hi_offset, "hi must differ from lo");
    }

    #[test]
    fn test_ldt_base_hi_is_correct() {
        // Same bug fix for LdtBase: hi should be 0x28C, not 0x288.
        let ldt = register_info(MmSaveStateRegister::LdtBase).unwrap();
        assert_eq!(ldt.lo_offset, 0x0288);
        assert_eq!(ldt.hi_offset, 0x028C);
        assert_ne!(ldt.lo_offset, ldt.hi_offset, "hi must differ from lo");
    }
}
