//! System Management Range Register (SMRR) configuration.
//!
//! This module programs the SMRR base and mask MSRs to protect the SMRAM region
//! from non SMM accesses, and enables the SMM Code Access Check feature. The
//! SMRRs are initialized on the first SMI entry and, on every subsequent SMI
//! rendezvous, enabled and finalized by setting the valid in the SMRR mask
//! register.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use crate::cpu::CPUID_VERSION_INFO;
use crate::cpu::read_msr;
use crate::cpu::write_msr;
use core::arch::x86_64::__cpuid;

const SIZE_4KB: u32 = 0x0000_1000;

// SMM Code Access Check related MSR and bit definitions
const MSR_SMM_MCA_CAP: u32 = 0x17D;
const SMM_CODE_ACCESS_CHK_BIT: u64 = 1 << 58;

const MSR_SMM_FEATURE_CONTROL: u32 = 0x4E0;
const SMM_FEATURE_CONTROL_LOCK_BIT: u64 = 1 << 0;
const SMM_CODE_CHK_EN_BIT: u64 = 1 << 2;

// SMM Range Register (SMRR) related MSR and bit definitions
const MSR_MTRR_CAP: u32 = 0x0FE;
const MTRR_CAP_SMRR_BIT: u64 = 1 << 11;
const MTRR_CAP_SMRR_EXT_BIT: u64 = 1 << 14;

const MSR_SMRR_BASE: u32 = 0x1F2;
const MSR_SMRR_MASK: u32 = 0x1F3;

const _MTRR_CACHE_WRITE_PROTECTED: u8 = 5;
const MTRR_CACHE_WRITE_BACK: u8 = 6;

const PHYS_ADDR_MASK: u64 = 0xFFFF_F000; // bits [31:12]
const MEMTYPE_MASK: u64 = 0x0000_00FF; // bits [7:0]
const MASK_BIT_10: u64 = 1 << 10;
const MASK_VALID_BIT: u64 = 1 << 11;

/// A region of SMRAM: a physical base address, a size in bytes, and whether it
/// was reported as pre-allocated in the HOB list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct SmramRegion {
    /// Physical base address of the region.
    pub base: u64,
    /// Size of the region, in bytes.
    pub size: u64,
    /// Whether the region was reported as pre-allocated (`EFI_ALLOCATED`).
    pub pre_allocated: bool,
}

/// Enables the SMM Code Access Check feature.
///
/// When enabled, the CPU raises a machine check if code is fetched from outside
/// the SMRR protected region while in SMM. This also sets the lock bit in the
/// SMM feature control MSR, after which the feature configuration cannot be
/// modified until the next processor reset.
///
/// # Panics
///
/// Panics if the CPU does not report support for SMM Code Access Check.
#[cfg_attr(coverage, coverage(off))]
pub(crate) fn configure_smm_code_access() {
    // SAFETY: MSR_SMM_MCA_CAP is a read-only architectural capability MSR that is
    // valid on all CPUs targeted by this code; reading it has no side effects.
    let smm_code_access_supported = (unsafe { read_msr(MSR_SMM_MCA_CAP) } & SMM_CODE_ACCESS_CHK_BIT) != 0;
    if !smm_code_access_supported {
        panic!("Unsupported CPU: SMM Code Access Check not supported");
    }

    // SAFETY: MSR_SMM_FEATURE_CONTROL is an architectural MSR; reading it has no
    // side effects.
    let smm_feature_control_msr = unsafe { read_msr(MSR_SMM_FEATURE_CONTROL) };
    let new_smm_feature_control_msr = smm_feature_control_msr | SMM_CODE_CHK_EN_BIT | SMM_FEATURE_CONTROL_LOCK_BIT;
    if new_smm_feature_control_msr != smm_feature_control_msr {
        // SAFETY: We only set the code check enable and lock bits, which is the
        // architecturally defined way to enable SMM Code Access Check. The write
        // is idempotent and performed only when the value actually changes.
        unsafe { write_msr(MSR_SMM_FEATURE_CONTROL, new_smm_feature_control_msr) };
    }
}

/// Returns the largest power of two less than or equal to `value`, or `0` if
/// `value` is `0`.
const fn get_power_of_two32(value: u32) -> u32 {
    if value == 0 { 0 } else { 1u32 << (31 - value.leading_zeros()) }
}

/// Sets the memory type field ([7:0]) of a raw SMRR base register value.
const fn base_reg_set_memtype(raw: u64, memtype: u8) -> u64 {
    (raw & !MEMTYPE_MASK) | (memtype as u64 & MEMTYPE_MASK)
}

/// Sets the physical base address field ([31:12]) of a raw SMRR base register
/// value.
const fn base_reg_set_base(raw: u64, base: u32) -> u64 {
    // `base` is a naturally aligned physical address whose low 12 bits are
    // zero, so its address bits [31:12] already sit at the register field's
    // positions; mask them in place without shifting.
    (raw & !PHYS_ADDR_MASK) | (base as u64 & PHYS_ADDR_MASK)
}

/// Sets the mask field ([31:12]) of a raw SMRR mask register value for the
/// given region `size`.
const fn mask_reg_set_mask(raw: u64, size: u32) -> u64 {
    // Mask field = ~(size - 1), which for a power-of-two `size` has zeros in the
    // low bits and ones above; mask its bits [31:12] in place without shifting.
    let mask_bits = (!(size.wrapping_sub(1))) as u64 & PHYS_ADDR_MASK;
    (raw & !PHYS_ADDR_MASK) | mask_bits
}

/// Returns `true` if bit 10 is set in a raw SMRR mask register value.
const fn mask_reg_bit10_set(raw: u64) -> bool {
    (raw & MASK_BIT_10) != 0
}

/// Validates that the SMRR base and size satisfy the hardware alignment and
/// size constraints.
///
/// A valid region must be at least 4 KiB, have a size that is a power of two,
/// and have a base address that is naturally aligned to its size.
pub(crate) const fn verify_smrr_base_size(smrr_base: u32, smrr_size: u32) -> bool {
    if smrr_size < SIZE_4KB
        || smrr_size != get_power_of_two32(smrr_size)
        || (smrr_base & !(smrr_size.wrapping_sub(1))) != smrr_base
    {
        return false;
    }

    true
}

/// Returns `true` if the CPU reports MTRR support via CPUID.
#[cfg_attr(coverage, coverage(off))]
fn is_mtrr_supported() -> bool {
    let version = __cpuid(CPUID_VERSION_INFO);
    let reg_edx = version.edx;
    (reg_edx & (1 << 12)) != 0
}

/// Returns `true` if the CPU reports SMRR support via `MTRR CAP`.
#[cfg_attr(coverage, coverage(off))]
fn is_smrr_supported() -> bool {
    // SAFETY: MSR_MTRR_CAP is a read-only architectural capability MSR; reading
    // it has no side effects.
    (unsafe { read_msr(MSR_MTRR_CAP) } & MTRR_CAP_SMRR_BIT) != 0
}

/// Returns `true` if the CPU reports the extended SMRR capability via `MTRR CAP`.
#[cfg_attr(coverage, coverage(off))]
fn is_smrr_ext_supported() -> bool {
    // SAFETY: MSR_MTRR_CAP is a read-only architectural capability MSR; reading
    // it has no side effects.
    (unsafe { read_msr(MSR_MTRR_CAP) } & MTRR_CAP_SMRR_EXT_BIT) != 0
}

/// Programs the SMRR base and mask registers to protect the given SMRAM region.
///
/// This is called on the first SMI entry. The region is configured as
/// write-back cacheable but is not yet enabled or finalized; call [`smrr_enable`]
/// to set the valid and bit-10 fields and activate the range.
///
/// # Panics
///
/// Panics if the CPU does not support MTRRs, SMRRs, or the extended SMRR
/// capability, or if the provided `range` fails [`verify_smrr_base_size`].
#[cfg_attr(coverage, coverage(off))]
pub(crate) fn smrr_initialize(range: SmramRegion) {
    let smrr_base = range.base as u32;
    let smrr_size = range.size as u32;

    if !is_mtrr_supported() {
        panic!("Unsupported CPU: MTRR not supported");
    }

    if !is_smrr_supported() {
        panic!("Unsupported CPU: SMRR not supported");
    }

    if !is_smrr_ext_supported() {
        panic!("Unsupported CPU: SMRR extended capability not supported");
    }

    if !verify_smrr_base_size(smrr_base, smrr_size) {
        panic!(
            "SMM Base/Size does not meet alignment/size requirement! Base: {:#X}, Size: {:#X}",
            smrr_base, smrr_size
        );
    }

    // SAFETY: SMRR support was verified above, so MSR_SMRR_BASE/MSR_SMRR_MASK
    // are valid architectural MSRs. `smrr_base`/`smrr_size` were validated by
    // `verify_smrr_base_size`, so the values written form a well-formed SMRR
    // range. The valid bit is left clear, so the range is not yet enforced.
    unsafe {
        let mut base = read_msr(MSR_SMRR_BASE);
        base = base_reg_set_memtype(base, MTRR_CACHE_WRITE_BACK);
        base = base_reg_set_base(base, smrr_base);
        write_msr(MSR_SMRR_BASE, base);

        let mut mask = read_msr(MSR_SMRR_MASK);
        mask = mask_reg_set_mask(mask, smrr_size);
        write_msr(MSR_SMRR_MASK, mask);
    }
}

/// Enables and finalizes the SMRR by setting the valid and bit-10 fields on the
/// mask register.
///
/// This is called on every subsequent SMI entry (rendezvous). If bit 10 is
/// already set, the function does nothing, since a finalized SMRR cannot be
/// modified until the next processor reset.
///
/// CPU MTRR/SMRR support is verified once in [`smrr_initialize`] on the first
/// SMI, so it is not re-checked here on every SMI.
#[cfg_attr(coverage, coverage(off))]
pub(crate) fn smrr_enable() {
    // SAFETY: SMRR support was verified in `smrr_initialize` on the first SMI,
    // so MSR_SMRR_MASK is a valid architectural MSR. We only set the valid and
    // bit-10 fields to enable and finalize the previously programmed range,
    // preserving all other bits. The write is skipped if the range is already
    // finalized.
    unsafe {
        let mut mask = read_msr(MSR_SMRR_MASK);
        if !mask_reg_bit10_set(mask) {
            mask |= MASK_VALID_BIT | MASK_BIT_10;
            write_msr(MSR_SMRR_MASK, mask);
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_smrr_verify_accepts_minimum_size() {
        // A 4 KiB region at base 0 is the smallest valid configuration.
        assert!(verify_smrr_base_size(0, SIZE_4KB));
    }

    #[test]
    fn test_smrr_verify_accepts_naturally_aligned_regions() {
        // Base is naturally aligned to the (power-of-two) size.
        assert!(verify_smrr_base_size(0x0080_0000, 0x0080_0000));
        assert!(verify_smrr_base_size(0x1000_0000, 0x1000_0000));
        assert!(verify_smrr_base_size(0x0000_8000, SIZE_4KB));
    }

    #[test]
    fn test_smrr_verify_rejects_size_below_4kb() {
        assert!(!verify_smrr_base_size(0, 0));
        assert!(!verify_smrr_base_size(0, SIZE_4KB - 1));
        assert!(!verify_smrr_base_size(0, 0x800));
    }

    #[test]
    fn test_smrr_verify_rejects_non_power_of_two_size() {
        assert!(!verify_smrr_base_size(0, 0x3000));
        assert!(!verify_smrr_base_size(0, 0x5000));
        assert!(!verify_smrr_base_size(0, SIZE_4KB + 1));
    }

    #[test]
    fn test_smrr_verify_rejects_misaligned_base() {
        // Base must be aligned to the size; these are off by 4 KiB or unaligned.
        assert!(!verify_smrr_base_size(SIZE_4KB, 0x0080_0000));
        assert!(!verify_smrr_base_size(0x0080_1000, 0x0080_0000));
        assert!(!verify_smrr_base_size(0x0000_1000, 0x0000_2000));
    }

    #[test]
    fn test_smrr_get_power_of_two32_returns_zero_for_zero() {
        assert_eq!(get_power_of_two32(0), 0);
    }

    #[test]
    fn test_smrr_get_power_of_two32_exact_powers_are_unchanged() {
        assert_eq!(get_power_of_two32(1), 1);
        assert_eq!(get_power_of_two32(2), 2);
        assert_eq!(get_power_of_two32(SIZE_4KB), SIZE_4KB);
        assert_eq!(get_power_of_two32(0x0080_0000), 0x0080_0000);
        assert_eq!(get_power_of_two32(0x8000_0000), 0x8000_0000);
    }

    #[test]
    fn test_smrr_get_power_of_two32_rounds_down_to_previous_power() {
        assert_eq!(get_power_of_two32(3), 2);
        assert_eq!(get_power_of_two32(0x0080_0001), 0x0080_0000);
        assert_eq!(get_power_of_two32(0x00FF_FFFF), 0x0080_0000);
        assert_eq!(get_power_of_two32(u32::MAX), 0x8000_0000);
    }

    #[test]
    fn test_smrr_base_reg_set_memtype_sets_low_byte_only() {
        // Memory type occupies bits [7:0]; all other bits must be preserved.
        assert_eq!(base_reg_set_memtype(0, MTRR_CACHE_WRITE_BACK), MTRR_CACHE_WRITE_BACK as u64);
        // Existing memory-type bits are replaced, not OR-ed.
        assert_eq!(base_reg_set_memtype(0xFF, MTRR_CACHE_WRITE_BACK), MTRR_CACHE_WRITE_BACK as u64);
        // Upper bits outside [7:0] are left untouched.
        assert_eq!(
            base_reg_set_memtype(0x1234_5600, MTRR_CACHE_WRITE_BACK),
            0x1234_5600 | MTRR_CACHE_WRITE_BACK as u64
        );
    }

    #[test]
    fn test_smrr_base_reg_set_base_sets_phys_addr_bits() {
        // Base address occupies bits [31:12].
        assert_eq!(base_reg_set_base(0, 0x0080_0000), 0x0080_0000);
        // Bits below [12] of the supplied base are ignored (masked out).
        assert_eq!(base_reg_set_base(0, 0x0080_0FFF), 0x0080_0000);
        // Existing [31:12] bits are replaced while [11:0] and [63:32] are preserved.
        assert_eq!(base_reg_set_base(0xFFFF_FFFF_FFFF_FFFF, 0x0080_0000), 0xFFFF_FFFF_0080_0FFF);
    }

    #[test]
    fn test_smrr_mask_reg_set_mask_computes_size_mask() {
        // Mask field = ~(size - 1) restricted to [31:12].
        assert_eq!(mask_reg_set_mask(0, SIZE_4KB), 0xFFFF_F000);
        assert_eq!(mask_reg_set_mask(0, 0x0080_0000), 0xFF80_0000);
        assert_eq!(mask_reg_set_mask(0, 0x1000_0000), 0xF000_0000);
        // Bits outside [31:12] of `raw` are preserved.
        assert_eq!(mask_reg_set_mask(0xFFFF_FFFF_0000_0FFF, 0x0080_0000), 0xFFFF_FFFF_FF80_0FFF);
    }

    #[test]
    fn test_smrr_mask_reg_bit10_detection() {
        assert!(!mask_reg_bit10_set(0));
        assert!(!mask_reg_bit10_set(MASK_VALID_BIT));
        assert!(mask_reg_bit10_set(MASK_BIT_10));
        assert!(mask_reg_bit10_set(MASK_BIT_10 | MASK_VALID_BIT));
    }
}
