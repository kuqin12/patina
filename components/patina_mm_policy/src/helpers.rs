//! Policy Helper Functions
//!
//! This module provides utility functions for policy manipulation:
//! - Dump/print policy for debugging
//! - Compare two policies (order-independent)
//! - Page table walking to generate memory policy

use crate::types::*;
use core::mem::size_of;

// ============================================================================
// Policy Dumping
// ============================================================================

/// Dumps a single memory policy entry for debugging.
pub fn dump_mem_policy_entry(desc: &MemDescriptorV1_0) {
    let r = if (desc.mem_attributes & RESOURCE_ATTR_READ) != 0 { "R" } else { "." };
    let w = if (desc.mem_attributes & RESOURCE_ATTR_WRITE) != 0 { "W" } else { "." };
    let x = if (desc.mem_attributes & RESOURCE_ATTR_EXECUTE) != 0 { "X" } else { "." };

    log::info!(
        "  MEM: [0x{:016x}-0x{:016x}] {}{}{}",
        desc.base_address,
        desc.base_address.saturating_add(desc.size).saturating_sub(1),
        r, w, x
    );
}

/// Dumps policy data for debugging (like `DumpSmmPolicyData`).
///
/// # Safety
///
/// The caller must ensure that `policy_ptr` points to a valid policy buffer.
pub unsafe fn dump_policy(policy_ptr: *const u8) {
    if policy_ptr.is_null() {
        log::error!("dump_policy: null pointer");
        return;
    }

    let policy = unsafe { &*(policy_ptr as *const SecurePolicyDataV1_0) };

    log::info!("SMM_SUPV_SECURE_POLICY_DATA_V1_0:");
    log::info!("  Version: {}.{}", policy.version_major, policy.version_minor);
    log::info!("  Size: 0x{:x}", policy.size);
    log::info!("  MemoryPolicyOffset: 0x{:x}", policy.memory_policy_offset);
    log::info!("  MemoryPolicyCount: 0x{:x}", policy.memory_policy_count);
    log::info!("  Flags: 0x{:x}", policy.flags);
    log::info!("  Capabilities: 0x{:x}", policy.capabilities);
    log::info!("  PolicyRootOffset: 0x{:x}", policy.policy_root_offset);
    log::info!("  PolicyRootCount: 0x{:x}", policy.policy_root_count);

    let policy_roots = unsafe { policy.get_policy_roots() };

    for (i, root) in policy_roots.iter().enumerate() {
        log::info!("Policy Root {}:", i);
        log::info!("  Version: {}", root.version);
        log::info!("  PolicyRootSize: {}", root.policy_root_size);
        log::info!("  Type: {}", root.policy_type);
        log::info!("  Offset: 0x{:x}", root.offset);
        log::info!("  Count: {}", root.count);
        log::info!(
            "  AccessAttr: {}",
            if root.access_attr == ACCESS_ATTR_ALLOW { "ALLOW" } else { "DENY" }
        );

        match root.policy_type {
            TYPE_MEM => {
                let descriptors = unsafe { root.get_mem_descriptors(policy_ptr) };
                for desc in descriptors {
                    dump_mem_policy_entry(desc);
                }
            }
            TYPE_IO => {
                let descriptors = unsafe { root.get_io_descriptors(policy_ptr) };
                for desc in descriptors {
                    let r = if (desc.attributes as u32 & RESOURCE_ATTR_READ) != 0 { "R" } else { "." };
                    let w = if (desc.attributes as u32 & RESOURCE_ATTR_WRITE) != 0 { "W" } else { "." };
                    log::info!(
                        "  IO: [0x{:04x}-0x{:04x}] {}{}",
                        desc.io_address,
                        (desc.io_address as u32)
                            .saturating_add(desc.length_or_width as u32)
                            .saturating_sub(1),
                        r, w
                    );
                }
            }
            TYPE_MSR => {
                let descriptors = unsafe { root.get_msr_descriptors(policy_ptr) };
                for desc in descriptors {
                    let r = if (desc.attributes as u32 & RESOURCE_ATTR_READ) != 0 { "R" } else { "." };
                    let w = if (desc.attributes as u32 & RESOURCE_ATTR_WRITE) != 0 { "W" } else { "." };
                    log::info!(
                        "  MSR: [0x{:08x}-0x{:08x}] {}{}",
                        desc.msr_address,
                        desc.msr_address
                            .saturating_add(desc.length as u32)
                            .saturating_sub(1),
                        r, w
                    );
                }
            }
            TYPE_INSTRUCTION => {
                let descriptors = unsafe { root.get_instruction_descriptors(policy_ptr) };
                for desc in descriptors {
                    let name = match desc.instruction_index {
                        0 => "CLI",
                        1 => "WBINVD",
                        2 => "HLT",
                        _ => "UNKNOWN",
                    };
                    let x = if (desc.attributes as u32 & RESOURCE_ATTR_EXECUTE) != 0 { "X" } else { "." };
                    log::info!("  INSTRUCTION: {} {}", name, x);
                }
            }
            TYPE_SAVE_STATE => {
                let descriptors = unsafe { root.get_save_state_descriptors(policy_ptr) };
                for desc in descriptors {
                    let field = match desc.map_field {
                        0 => "RAX",
                        1 => "IO_TRAP",
                        _ => "UNKNOWN",
                    };
                    let condition = match desc.access_condition {
                        0 => "Unconditional",
                        1 => "IoRead",
                        2 => "IoWrite",
                        _ => "Unknown",
                    };
                    log::info!(
                        "  SAVESTATE: {} attr=0x{:x} cond={}",
                        field,
                        desc.attributes,
                        condition
                    );
                }
            }
            _ => {
                log::error!("  Unknown policy type: {}", root.policy_type);
            }
        }
    }
}

// ============================================================================
// Policy Comparison
// ============================================================================

/// Compares two policies of a given type (order-independent).
///
/// # Safety
///
/// The caller must ensure that both policy pointers are valid.
pub unsafe fn compare_policy_with_type(
    policy1_ptr: *const u8,
    policy2_ptr: *const u8,
    policy_type: u32,
) -> bool {
    if policy1_ptr.is_null() || policy2_ptr.is_null() {
        return false;
    }

    let policy1 = unsafe { &*(policy1_ptr as *const SecurePolicyDataV1_0) };
    let policy2 = unsafe { &*(policy2_ptr as *const SecurePolicyDataV1_0) };

    // Find policy roots for the given type
    let roots1 = unsafe { policy1.get_policy_roots() };
    let roots2 = unsafe { policy2.get_policy_roots() };

    let root1 = roots1.iter().find(|r| r.policy_type == policy_type);
    let root2 = roots2.iter().find(|r| r.policy_type == policy_type);

    match (root1, root2) {
        (None, None) => true, // Neither has this type
        (Some(_), None) | (None, Some(_)) => false, // Only one has this type
        (Some(r1), Some(r2)) => {
            // Both have this type, compare
            if r1.count != r2.count || r1.access_attr != r2.access_attr {
                return false;
            }

            // Compare descriptors (order-independent)
            match policy_type {
                TYPE_MEM => {
                    let descs1 = unsafe { r1.get_mem_descriptors(policy1_ptr) };
                    let descs2 = unsafe { r2.get_mem_descriptors(policy2_ptr) };
                    compare_mem_descriptors(descs1, descs2)
                }
                TYPE_IO => {
                    let descs1 = unsafe { r1.get_io_descriptors(policy1_ptr) };
                    let descs2 = unsafe { r2.get_io_descriptors(policy2_ptr) };
                    compare_io_descriptors(descs1, descs2)
                }
                TYPE_MSR => {
                    let descs1 = unsafe { r1.get_msr_descriptors(policy1_ptr) };
                    let descs2 = unsafe { r2.get_msr_descriptors(policy2_ptr) };
                    compare_msr_descriptors(descs1, descs2)
                }
                TYPE_INSTRUCTION => {
                    let descs1 = unsafe { r1.get_instruction_descriptors(policy1_ptr) };
                    let descs2 = unsafe { r2.get_instruction_descriptors(policy2_ptr) };
                    compare_instruction_descriptors(descs1, descs2)
                }
                TYPE_SAVE_STATE => {
                    let descs1 = unsafe { r1.get_save_state_descriptors(policy1_ptr) };
                    let descs2 = unsafe { r2.get_save_state_descriptors(policy2_ptr) };
                    compare_save_state_descriptors(descs1, descs2)
                }
                _ => false,
            }
        }
    }
}

/// Compares memory policies (convenience wrapper).
///
/// # Safety
///
/// The caller must ensure that both policy pointers are valid.
pub unsafe fn compare_memory_policy(policy1_ptr: *const u8, policy2_ptr: *const u8) -> bool {
    unsafe { compare_policy_with_type(policy1_ptr, policy2_ptr, TYPE_MEM) }
}

// Helper functions for order-independent comparison

fn compare_mem_descriptors(descs1: &[MemDescriptorV1_0], descs2: &[MemDescriptorV1_0]) -> bool {
    if descs1.len() != descs2.len() {
        return false;
    }

    // For each descriptor in descs1, check if it exists in descs2
    for d1 in descs1 {
        let found = descs2.iter().any(|d2| {
            d1.base_address == d2.base_address
                && d1.size == d2.size
                && d1.mem_attributes == d2.mem_attributes
        });
        if !found {
            return false;
        }
    }
    true
}

fn compare_io_descriptors(descs1: &[IoDescriptorV1_0], descs2: &[IoDescriptorV1_0]) -> bool {
    if descs1.len() != descs2.len() {
        return false;
    }

    for d1 in descs1 {
        let found = descs2.iter().any(|d2| {
            d1.io_address == d2.io_address
                && d1.length_or_width == d2.length_or_width
                && d1.attributes == d2.attributes
        });
        if !found {
            return false;
        }
    }
    true
}

fn compare_msr_descriptors(descs1: &[MsrDescriptorV1_0], descs2: &[MsrDescriptorV1_0]) -> bool {
    if descs1.len() != descs2.len() {
        return false;
    }

    for d1 in descs1 {
        let found = descs2.iter().any(|d2| {
            d1.msr_address == d2.msr_address
                && d1.length == d2.length
                && d1.attributes == d2.attributes
        });
        if !found {
            return false;
        }
    }
    true
}

fn compare_instruction_descriptors(
    descs1: &[InstructionDescriptorV1_0],
    descs2: &[InstructionDescriptorV1_0],
) -> bool {
    if descs1.len() != descs2.len() {
        return false;
    }

    for d1 in descs1 {
        let found = descs2.iter().any(|d2| {
            d1.instruction_index == d2.instruction_index && d1.attributes == d2.attributes
        });
        if !found {
            return false;
        }
    }
    true
}

fn compare_save_state_descriptors(
    descs1: &[SaveStateDescriptorV1_0],
    descs2: &[SaveStateDescriptorV1_0],
) -> bool {
    if descs1.len() != descs2.len() {
        return false;
    }

    for d1 in descs1 {
        let found = descs2.iter().any(|d2| {
            d1.map_field == d2.map_field
                && d1.attributes == d2.attributes
                && d1.access_condition == d2.access_condition
        });
        if !found {
            return false;
        }
    }
    true
}

// ============================================================================
// Policy Validation
// ============================================================================

/// Errors that can occur during policy validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyCheckError {
    /// The policy pointer is null.
    NullPointer,
    /// Invalid policy version.
    InvalidVersion { major: u16, minor: u16 },
    /// A reserved field contains non-zero data.
    InvalidReservedField { policy_type: u32, entry_index: usize },
    /// The same policy type appears multiple times.
    DuplicatePolicyType { policy_type: u32 },
    /// Overlapping entries detected.
    OverlappingEntries { policy_type: u32, entry1: usize, entry2: usize },
    /// Duplicate entries detected.
    DuplicateEntries { policy_type: u32, entry1: usize, entry2: usize },
    /// Size mismatch.
    SizeMismatch { expected: usize, declared: usize },
    /// Unrecognized policy type.
    UnrecognizedPolicyType { policy_type: u32 },
    /// Unrecognized header bits.
    UnrecognizedHeaderBits,
    /// Unsupported attribute.
    UnsupportedAttribute { policy_type: u32, entry_index: usize, attributes: u32 },
    /// Conflicting condition.
    ConflictingCondition { entry_index: usize },
    /// Legacy memory policy detected.
    LegacyMemoryPolicyDetected,
    /// Range overflow.
    RangeOverflow,
}

/// Performs comprehensive security policy validation.
///
/// # Safety
///
/// The caller must ensure that `policy_ptr` points to a valid policy buffer.
pub unsafe fn security_policy_check(policy_ptr: *const u8) -> Result<(), PolicyCheckError> {
    if policy_ptr.is_null() {
        return Err(PolicyCheckError::NullPointer);
    }

    let policy = unsafe { &*(policy_ptr as *const SecurePolicyDataV1_0) };

    log::info!("Security policy check entry...");

    // Version check
    if !policy.is_valid_version() {
        return Err(PolicyCheckError::InvalidVersion {
            major: policy.version_major,
            minor: policy.version_minor,
        });
    }

    // Check for unrecognized header bits
    if policy.reserved != 0 || policy.flags != 0 || policy.capabilities != 0 {
        return Err(PolicyCheckError::UnrecognizedHeaderBits);
    }

    let mut total_scanned_size = size_of::<SecurePolicyDataV1_0>();
    let mut type_flags: u64 = 0;

    let policy_roots = unsafe { policy.get_policy_roots() };

    for root in policy_roots.iter() {
        let type_bit = 1u64 << root.policy_type;

        if (type_flags & type_bit) != 0 {
            return Err(PolicyCheckError::DuplicatePolicyType {
                policy_type: root.policy_type,
            });
        }
        type_flags |= type_bit;

        if !root.has_valid_reserved() {
            return Err(PolicyCheckError::InvalidReservedField {
                policy_type: root.policy_type,
                entry_index: 0,
            });
        }

        match root.policy_type {
            TYPE_IO => {
                unsafe { validate_io_policy(policy_ptr, root)? };
                total_scanned_size += (root.count as usize) * size_of::<IoDescriptorV1_0>();
            }
            TYPE_MEM => {
                unsafe { validate_mem_policy(policy_ptr, root)? };
                total_scanned_size += (root.count as usize) * size_of::<MemDescriptorV1_0>();
            }
            TYPE_MSR => {
                unsafe { validate_msr_policy(policy_ptr, root)? };
                total_scanned_size += (root.count as usize) * size_of::<MsrDescriptorV1_0>();
            }
            TYPE_INSTRUCTION => {
                unsafe { validate_instruction_policy(policy_ptr, root)? };
                total_scanned_size += (root.count as usize) * size_of::<InstructionDescriptorV1_0>();
            }
            TYPE_SAVE_STATE => {
                unsafe { validate_save_state_policy(policy_ptr, root)? };
                total_scanned_size += (root.count as usize) * size_of::<SaveStateDescriptorV1_0>();
            }
            _ => {
                return Err(PolicyCheckError::UnrecognizedPolicyType {
                    policy_type: root.policy_type,
                });
            }
        }

        total_scanned_size += size_of::<PolicyRootV1>();
    }

    if policy.memory_policy_count != 0 {
        return Err(PolicyCheckError::LegacyMemoryPolicyDetected);
    }

    if total_scanned_size != policy.size as usize {
        return Err(PolicyCheckError::SizeMismatch {
            expected: total_scanned_size,
            declared: policy.size as usize,
        });
    }

    log::info!("Security policy check passed.");
    Ok(())
}

// Validation helper functions

unsafe fn validate_io_policy(policy_base: *const u8, root: &PolicyRootV1) -> Result<(), PolicyCheckError> {
    let descriptors = unsafe { root.get_io_descriptors(policy_base) };

    for (i, desc) in descriptors.iter().enumerate() {
        if desc.reserved != 0 {
            return Err(PolicyCheckError::InvalidReservedField {
                policy_type: TYPE_IO,
                entry_index: i,
            });
        }
    }
    Ok(())
}

unsafe fn validate_mem_policy(policy_base: *const u8, root: &PolicyRootV1) -> Result<(), PolicyCheckError> {
    let descriptors = unsafe { root.get_mem_descriptors(policy_base) };

    for (i, desc) in descriptors.iter().enumerate() {
        if desc.reserved != 0 {
            return Err(PolicyCheckError::InvalidReservedField {
                policy_type: TYPE_MEM,
                entry_index: i,
            });
        }
    }
    Ok(())
}

unsafe fn validate_msr_policy(_policy_base: *const u8, _root: &PolicyRootV1) -> Result<(), PolicyCheckError> {
    // MSR descriptors don't have reserved fields
    Ok(())
}

unsafe fn validate_instruction_policy(policy_base: *const u8, root: &PolicyRootV1) -> Result<(), PolicyCheckError> {
    let descriptors = unsafe { root.get_instruction_descriptors(policy_base) };

    for (i, desc) in descriptors.iter().enumerate() {
        if desc.reserved != 0 {
            return Err(PolicyCheckError::InvalidReservedField {
                policy_type: TYPE_INSTRUCTION,
                entry_index: i,
            });
        }
    }
    Ok(())
}

unsafe fn validate_save_state_policy(policy_base: *const u8, root: &PolicyRootV1) -> Result<(), PolicyCheckError> {
    let descriptors = unsafe { root.get_save_state_descriptors(policy_base) };

    for (i, desc) in descriptors.iter().enumerate() {
        // Check for unsupported write attributes
        if (desc.attributes & (RESOURCE_ATTR_WRITE | RESOURCE_ATTR_COND_WRITE)) != 0 {
            return Err(PolicyCheckError::UnsupportedAttribute {
                policy_type: TYPE_SAVE_STATE,
                entry_index: i,
                attributes: desc.attributes,
            });
        }

        // Check for conflicting conditions
        if (desc.attributes & RESOURCE_ATTR_COND_READ) == 0
            && desc.access_condition != SVST_UNCONDITIONAL
        {
            return Err(PolicyCheckError::ConflictingCondition { entry_index: i });
        }

        if desc.reserved != 0 {
            return Err(PolicyCheckError::InvalidReservedField {
                policy_type: TYPE_SAVE_STATE,
                entry_index: i,
            });
        }
    }
    Ok(())
}

// ============================================================================
// Page Table Walking (Memory Policy Generation)
// ============================================================================

/// Memory policy builder for collecting memory descriptors from page table walking.
///
/// This is used to generate memory policy from page table entries.
pub struct MemoryPolicyBuilder {
    /// Current descriptor being built
    current: Option<MemDescriptorV1_0>,
    /// Maximum number of descriptors we can store
    max_count: usize,
    /// Buffer for descriptors
    buffer_ptr: *mut MemDescriptorV1_0,
    /// Current count of descriptors
    count: usize,
}

impl MemoryPolicyBuilder {
    /// Creates a new memory policy builder.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `buffer_ptr` points to a valid buffer
    /// with space for at least `max_count` descriptors.
    pub unsafe fn new(buffer_ptr: *mut MemDescriptorV1_0, max_count: usize) -> Self {
        Self {
            current: None,
            max_count,
            buffer_ptr,
            count: 0,
        }
    }

    /// Adds a memory region to the policy.
    ///
    /// Adjacent regions with the same attributes will be coalesced.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, or `Err(())` if the buffer is full.
    pub fn add_region(&mut self, base: u64, size: u64, attributes: u32) -> Result<(), ()> {
        let new_desc = MemDescriptorV1_0 {
            base_address: base,
            size,
            mem_attributes: attributes,
            reserved: 0,
        };

        if let Some(ref mut current) = self.current {
            // Check if we can coalesce with current
            let current_end = current.base_address.saturating_add(current.size);
            if base == current_end && attributes == current.mem_attributes {
                // Coalesce
                current.size = current.size.saturating_add(size);
                return Ok(());
            } else {
                // Flush current and start new
                self.flush_current()?;
            }
        }

        self.current = Some(new_desc);
        Ok(())
    }

    /// Flushes the current descriptor to the buffer.
    fn flush_current(&mut self) -> Result<(), ()> {
        if let Some(desc) = self.current.take() {
            if self.count >= self.max_count {
                return Err(());
            }

            // SAFETY: We checked bounds
            unsafe {
                *self.buffer_ptr.add(self.count) = desc;
            }
            self.count += 1;
        }
        Ok(())
    }

    /// Finishes building and returns the count of descriptors.
    pub fn finish(mut self) -> Result<usize, ()> {
        self.flush_current()?;
        Ok(self.count)
    }

    /// Gets the current count of descriptors.
    pub fn count(&self) -> usize {
        self.count + if self.current.is_some() { 1 } else { 0 }
    }
}

// ============================================================================
// Page Table Walking (x86_64 4-Level Paging)
// ============================================================================

/// Page table entry for x86_64 4-level paging.
///
/// This structure represents entries in PML4, PDPE, PDE, and PTE tables.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PageTableEntry {
    /// Raw 64-bit value of the page table entry.
    pub value: u64,
}

impl PageTableEntry {
    /// Present bit (bit 0): Entry is valid if set.
    pub const PRESENT: u64 = 1 << 0;
    /// Read/Write bit (bit 1): Writable if set.
    pub const READ_WRITE: u64 = 1 << 1;
    /// User/Supervisor bit (bit 2): User-mode accessible if set.
    pub const USER_SUPERVISOR: u64 = 1 << 2;
    /// Page Size bit (bit 7): Large page (2MB or 1GB) if set.
    pub const PAGE_SIZE: u64 = 1 << 7;
    /// No Execute bit (bit 63): Not executable if set.
    pub const NO_EXECUTE: u64 = 1 << 63;

    /// Address mask for 4KB page table base addresses.
    pub const ADDR_MASK_4K: u64 = 0x000F_FFFF_FFFF_F000;
    /// Address mask for 2MB large page addresses.
    pub const ADDR_MASK_2M: u64 = 0x000F_FFFF_FFE0_0000;
    /// Address mask for 1GB huge page addresses.
    pub const ADDR_MASK_1G: u64 = 0x000F_FFFF_C000_0000;

    /// Size of a 4KB page.
    pub const SIZE_4K: u64 = 0x1000;
    /// Size of a 2MB large page.
    pub const SIZE_2M: u64 = 0x20_0000;
    /// Size of a 1GB huge page.
    pub const SIZE_1G: u64 = 0x4000_0000;

    /// Creates a new page table entry from a raw value.
    #[inline]
    pub const fn new(value: u64) -> Self {
        Self { value }
    }

    /// Returns true if the entry is present.
    #[inline]
    pub const fn is_present(&self) -> bool {
        (self.value & Self::PRESENT) != 0
    }

    /// Returns true if the entry is writable.
    #[inline]
    pub const fn is_writable(&self) -> bool {
        (self.value & Self::READ_WRITE) != 0
    }

    /// Returns true if the entry is user-mode accessible.
    #[inline]
    pub const fn is_user(&self) -> bool {
        (self.value & Self::USER_SUPERVISOR) != 0
    }

    /// Returns true if this is a large/huge page (PS bit set).
    #[inline]
    pub const fn is_large_page(&self) -> bool {
        (self.value & Self::PAGE_SIZE) != 0
    }

    /// Returns true if the page is executable (NX bit NOT set).
    #[inline]
    pub const fn is_executable(&self) -> bool {
        (self.value & Self::NO_EXECUTE) == 0
    }

    /// Gets the physical address of the next-level page table (4KB aligned).
    #[inline]
    pub const fn next_table_addr(&self) -> u64 {
        self.value & Self::ADDR_MASK_4K
    }

    /// Gets the physical address of a 2MB large page.
    #[inline]
    pub const fn large_page_addr(&self) -> u64 {
        self.value & Self::ADDR_MASK_2M
    }

    /// Gets the physical address of a 1GB huge page.
    #[inline]
    pub const fn huge_page_addr(&self) -> u64 {
        self.value & Self::ADDR_MASK_1G
    }

    /// Gets the physical address of a 4KB page.
    #[inline]
    pub const fn page_addr(&self) -> u64 {
        self.value & Self::ADDR_MASK_4K
    }

    /// Converts page table entry attributes to policy memory attributes.
    ///
    /// Inherits R/W/X permissions from upper-level entries.
    #[inline]
    pub fn to_policy_attrs(&self, inherited_attrs: u32) -> u32 {
        if !self.is_present() {
            return 0;
        }

        let mut attrs = RESOURCE_ATTR_READ;

        if self.is_writable() {
            attrs |= RESOURCE_ATTR_WRITE;
        }

        if self.is_executable() {
            attrs |= RESOURCE_ATTR_EXECUTE;
        }

        // Inherit restrictions from upper-level tables
        attrs & inherited_attrs
    }
}

/// Errors that can occur during page table walking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageTableWalkError {
    /// Buffer is full, cannot add more descriptors.
    BufferFull,
    /// The CR3 value is invalid (null).
    InvalidCr3,
    /// Paging is not in the expected mode.
    UnsupportedPagingMode,
}

/// Callback type for checking if a buffer is inside MMRAM.
///
/// Returns `true` if the buffer `[base, base + size)` is fully inside MMRAM.
pub type IsInsideMmramFn = fn(base: u64, size: u64) -> bool;

/// Walks x86_64 4-level page tables and generates memory policy descriptors.
///
/// This function traverses the page table hierarchy starting from the PML4
/// table (pointed to by CR3), and for each mapped page, generates a memory
/// policy descriptor with the effective R/W/X attributes.
///
/// Adjacent pages with the same attributes are coalesced into single descriptors.
///
/// # Arguments
///
/// * `cr3` - The CR3 register value (physical address of PML4 table)
/// * `buffer` - Buffer to store the generated memory descriptors
/// * `max_count` - Maximum number of descriptors the buffer can hold
/// * `is_inside_mmram` - Callback to check if a region is inside MMRAM
///   (regions fully inside MMRAM are skipped)
///
/// # Returns
///
/// The number of memory descriptors generated, or an error.
///
/// # Safety
///
/// The caller must ensure that:
/// - `cr3` points to a valid PML4 table
/// - `buffer` has space for at least `max_count` descriptors
/// - The page table memory is accessible and won't change during the walk
///
/// # Example
///
/// ```rust,ignore
/// use patina_mm_policy::{walk_page_table, MemDescriptorV1_0, default_mmram_check};
///
/// let cr3 = read_cr3(); // Read from hardware
/// let mut buffer = [MemDescriptorV1_0::default(); 1024];
///
/// let count = unsafe {
///     walk_page_table(cr3, buffer.as_mut_ptr(), buffer.len(), default_mmram_check)?
/// };
///
/// println!("Generated {} memory policy descriptors", count);
/// ```
pub unsafe fn walk_page_table(
    cr3: u64,
    buffer: *mut MemDescriptorV1_0,
    max_count: usize,
    is_inside_mmram: IsInsideMmramFn,
) -> Result<usize, PageTableWalkError> {
    if cr3 == 0 || buffer.is_null() {
        return Err(PageTableWalkError::InvalidCr3);
    }

    let pml4_base = cr3 & PageTableEntry::ADDR_MASK_4K;
    let mut builder = unsafe { MemoryPolicyBuilder::new(buffer, max_count) };

    // Walk PML4 (512 entries)
    let pml4_table = pml4_base as *const PageTableEntry;

    for i in 0..512u64 {
        let pml4e = unsafe { *pml4_table.add(i as usize) };
        if !pml4e.is_present() {
            continue;
        }

        // Calculate inherited attributes from PML4 entry
        let pml4_attrs = pml4e.to_policy_attrs(
            RESOURCE_ATTR_READ | RESOURCE_ATTR_WRITE | RESOURCE_ATTR_EXECUTE,
        );

        // Walk PDPE (512 entries)
        let pdpe_table = pml4e.next_table_addr() as *const PageTableEntry;

        for j in 0..512u64 {
            let pdpe = unsafe { *pdpe_table.add(j as usize) };
            if !pdpe.is_present() {
                continue;
            }

            let pdpe_attrs = pdpe.to_policy_attrs(pml4_attrs);

            // Check for 1GB huge page
            if pdpe.is_large_page() {
                let page_addr = pdpe.huge_page_addr();
                let page_size = PageTableEntry::SIZE_1G;

                // Skip if fully inside MMRAM
                if is_inside_mmram(page_addr, page_size) {
                    continue;
                }

                if builder.add_region(page_addr, page_size, pdpe_attrs).is_err() {
                    return Err(PageTableWalkError::BufferFull);
                }
                continue;
            }

            // Walk PDE (512 entries)
            let pde_table = pdpe.next_table_addr() as *const PageTableEntry;

            for k in 0..512u64 {
                let pde = unsafe { *pde_table.add(k as usize) };
                if !pde.is_present() {
                    continue;
                }

                let pde_attrs = pde.to_policy_attrs(pdpe_attrs);

                // Check for 2MB large page
                if pde.is_large_page() {
                    let page_addr = pde.large_page_addr();
                    let page_size = PageTableEntry::SIZE_2M;

                    // Skip if fully inside MMRAM
                    if is_inside_mmram(page_addr, page_size) {
                        continue;
                    }

                    if builder.add_region(page_addr, page_size, pde_attrs).is_err() {
                        return Err(PageTableWalkError::BufferFull);
                    }
                    continue;
                }

                // Walk PTE (512 entries)
                let pte_table = pde.next_table_addr() as *const PageTableEntry;

                for l in 0..512u64 {
                    let pte = unsafe { *pte_table.add(l as usize) };
                    if !pte.is_present() {
                        continue;
                    }

                    let page_addr = pte.page_addr();
                    let page_size = PageTableEntry::SIZE_4K;

                    // Skip if fully inside MMRAM
                    if is_inside_mmram(page_addr, page_size) {
                        continue;
                    }

                    let pte_attrs = pte.to_policy_attrs(pde_attrs);

                    if builder.add_region(page_addr, page_size, pte_attrs).is_err() {
                        return Err(PageTableWalkError::BufferFull);
                    }
                }
            }
        }
    }

    builder.finish().map_err(|()| PageTableWalkError::BufferFull)
}

/// Statistics from page table walking.
#[derive(Debug, Clone, Copy, Default)]
pub struct PageTableWalkStats {
    /// Number of PML4 entries traversed.
    pub pml4_entries: usize,
    /// Number of PDPE entries traversed.
    pub pdpe_entries: usize,
    /// Number of PDE entries traversed.
    pub pde_entries: usize,
    /// Number of PTE entries traversed.
    pub pte_entries: usize,
    /// Number of 1GB huge pages found.
    pub huge_pages_1g: usize,
    /// Number of 2MB large pages found.
    pub large_pages_2m: usize,
    /// Number of 4KB pages found.
    pub pages_4k: usize,
    /// Number of pages skipped (inside MMRAM).
    pub skipped_mmram: usize,
}

/// Walks x86_64 4-level page tables with statistics.
///
/// This is the same as [`walk_page_table`] but also returns statistics
/// about the page table structure.
///
/// # Safety
///
/// Same requirements as [`walk_page_table`].
pub unsafe fn walk_page_table_with_stats(
    cr3: u64,
    buffer: *mut MemDescriptorV1_0,
    max_count: usize,
    is_inside_mmram: IsInsideMmramFn,
) -> Result<(usize, PageTableWalkStats), PageTableWalkError> {
    if cr3 == 0 || buffer.is_null() {
        return Err(PageTableWalkError::InvalidCr3);
    }

    let pml4_base = cr3 & PageTableEntry::ADDR_MASK_4K;
    let mut builder = unsafe { MemoryPolicyBuilder::new(buffer, max_count) };
    let mut stats = PageTableWalkStats::default();

    // Walk PML4 (512 entries)
    let pml4_table = pml4_base as *const PageTableEntry;

    for i in 0..512u64 {
        let pml4e = unsafe { *pml4_table.add(i as usize) };
        if !pml4e.is_present() {
            continue;
        }

        stats.pml4_entries += 1;

        let pml4_attrs = pml4e.to_policy_attrs(
            RESOURCE_ATTR_READ | RESOURCE_ATTR_WRITE | RESOURCE_ATTR_EXECUTE,
        );

        // Walk PDPE
        let pdpe_table = pml4e.next_table_addr() as *const PageTableEntry;

        for j in 0..512u64 {
            let pdpe = unsafe { *pdpe_table.add(j as usize) };
            if !pdpe.is_present() {
                continue;
            }

            stats.pdpe_entries += 1;
            let pdpe_attrs = pdpe.to_policy_attrs(pml4_attrs);

            // 1GB huge page
            if pdpe.is_large_page() {
                let page_addr = pdpe.huge_page_addr();
                let page_size = PageTableEntry::SIZE_1G;

                if is_inside_mmram(page_addr, page_size) {
                    stats.skipped_mmram += 1;
                    continue;
                }

                stats.huge_pages_1g += 1;
                if builder.add_region(page_addr, page_size, pdpe_attrs).is_err() {
                    return Err(PageTableWalkError::BufferFull);
                }
                continue;
            }

            // Walk PDE
            let pde_table = pdpe.next_table_addr() as *const PageTableEntry;

            for k in 0..512u64 {
                let pde = unsafe { *pde_table.add(k as usize) };
                if !pde.is_present() {
                    continue;
                }

                stats.pde_entries += 1;
                let pde_attrs = pde.to_policy_attrs(pdpe_attrs);

                // 2MB large page
                if pde.is_large_page() {
                    let page_addr = pde.large_page_addr();
                    let page_size = PageTableEntry::SIZE_2M;

                    if is_inside_mmram(page_addr, page_size) {
                        stats.skipped_mmram += 1;
                        continue;
                    }

                    stats.large_pages_2m += 1;
                    if builder.add_region(page_addr, page_size, pde_attrs).is_err() {
                        return Err(PageTableWalkError::BufferFull);
                    }
                    continue;
                }

                // Walk PTE
                let pte_table = pde.next_table_addr() as *const PageTableEntry;

                for l in 0..512u64 {
                    let pte = unsafe { *pte_table.add(l as usize) };
                    if !pte.is_present() {
                        continue;
                    }

                    stats.pte_entries += 1;
                    let page_addr = pte.page_addr();
                    let page_size = PageTableEntry::SIZE_4K;

                    if is_inside_mmram(page_addr, page_size) {
                        stats.skipped_mmram += 1;
                        continue;
                    }

                    stats.pages_4k += 1;
                    let pte_attrs = pte.to_policy_attrs(pde_attrs);

                    if builder.add_region(page_addr, page_size, pte_attrs).is_err() {
                        return Err(PageTableWalkError::BufferFull);
                    }
                }
            }
        }
    }

    let count = builder.finish().map_err(|()| PageTableWalkError::BufferFull)?;
    Ok((count, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dump_mem_policy_entry() {
        // Just verify it doesn't panic
        let desc = MemDescriptorV1_0 {
            base_address: 0x1000,
            size: 0x1000,
            mem_attributes: RESOURCE_ATTR_READ | RESOURCE_ATTR_WRITE,
            reserved: 0,
        };
        dump_mem_policy_entry(&desc);
    }

    #[test]
    fn test_compare_mem_descriptors() {
        let descs1 = [
            MemDescriptorV1_0 { base_address: 0x1000, size: 0x1000, mem_attributes: 1, reserved: 0 },
            MemDescriptorV1_0 { base_address: 0x2000, size: 0x1000, mem_attributes: 2, reserved: 0 },
        ];
        let descs2 = [
            MemDescriptorV1_0 { base_address: 0x2000, size: 0x1000, mem_attributes: 2, reserved: 0 },
            MemDescriptorV1_0 { base_address: 0x1000, size: 0x1000, mem_attributes: 1, reserved: 0 },
        ];

        // Order-independent comparison should succeed
        assert!(compare_mem_descriptors(&descs1, &descs2));
    }

    #[test]
    fn test_compare_mem_descriptors_mismatch() {
        let descs1 = [
            MemDescriptorV1_0 { base_address: 0x1000, size: 0x1000, mem_attributes: 1, reserved: 0 },
        ];
        let descs2 = [
            MemDescriptorV1_0 { base_address: 0x1000, size: 0x2000, mem_attributes: 1, reserved: 0 },
        ];

        assert!(!compare_mem_descriptors(&descs1, &descs2));
    }
}
