//! MM Supervisor Secure Policy Data Structures
//!
//! Rust definitions matching the C structures in `SmmSecurePolicy.h`.

use core::slice;

// ============================================================================
// Policy Descriptor Types
// ============================================================================

/// Memory policy descriptor type.
pub const TYPE_MEM: u32 = 1;
/// I/O policy descriptor type.
pub const TYPE_IO: u32 = 2;
/// MSR policy descriptor type.
pub const TYPE_MSR: u32 = 3;
/// Instruction policy descriptor type.
pub const TYPE_INSTRUCTION: u32 = 4;
/// Save state policy descriptor type.
pub const TYPE_SAVE_STATE: u32 = 5;

// ============================================================================
// Access Attributes
// ============================================================================

/// Access attribute: Allow access to resources described by this policy root.
pub const ACCESS_ATTR_ALLOW: u8 = 0;
/// Access attribute: Deny access to resources described by this policy root.
pub const ACCESS_ATTR_DENY: u8 = 1;

// ============================================================================
// Resource Attributes
// ============================================================================

/// Resource attribute: Read access.
pub const RESOURCE_ATTR_READ: u32 = 0x01;
/// Resource attribute: Write access.
pub const RESOURCE_ATTR_WRITE: u32 = 0x02;
/// Resource attribute: Execute access.
pub const RESOURCE_ATTR_EXECUTE: u32 = 0x04;
/// Resource attribute: Strict width (for I/O - must match exact width).
pub const RESOURCE_ATTR_STRICT_WIDTH: u32 = 0x08;
/// Resource attribute: Conditional read access.
pub const RESOURCE_ATTR_COND_READ: u32 = 0x10;
/// Resource attribute: Conditional write access.
pub const RESOURCE_ATTR_COND_WRITE: u32 = 0x20;

// ============================================================================
// Instruction Indices
// ============================================================================

/// Instruction index for CLI.
pub const INSTRUCTION_CLI: u16 = 0;
/// Instruction index for WBINVD.
pub const INSTRUCTION_WBINVD: u16 = 1;
/// Instruction index for HLT.
pub const INSTRUCTION_HLT: u16 = 2;
/// Total count of privileged instructions tracked.
pub const INSTRUCTION_COUNT: u16 = 3;

/// Privileged instruction types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum Instruction {
    /// CLI - Clear Interrupt Flag
    Cli = 0,
    /// WBINVD - Write Back and Invalidate Cache
    Wbinvd = 1,
    /// HLT - Halt
    Hlt = 2,
}

impl Instruction {
    /// Convert to instruction index.
    pub fn as_index(self) -> u16 {
        self as u16
    }

    /// Create from instruction index.
    pub fn from_index(index: u16) -> Option<Self> {
        match index {
            0 => Some(Self::Cli),
            1 => Some(Self::Wbinvd),
            2 => Some(Self::Hlt),
            _ => None,
        }
    }
}

// ============================================================================
// Save State Map Fields
// ============================================================================

/// Save state field: RAX register.
pub const SVST_RAX: u32 = 0;
/// Save state field: I/O trap information.
pub const SVST_IO_TRAP: u32 = 1;
/// Total count of save state fields tracked.
pub const SVST_COUNT: u32 = 2;

/// Save state map fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SaveStateField {
    /// RAX register
    Rax = 0,
    /// I/O trap information
    IoTrap = 1,
}

impl SaveStateField {
    /// Convert to field index.
    pub fn as_index(self) -> u32 {
        self as u32
    }

    /// Create from field index.
    pub fn from_index(index: u32) -> Option<Self> {
        match index {
            0 => Some(Self::Rax),
            1 => Some(Self::IoTrap),
            _ => None,
        }
    }
}

// ============================================================================
// Save State Access Conditions
// ============================================================================

/// Save state access is unconditional.
pub const SVST_UNCONDITIONAL: u32 = 0;
/// Save state access is conditional on I/O read trap.
pub const SVST_CONDITION_IO_RD: u32 = 1;
/// Save state access is conditional on I/O write trap.
pub const SVST_CONDITION_IO_WR: u32 = 2;
/// Total count of save state conditions.
pub const SVST_CONDITION_COUNT: u32 = 3;

/// Save state access conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SaveStateCondition {
    /// Unconditional access
    Unconditional = 0,
    /// Conditional on I/O read trap
    IoRead = 1,
    /// Conditional on I/O write trap
    IoWrite = 2,
}

// ============================================================================
// Access Types
// ============================================================================

/// Type of access being requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    /// Read access
    Read,
    /// Write access
    Write,
    /// Execute access (for instructions)
    Execute,
}

impl AccessType {
    /// Convert to resource attribute mask.
    pub fn as_attr_mask(self) -> u32 {
        match self {
            AccessType::Read => RESOURCE_ATTR_READ,
            AccessType::Write => RESOURCE_ATTR_WRITE,
            AccessType::Execute => RESOURCE_ATTR_EXECUTE,
        }
    }
}

// ============================================================================
// I/O Width
// ============================================================================

/// I/O access width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IoWidth {
    /// 8-bit (1 byte) access
    Byte = 1,
    /// 16-bit (2 byte) access
    Word = 2,
    /// 32-bit (4 byte) access
    Dword = 4,
}

impl IoWidth {
    /// Get the size in bytes.
    pub fn size(self) -> u32 {
        self as u32
    }

    /// Create from size in bytes.
    pub fn from_size(size: u32) -> Option<Self> {
        match size {
            1 => Some(Self::Byte),
            2 => Some(Self::Word),
            4 => Some(Self::Dword),
            _ => None,
        }
    }
}

// ============================================================================
// Policy Data Structures (V1.0)
// ============================================================================

/// Memory policy descriptor (V1.0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemDescriptorV1_0 {
    /// Base address of memory region.
    pub base_address: u64,
    /// Size of memory region in bytes.
    pub size: u64,
    /// Memory attributes (combination of `RESOURCE_ATTR_*`).
    pub mem_attributes: u32,
    /// Reserved, must be 0.
    pub reserved: u32,
}

/// I/O policy descriptor (V1.0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IoDescriptorV1_0 {
    /// Base I/O port address.
    pub io_address: u16,
    /// Length or width of the I/O range.
    pub length_or_width: u16,
    /// I/O attributes (combination of `RESOURCE_ATTR_*`).
    pub attributes: u16,
    /// Reserved, must be 0.
    pub reserved: u16,
}

/// MSR policy descriptor (V1.0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MsrDescriptorV1_0 {
    /// Base MSR address.
    pub msr_address: u32,
    /// Length of MSR range.
    pub length: u16,
    /// MSR attributes (combination of `RESOURCE_ATTR_*`).
    pub attributes: u16,
}

/// Instruction policy descriptor (V1.0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InstructionDescriptorV1_0 {
    /// Instruction index (one of `INSTRUCTION_*` constants).
    pub instruction_index: u16,
    /// Instruction attributes (combination of `RESOURCE_ATTR_*`).
    pub attributes: u16,
    /// Reserved, must be 0.
    pub reserved: u32,
}

/// Save state policy descriptor (V1.0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SaveStateDescriptorV1_0 {
    /// Save state map field (one of `SVST_*` constants).
    pub map_field: u32,
    /// Save state attributes (combination of `RESOURCE_ATTR_*`).
    pub attributes: u32,
    /// Access condition (one of `SVST_CONDITION_*` constants).
    pub access_condition: u32,
    /// Reserved, must be 0.
    pub reserved: u32,
}

/// Policy root structure (V1).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PolicyRootV1 {
    /// Version of this policy root structure.
    pub version: u32,
    /// Size of this policy root structure in bytes.
    pub policy_root_size: u32,
    /// Type of descriptors (one of `TYPE_*` constants).
    pub policy_type: u32,
    /// Offset in bytes from policy data start to the descriptors.
    pub offset: u32,
    /// Number of descriptor entries.
    pub count: u32,
    /// Access attribute (one of `ACCESS_ATTR_*` constants).
    pub access_attr: u8,
    /// Reserved, must be all zeros.
    pub reserved: [u8; 3],
}

/// Secure policy data header (V1.0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SecurePolicyDataV1_0 {
    /// Minor version (should be 0x0000).
    pub version_minor: u16,
    /// Major version (should be 0x0001).
    pub version_major: u16,
    /// Total size in bytes of the entire policy block.
    pub size: u32,
    /// Offset to legacy memory policy (0 if not supported).
    pub memory_policy_offset: u32,
    /// Count of legacy memory policy entries (0 if not supported).
    pub memory_policy_count: u32,
    /// Flag field indicating supervisor status.
    pub flags: u32,
    /// Capability field indicating features supported by supervisor.
    pub capabilities: u32,
    /// Reserved, must be 0.
    pub reserved: u64,
    /// Offset from this structure to the policy root array.
    pub policy_root_offset: u32,
    /// Number of policy roots.
    pub policy_root_count: u32,
}

impl SecurePolicyDataV1_0 {
    /// Returns true if this is a valid V1.0 policy header.
    pub fn is_valid_version(&self) -> bool {
        self.version_major == 1 && self.version_minor == 0
    }

    /// Gets a pointer to the policy root array.
    ///
    /// # Safety
    ///
    /// The caller must ensure that this structure is part of a valid policy buffer.
    pub unsafe fn get_policy_roots_ptr(&self) -> *const PolicyRootV1 {
        let base = self as *const Self as *const u8;
        unsafe { base.add(self.policy_root_offset as usize) as *const PolicyRootV1 }
    }

    /// Gets a slice of policy roots.
    ///
    /// # Safety
    ///
    /// The caller must ensure that this structure is part of a valid policy buffer.
    pub unsafe fn get_policy_roots(&self) -> &[PolicyRootV1] {
        unsafe { slice::from_raw_parts(self.get_policy_roots_ptr(), self.policy_root_count as usize) }
    }
}

impl PolicyRootV1 {
    /// Returns true if the reserved fields are all zeros.
    pub fn has_valid_reserved(&self) -> bool {
        self.reserved == [0, 0, 0]
    }

    /// Gets a pointer to the descriptors for this policy root.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `policy_base` points to a valid policy buffer.
    pub unsafe fn get_descriptors_ptr<T>(&self, policy_base: *const u8) -> *const T {
        unsafe { policy_base.add(self.offset as usize) as *const T }
    }

    /// Gets memory descriptors from this policy root.
    ///
    /// # Safety
    ///
    /// Caller must ensure this policy root has `policy_type == TYPE_MEM`.
    pub unsafe fn get_mem_descriptors(&self, policy_base: *const u8) -> &[MemDescriptorV1_0] {
        unsafe {
            slice::from_raw_parts(
                self.get_descriptors_ptr::<MemDescriptorV1_0>(policy_base),
                self.count as usize,
            )
        }
    }

    /// Gets I/O descriptors from this policy root.
    ///
    /// # Safety
    ///
    /// Caller must ensure this policy root has `policy_type == TYPE_IO`.
    pub unsafe fn get_io_descriptors(&self, policy_base: *const u8) -> &[IoDescriptorV1_0] {
        unsafe {
            slice::from_raw_parts(
                self.get_descriptors_ptr::<IoDescriptorV1_0>(policy_base),
                self.count as usize,
            )
        }
    }

    /// Gets MSR descriptors from this policy root.
    ///
    /// # Safety
    ///
    /// Caller must ensure this policy root has `policy_type == TYPE_MSR`.
    pub unsafe fn get_msr_descriptors(&self, policy_base: *const u8) -> &[MsrDescriptorV1_0] {
        unsafe {
            slice::from_raw_parts(
                self.get_descriptors_ptr::<MsrDescriptorV1_0>(policy_base),
                self.count as usize,
            )
        }
    }

    /// Gets instruction descriptors from this policy root.
    ///
    /// # Safety
    ///
    /// Caller must ensure this policy root has `policy_type == TYPE_INSTRUCTION`.
    pub unsafe fn get_instruction_descriptors(&self, policy_base: *const u8) -> &[InstructionDescriptorV1_0] {
        unsafe {
            slice::from_raw_parts(
                self.get_descriptors_ptr::<InstructionDescriptorV1_0>(policy_base),
                self.count as usize,
            )
        }
    }

    /// Gets save state descriptors from this policy root.
    ///
    /// # Safety
    ///
    /// Caller must ensure this policy root has `policy_type == TYPE_SAVE_STATE`.
    pub unsafe fn get_save_state_descriptors(&self, policy_base: *const u8) -> &[SaveStateDescriptorV1_0] {
        unsafe {
            slice::from_raw_parts(
                self.get_descriptors_ptr::<SaveStateDescriptorV1_0>(policy_base),
                self.count as usize,
            )
        }
    }
}

// ============================================================================
// Size assertions
// ============================================================================

const _: () = {
    assert!(core::mem::size_of::<MemDescriptorV1_0>() == 24);
    assert!(core::mem::size_of::<IoDescriptorV1_0>() == 8);
    assert!(core::mem::size_of::<MsrDescriptorV1_0>() == 8);
    assert!(core::mem::size_of::<InstructionDescriptorV1_0>() == 8);
    assert!(core::mem::size_of::<SaveStateDescriptorV1_0>() == 16);
    assert!(core::mem::size_of::<PolicyRootV1>() == 24);
    assert!(core::mem::size_of::<SecurePolicyDataV1_0>() == 40);
};
