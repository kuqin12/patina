//! Policy Gate - Runtime access validation
//!
//! This module provides the `PolicyGate` struct that wraps a policy buffer
//! and provides methods to check if various operations are allowed.

use crate::types::*;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during policy gate operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyError {
    /// The policy pointer is null.
    NullPointer,
    /// Invalid policy version.
    InvalidVersion,
    /// Invalid access mask specified.
    InvalidAccessMask,
    /// Invalid I/O width specified.
    InvalidIoWidth,
    /// Invalid I/O address (out of 16-bit range).
    InvalidIoAddress,
    /// Invalid I/O address range (overflow).
    InvalidIoRange,
    /// Invalid instruction index.
    InvalidInstructionIndex,
    /// Invalid save state map field.
    InvalidSaveStateField,
    /// Policy root not found for the requested type.
    PolicyRootNotFound,
    /// Access denied by policy.
    AccessDenied,
    /// Internal error during policy evaluation.
    InternalError,
}

// ============================================================================
// Policy Gate
// ============================================================================

/// Policy gate for runtime access validation.
///
/// This struct wraps a policy buffer and provides methods to check if
/// various operations (I/O, MSR, instruction, save state) are allowed.
///
/// ## Example
///
/// ```rust,ignore
/// use patina_mm_policy::{PolicyGate, AccessType, IoWidth};
///
/// let gate = unsafe { PolicyGate::new(policy_ptr) }?;
///
/// // Check I/O access
/// if gate.is_io_allowed(0x3F8, IoWidth::Byte, AccessType::Read).is_ok() {
///     // Access allowed
/// }
/// ```
pub struct PolicyGate {
    /// Pointer to the policy data.
    policy_ptr: *const u8,
}

// SAFETY: PolicyGate only holds a pointer to read-only policy data.
unsafe impl Send for PolicyGate {}
unsafe impl Sync for PolicyGate {}

impl PolicyGate {
    /// Creates a new policy gate from a policy buffer pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `policy_ptr` points to a valid policy buffer
    /// that remains valid for the lifetime of this PolicyGate.
    ///
    /// # Returns
    ///
    /// Returns `Ok(PolicyGate)` if the policy is valid, or an error otherwise.
    pub unsafe fn new(policy_ptr: *const u8) -> Result<Self, PolicyError> {
        if policy_ptr.is_null() {
            return Err(PolicyError::NullPointer);
        }

        let policy = unsafe { &*(policy_ptr as *const SecurePolicyDataV1_0) };
        if !policy.is_valid_version() {
            return Err(PolicyError::InvalidVersion);
        }

        Ok(Self { policy_ptr })
    }

    /// Gets a reference to the policy header.
    fn policy(&self) -> &SecurePolicyDataV1_0 {
        // SAFETY: Constructor validated the pointer
        unsafe { &*(self.policy_ptr as *const SecurePolicyDataV1_0) }
    }

    /// Finds a policy root by type.
    fn find_policy_root(&self, policy_type: u32) -> Option<&PolicyRootV1> {
        let policy = self.policy();
        // SAFETY: Constructor validated the policy
        let roots = unsafe { policy.get_policy_roots() };
        roots.iter().find(|r| r.policy_type == policy_type)
    }

    /// Checks if I/O access is allowed.
    ///
    /// # Arguments
    ///
    /// * `io_address` - The I/O port address (must be <= 0xFFFF)
    /// * `width` - The access width
    /// * `access_type` - Read or Write access
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if access is allowed, or `Err(PolicyError)` otherwise.
    pub fn is_io_allowed(
        &self,
        io_address: u32,
        width: IoWidth,
        access_type: AccessType,
    ) -> Result<(), PolicyError> {
        // Validate access type (must be read or write, not execute)
        if access_type == AccessType::Execute {
            return Err(PolicyError::InvalidAccessMask);
        }

        let io_size = width.size();

        // Validate I/O address range (16-bit port space)
        if io_address > u16::MAX as u32 {
            return Err(PolicyError::InvalidIoAddress);
        }

        // Check for overflow (MAX_UINT16 + 1 is valid for end address)
        if io_address.saturating_add(io_size) > (u16::MAX as u32) + 1 {
            return Err(PolicyError::InvalidIoRange);
        }

        let policy_root = match self.find_policy_root(TYPE_IO) {
            Some(root) => root,
            None => {
                log::warn!("Could not find IO policy root, denying access to be safe.");
                return Err(PolicyError::PolicyRootNotFound);
            }
        };

        // SAFETY: We validated the policy in the constructor
        let descriptors = unsafe { policy_root.get_io_descriptors(self.policy_ptr) };
        let access_mask = access_type.as_attr_mask();

        let mut found_match = false;

        for desc in descriptors {
            let desc_start = desc.io_address as u32;
            let desc_size = desc.length_or_width as u32;
            let is_strict_width = (desc.attributes as u32 & RESOURCE_ATTR_STRICT_WIDTH) != 0;

            if is_strict_width {
                // Strict width: address and size must match exactly
                if io_address == desc_start && io_size == desc_size {
                    // Check if the access type matches
                    if (desc.attributes as u32 & access_mask) != 0 {
                        found_match = true;
                        break;
                    }
                }
            } else {
                // Non-strict: check if our range is covered by this descriptor
                let desc_end = desc_start.saturating_add(desc_size);
                let our_end = io_address.saturating_add(io_size);

                if io_address >= desc_start && our_end <= desc_end {
                    // Check if the access type matches
                    if (desc.attributes as u32 & access_mask) != 0 {
                        found_match = true;
                        break;
                    }
                }
            }
        }

        // Evaluate based on allow/deny list semantics
        if (found_match && policy_root.access_attr == ACCESS_ATTR_DENY)
            || (!found_match && policy_root.access_attr == ACCESS_ATTR_ALLOW)
        {
            log::debug!(
                "Rejecting IO access: port=0x{:x}, width={}, type={:?}",
                io_address,
                io_size,
                access_type
            );
            return Err(PolicyError::AccessDenied);
        }

        Ok(())
    }

    /// Checks if MSR access is allowed.
    ///
    /// # Arguments
    ///
    /// * `msr_address` - The MSR address
    /// * `access_type` - Read or Write access
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if access is allowed, or `Err(PolicyError)` otherwise.
    pub fn is_msr_allowed(&self, msr_address: u32, access_type: AccessType) -> Result<(), PolicyError> {
        // Validate access type
        if access_type == AccessType::Execute {
            return Err(PolicyError::InvalidAccessMask);
        }

        let policy_root = match self.find_policy_root(TYPE_MSR) {
            Some(root) => root,
            None => {
                log::warn!("Could not find MSR policy root, denying access to be safe.");
                return Err(PolicyError::PolicyRootNotFound);
            }
        };

        // SAFETY: We validated the policy in the constructor
        let descriptors = unsafe { policy_root.get_msr_descriptors(self.policy_ptr) };
        let access_mask = access_type.as_attr_mask();

        let mut found_match = false;

        for desc in descriptors {
            let desc_start = desc.msr_address;
            let desc_end = desc_start.saturating_add(desc.length as u32);

            if msr_address >= desc_start && msr_address < desc_end {
                if (desc.attributes as u32 & access_mask) != 0 {
                    found_match = true;
                    break;
                }
            }
        }

        // Evaluate based on allow/deny list semantics
        if (found_match && policy_root.access_attr == ACCESS_ATTR_DENY)
            || (!found_match && policy_root.access_attr == ACCESS_ATTR_ALLOW)
        {
            log::debug!(
                "Rejecting MSR access: address=0x{:x}, type={:?}",
                msr_address,
                access_type
            );
            return Err(PolicyError::AccessDenied);
        }

        Ok(())
    }

    /// Checks if instruction execution is allowed.
    ///
    /// # Arguments
    ///
    /// * `instruction` - The instruction to check
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if execution is allowed, or `Err(PolicyError)` otherwise.
    pub fn is_instruction_allowed(&self, instruction: Instruction) -> Result<(), PolicyError> {
        let instruction_index = instruction.as_index();

        if instruction_index >= INSTRUCTION_COUNT {
            return Err(PolicyError::InvalidInstructionIndex);
        }

        let policy_root = match self.find_policy_root(TYPE_INSTRUCTION) {
            Some(root) => root,
            None => {
                log::warn!("Could not find Instruction policy root, denying access to be safe.");
                return Err(PolicyError::PolicyRootNotFound);
            }
        };

        // SAFETY: We validated the policy in the constructor
        let descriptors = unsafe { policy_root.get_instruction_descriptors(self.policy_ptr) };

        let mut found_match = false;

        for desc in descriptors {
            if instruction_index == desc.instruction_index {
                if (desc.attributes as u32 & RESOURCE_ATTR_EXECUTE) != 0 {
                    found_match = true;
                    break;
                }
            }
        }

        // Evaluate based on allow/deny list semantics
        if (found_match && policy_root.access_attr == ACCESS_ATTR_DENY)
            || (!found_match && policy_root.access_attr == ACCESS_ATTR_ALLOW)
        {
            log::debug!("Rejecting instruction execution: {:?}", instruction);
            return Err(PolicyError::AccessDenied);
        }

        Ok(())
    }

    /// Checks if save state read access is allowed.
    ///
    /// # Arguments
    ///
    /// * `field` - The save state field to read
    /// * `width` - The access width in bytes
    /// * `current_condition` - The current I/O trap condition (if applicable)
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if access is allowed, or `Err(PolicyError)` otherwise.
    pub fn is_save_state_read_allowed(
        &self,
        field: SaveStateField,
        width: usize,
        current_condition: Option<SaveStateCondition>,
    ) -> Result<(), PolicyError> {
        let policy_root = match self.find_policy_root(TYPE_SAVE_STATE) {
            Some(root) => root,
            None => {
                // No save state policy = level 20, allow all reads
                log::debug!("No save state policy root found, allowing read (level 20 policy).");
                return Ok(());
            }
        };

        // SAFETY: We validated the policy in the constructor
        let descriptors = unsafe { policy_root.get_save_state_descriptors(self.policy_ptr) };

        let mut found_match = false;

        for desc in descriptors {
            if desc.map_field == field.as_index() {
                // Check if this is a read-allowed policy
                let is_read = (desc.attributes & RESOURCE_ATTR_READ) != 0;
                let is_cond_read = (desc.attributes & RESOURCE_ATTR_COND_READ) != 0;

                if is_read || is_cond_read {
                    // Check condition if this is conditional read
                    if is_cond_read {
                        if let Some(current) = current_condition {
                            if desc.access_condition == current as u32 {
                                found_match = true;
                                break;
                            }
                        }
                        // Condition doesn't match, continue looking
                    } else {
                        // Unconditional read
                        if desc.access_condition == SVST_UNCONDITIONAL {
                            found_match = true;
                            break;
                        }
                    }
                }
            }
        }

        // Evaluate based on allow/deny list semantics
        if (found_match && policy_root.access_attr == ACCESS_ATTR_DENY)
            || (!found_match && policy_root.access_attr == ACCESS_ATTR_ALLOW)
        {
            log::debug!(
                "Rejecting save state read: field={:?}, width={}",
                field,
                width
            );
            return Err(PolicyError::AccessDenied);
        }

        Ok(())
    }

    /// Gets the raw policy pointer.
    pub fn as_ptr(&self) -> *const u8 {
        self.policy_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_width() {
        assert_eq!(IoWidth::Byte.size(), 1);
        assert_eq!(IoWidth::Word.size(), 2);
        assert_eq!(IoWidth::Dword.size(), 4);
    }

    #[test]
    fn test_access_type_mask() {
        assert_eq!(AccessType::Read.as_attr_mask(), RESOURCE_ATTR_READ);
        assert_eq!(AccessType::Write.as_attr_mask(), RESOURCE_ATTR_WRITE);
        assert_eq!(AccessType::Execute.as_attr_mask(), RESOURCE_ATTR_EXECUTE);
    }

    #[test]
    fn test_instruction_conversion() {
        assert_eq!(Instruction::Cli.as_index(), 0);
        assert_eq!(Instruction::from_index(0), Some(Instruction::Cli));
        assert_eq!(Instruction::from_index(99), None);
    }
}
