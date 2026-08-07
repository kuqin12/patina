//! Page Table Ownership Queries for the MM Supervisor Core
//!
//! Provides helpers to determine whether a memory range is owned by the supervisor
//! (CPL0) or the user module (CPL3), based on the page table's `Supervisor` attribute
//! (the X64 U/S bit).
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use patina::{UEFI_PAGE_SIZE, align_range};
use patina_paging::{MemoryAttributes, PageTable};

use crate::state::security_state;

/// Result of a page table ownership query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PageOwnership {
    /// The page is user-accessible (U/S = 1, SpecialPurpose clear).
    User,
    /// The page is supervisor-only (U/S = 0, SpecialPurpose set).
    Supervisor,
}

/// Queries the page table to determine the ownership (user vs supervisor) of an address.
///
/// The address and size are page-aligned before querying (rounded down / up respectively).
///
/// Checks the `Supervisor` attribute which maps to the U/S bit on X64:
///   - `Supervisor` set  => `PageOwnership::Supervisor` (U/S = 0)
///   - `Supervisor` clear => `PageOwnership::User` (U/S = 1)
///
/// Returns `None` if the page table is not initialized or the address is unmapped.
pub(crate) fn query_address_ownership(address: u64, size: u64) -> Option<PageOwnership> {
    let (aligned_addr, aligned_size) = align_range(address, size, UEFI_PAGE_SIZE as u64).ok()?;
    let page_table = security_state().lock_page_table();
    let pt = page_table.as_ref()?;
    let attrs = pt.query_memory_region(aligned_addr, aligned_size).ok()?;
    log::trace!(
        "Queried page ownership for address range 0x{:016x}-0x{:016x}: attributes={:?}",
        aligned_addr,
        aligned_addr + aligned_size,
        attrs
    );
    if attrs.contains(MemoryAttributes::Supervisor) {
        Some(PageOwnership::Supervisor)
    } else {
        Some(PageOwnership::User)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_ownership_is_copy_and_comparable() {
        let owner = PageOwnership::Supervisor;
        let copied = owner; // relies on `Copy`
        assert_eq!(owner, copied);
        assert_ne!(PageOwnership::Supervisor, PageOwnership::User);
    }

    #[test]
    fn test_query_address_ownership_none_when_page_table_uninitialized() {
        // With no page table installed, ownership cannot be determined.
        *security_state().lock_page_table() = None;
        assert_eq!(query_address_ownership(0x1000, 0x1000), None);
    }
}
