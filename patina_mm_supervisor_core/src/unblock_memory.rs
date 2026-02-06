//! Unblocked Memory Region Management
//!
//! This module provides functionality to track and manage memory regions that have been
//! unblocked for access in the MM (Management Mode) environment, similar to `UnblockMemory.c`.
//!
//! ## Overview
//!
//! The MM Supervisor maintains a list of memory regions that have been explicitly unblocked
//! for access. By default, all memory outside MMRAM is blocked. Drivers and handlers can
//! request specific regions to be unblocked via the `unblock_memory` interface.
//!
//! ## Design
//!
//! - The unblocked region tracker is initialized from memory policy descriptors
//! - Regions can be dynamically added via `unblock_memory()`
//! - Access checks use `is_memory_blocked()` to validate memory access requests
//! - Duplicate unblock requests with identical attributes are allowed (idempotent)
//! - Overlapping requests with different attributes are rejected
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

use patina_mm_policy::{MemDescriptorV1_0, RESOURCE_ATTR_READ, RESOURCE_ATTR_WRITE, RESOURCE_ATTR_EXECUTE};

use crate::mm_mem::PAGE_ALLOCATOR;

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of unblocked memory regions that can be tracked.
/// This is a conservative limit; in practice, most systems will have far fewer.
const MAX_UNBLOCKED_REGIONS: usize = 256;

/// EFI_MEMORY_SP attribute bit - Supervisor page (kernel-only access).
pub const EFI_MEMORY_SP: u64 = 0x0000000000040000;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during unblock memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnblockError {
    /// The region tracker has not been initialized.
    NotInitialized,
    /// Already initialized (cannot re-initialize).
    AlreadyInitialized,
    /// Too many regions to track (exceeded MAX_UNBLOCKED_REGIONS).
    TooManyRegions,
    /// The requested region overlaps with MMRAM.
    OverlapsWithMmram,
    /// The requested region overlaps with an existing unblocked region
    /// but has different attributes.
    ConflictingAttributes,
    /// The requested region is already unblocked (identical request).
    AlreadyUnblocked,
    /// Invalid parameters (null pointer, zero length, etc.).
    InvalidParameter,
    /// The region's address + size would overflow.
    AddressOverflow,
}

// ============================================================================
// Unblocked Memory Entry
// ============================================================================

/// A single entry in the unblocked memory region list.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnblockedMemoryEntry {
    /// Base address of the unblocked region.
    pub base_address: u64,
    /// Size of the unblocked region in bytes.
    pub size: u64,
    /// Memory attributes (combination of `RESOURCE_ATTR_*`).
    pub attributes: u32,
    /// Whether this entry is valid (in use).
    pub valid: bool,
}

impl UnblockedMemoryEntry {
    /// Creates a new empty (invalid) entry.
    pub const fn empty() -> Self {
        Self {
            base_address: 0,
            size: 0,
            attributes: 0,
            valid: false,
        }
    }

    /// Creates a new valid entry from base, size, and attributes.
    pub const fn new(base_address: u64, size: u64, attributes: u32) -> Self {
        Self {
            base_address,
            size,
            attributes,
            valid: true,
        }
    }

    /// Returns the end address (exclusive) of this region.
    pub fn end_address(&self) -> u64 {
        self.base_address.saturating_add(self.size)
    }

    /// Checks if the given range [base, base + size) is fully contained within this entry.
    pub fn contains(&self, base: u64, size: u64) -> bool {
        if !self.valid || size == 0 {
            return false;
        }
        let query_end = base.saturating_add(size);
        base >= self.base_address && query_end <= self.end_address()
    }

    /// Checks if the given range [base, base + size) overlaps with this entry.
    pub fn overlaps(&self, base: u64, size: u64) -> bool {
        if !self.valid || size == 0 {
            return false;
        }
        let query_end = base.saturating_add(size);
        let entry_end = self.end_address();

        // Two ranges overlap if: start1 < end2 && start2 < end1
        base < entry_end && self.base_address < query_end
    }

    /// Checks if this entry has identical base, size, and attributes as the query.
    pub fn is_identical(&self, base: u64, size: u64, attributes: u32) -> bool {
        self.valid
            && self.base_address == base
            && self.size == size
            && self.attributes == attributes
    }
}

// ============================================================================
// Unblocked Memory Tracker
// ============================================================================

/// Internal state for the unblocked memory tracker.
struct UnblockedMemoryState {
    /// Array of unblocked memory entries.
    entries: [UnblockedMemoryEntry; MAX_UNBLOCKED_REGIONS],
    /// Number of valid entries in the array.
    count: usize,
}

impl UnblockedMemoryState {
    /// Creates a new empty state.
    const fn new() -> Self {
        Self {
            entries: [UnblockedMemoryEntry::empty(); MAX_UNBLOCKED_REGIONS],
            count: 0,
        }
    }

    /// Finds an entry that exactly matches the given base and size.
    fn find_exact_match(&self, base: u64, size: u64) -> Option<&UnblockedMemoryEntry> {
        self.entries[..self.count].iter().find(|e| {
            e.valid && e.base_address == base && e.size == size
        })
    }

    /// Finds all entries that overlap with the given range.
    #[allow(dead_code)]
    fn find_overlapping(&self, base: u64, size: u64) -> impl Iterator<Item = &UnblockedMemoryEntry> {
        self.entries[..self.count]
            .iter()
            .filter(move |e| e.overlaps(base, size))
    }

    /// Adds a new entry if there's space.
    fn add_entry(&mut self, base: u64, size: u64, attributes: u32) -> Result<(), UnblockError> {
        if self.count >= MAX_UNBLOCKED_REGIONS {
            return Err(UnblockError::TooManyRegions);
        }

        self.entries[self.count] = UnblockedMemoryEntry::new(base, size, attributes);
        self.count += 1;
        Ok(())
    }
}

/// Global unblocked memory region tracker.
///
/// This struct manages a list of memory regions that have been unblocked for
/// access within the MM environment. It provides thread-safe access to the
/// region list through internal locking.
pub struct UnblockedMemoryTracker {
    /// Whether the tracker has been initialized.
    initialized: AtomicBool,
    /// Flag indicating if core initialization is complete (after which we enforce checks).
    core_init_complete: AtomicBool,
    /// Internal state protected by a mutex.
    state: Mutex<UnblockedMemoryState>,
}

impl UnblockedMemoryTracker {
    /// Creates a new unblocked memory tracker.
    pub const fn new() -> Self {
        Self {
            initialized: AtomicBool::new(false),
            core_init_complete: AtomicBool::new(false),
            state: Mutex::new(UnblockedMemoryState::new()),
        }
    }

    /// Initializes the tracker from an array of memory policy descriptors.
    ///
    /// This should be called once during BSP initialization after the memory
    /// policy has been generated from the page table walk.
    ///
    /// # Arguments
    ///
    /// * `descriptors` - Slice of memory policy descriptors from page table walk.
    ///                   These represent the initial "unblocked" regions.
    ///
    /// # Returns
    ///
    /// `Ok(())` if initialization succeeded, or an error if:
    /// - Already initialized
    /// - Too many descriptors to track
    pub fn init_from_descriptors(&self, descriptors: &[MemDescriptorV1_0]) -> Result<(), UnblockError> {
        // Check if already initialized
        if self.initialized.swap(true, Ordering::SeqCst) {
            return Err(UnblockError::AlreadyInitialized);
        }

        let mut state = self.state.lock();

        // Add each descriptor as an unblocked region
        for desc in descriptors {
            if desc.size == 0 {
                continue; // Skip zero-size entries
            }

            // Skip regions inside MMRAM - those are supervisor-controlled, not "unblocked"
            if PAGE_ALLOCATOR.is_region_inside_mmram(desc.base_address, desc.size) {
                log::trace!(
                    "Skipping MMRAM region during unblock init: 0x{:016x} - 0x{:016x}",
                    desc.base_address,
                    desc.base_address.saturating_add(desc.size)
                );
                continue;
            }

            state.add_entry(desc.base_address, desc.size, desc.mem_attributes)?;
        }

        log::info!(
            "UnblockedMemoryTracker initialized with {} regions",
            state.count
        );

        Ok(())
    }

    /// Initializes the tracker from a raw buffer of memory policy descriptors.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `buffer` points to a valid array of `MemDescriptorV1_0` structures
    /// - `count` is the number of valid entries in the buffer
    pub unsafe fn init_from_buffer(
        &self,
        buffer: *const MemDescriptorV1_0,
        count: usize,
    ) -> Result<(), UnblockError> {
        if buffer.is_null() || count == 0 {
            // Empty initialization is valid
            if self.initialized.swap(true, Ordering::SeqCst) {
                return Err(UnblockError::AlreadyInitialized);
            }
            log::info!("UnblockedMemoryTracker initialized with 0 regions (empty)");
            return Ok(());
        }

        // SAFETY: Caller guarantees buffer is valid for count entries
        let descriptors = unsafe { core::slice::from_raw_parts(buffer, count) };
        self.init_from_descriptors(descriptors)
    }

    /// Marks core initialization as complete.
    ///
    /// After this is called, memory access checks will be enforced.
    /// Before this, all memory is considered accessible (for bootstrap).
    pub fn set_core_init_complete(&self) {
        self.core_init_complete.store(true, Ordering::Release);
        log::info!("UnblockedMemoryTracker: Core initialization complete, enforcing checks");
    }

    /// Checks if core initialization is complete.
    pub fn is_core_init_complete(&self) -> bool {
        self.core_init_complete.load(Ordering::Acquire)
    }

    /// Unblocks a memory region for access.
    ///
    /// This adds a new region to the unblocked list after validating:
    /// - The region does not overlap with MMRAM
    /// - The region is not already unblocked with different attributes
    /// - Identical unblock requests are allowed (idempotent)
    ///
    /// # Arguments
    ///
    /// * `base` - Base address of the region to unblock
    /// * `size` - Size of the region in bytes
    /// * `attributes` - Memory attributes (RESOURCE_ATTR_READ | WRITE | EXECUTE)
    ///
    /// # Returns
    ///
    /// `Ok(())` if the region was successfully unblocked or already unblocked with same attributes.
    /// `Err(UnblockError)` if the request is invalid or conflicts with existing regions.
    pub fn unblock_memory(
        &self,
        base: u64,
        size: u64,
        attributes: u32,
    ) -> Result<(), UnblockError> {
        // Validate parameters
        if size == 0 {
            return Err(UnblockError::InvalidParameter);
        }

        // Check for address overflow
        if base.checked_add(size).is_none() {
            return Err(UnblockError::AddressOverflow);
        }

        // Check if the region overlaps with MMRAM
        if PAGE_ALLOCATOR.is_region_inside_mmram(base, size) {
            log::error!(
                "unblock_memory: Region 0x{:016x} - 0x{:016x} overlaps with MMRAM",
                base,
                base.saturating_add(size)
            );
            return Err(UnblockError::OverlapsWithMmram);
        }

        let mut state = self.state.lock();

        // Check for existing entries that might conflict
        // First, check for exact match (idempotent unblock)
        if let Some(existing) = state.find_exact_match(base, size) {
            if existing.attributes == attributes {
                // Identical request - this is allowed (idempotent)
                log::debug!(
                    "unblock_memory: Region 0x{:016x} - 0x{:016x} already unblocked with same attributes",
                    base,
                    base.saturating_add(size)
                );
                return Ok(());
            } else {
                // Same base/size but different attributes - conflict
                log::error!(
                    "unblock_memory: Region 0x{:016x} - 0x{:016x} already unblocked with different attributes (existing: 0x{:x}, requested: 0x{:x})",
                    base,
                    base.saturating_add(size),
                    existing.attributes,
                    attributes
                );
                return Err(UnblockError::ConflictingAttributes);
            }
        }

        // Check for partial overlaps (not allowed)
        // We iterate directly without collecting to avoid heap allocation
        let mut has_overlap = false;
        for entry in &state.entries[..state.count] {
            if entry.overlaps(base, size) {
                log::error!(
                    "unblock_memory: Region 0x{:016x} - 0x{:016x} overlaps with existing region 0x{:016x} - 0x{:016x}",
                    base,
                    base.saturating_add(size),
                    entry.base_address,
                    entry.end_address()
                );
                has_overlap = true;
                // Continue to log all overlaps for debugging
            }
        }

        if has_overlap {
            return Err(UnblockError::ConflictingAttributes);
        }

        // No conflicts - add the new entry
        state.add_entry(base, size, attributes)?;

        log::info!(
            "unblock_memory: Unblocked region 0x{:016x} - 0x{:016x} with attributes 0x{:x}",
            base,
            base.saturating_add(size),
            attributes
        );

        Ok(())
    }

    /// Checks if a memory region is blocked (i.e., NOT in the unblocked list).
    ///
    /// This is the inverse of checking if memory is accessible - a blocked region
    /// should not be accessed by MM handlers.
    ///
    /// # Arguments
    ///
    /// * `base` - Base address of the region to check
    /// * `size` - Size of the region in bytes
    ///
    /// # Returns
    ///
    /// `true` if the region is blocked (not accessible), `false` if unblocked.
    ///
    /// # Note
    ///
    /// Before core initialization is complete, this always returns `false`
    /// (everything is accessible during bootstrap).
    pub fn is_memory_blocked(&self, base: u64, size: u64) -> bool {
        // During initialization, everything is accessible
        if !self.core_init_complete.load(Ordering::Acquire) {
            return false;
        }

        // Zero-size queries are invalid
        if size == 0 {
            log::warn!("is_memory_blocked: Zero-size query for address 0x{:016x}", base);
            return true; // Invalid query = blocked
        }

        // Check for address overflow
        if base.checked_add(size).is_none() {
            log::warn!("is_memory_blocked: Address overflow for 0x{:016x} + 0x{:x}", base, size);
            return true; // Invalid query = blocked
        }

        let state = self.state.lock();

        // Check if the queried region is fully contained within any unblocked entry
        for entry in &state.entries[..state.count] {
            if entry.contains(base, size) {
                log::trace!(
                    "is_memory_blocked: Region 0x{:016x} - 0x{:016x} is within unblocked region 0x{:016x} - 0x{:016x}",
                    base,
                    base.saturating_add(size),
                    entry.base_address,
                    entry.end_address()
                );
                return false; // Found within unblocked region
            }
        }

        log::trace!(
            "is_memory_blocked: Region 0x{:016x} - 0x{:016x} is NOT within any unblocked region",
            base,
            base.saturating_add(size)
        );

        true // Not found in any unblocked region = blocked
    }

    /// Checks if a memory region is within unblocked regions (the inverse of `is_memory_blocked`).
    ///
    /// This is a convenience method that returns `true` if the region is accessible.
    #[inline]
    pub fn is_within_unblocked_region(&self, base: u64, size: u64) -> bool {
        !self.is_memory_blocked(base, size)
    }

    /// Gets the current count of unblocked regions.
    pub fn region_count(&self) -> usize {
        self.state.lock().count
    }

    /// Collects unblocked regions into a provided buffer.
    ///
    /// This is useful for reporting or serializing the unblocked region list.
    ///
    /// # Arguments
    ///
    /// * `start_index` - Starting index in the region list
    /// * `buffer` - Buffer to fill with region descriptors
    ///
    /// # Returns
    ///
    /// The number of entries actually copied to the buffer.
    pub fn collect_regions(
        &self,
        start_index: usize,
        buffer: &mut [MemDescriptorV1_0],
    ) -> usize {
        let state = self.state.lock();

        if start_index >= state.count || buffer.is_empty() {
            return 0;
        }

        let mut copied = 0;
        for (i, entry) in state.entries[start_index..state.count].iter().enumerate() {
            if i >= buffer.len() {
                break;
            }
            if entry.valid {
                buffer[i] = MemDescriptorV1_0 {
                    base_address: entry.base_address,
                    size: entry.size,
                    mem_attributes: entry.attributes,
                    reserved: 0,
                };
                copied += 1;
            }
        }

        copied
    }

    /// Dumps the unblocked regions for debugging.
    pub fn dump_regions(&self) {
        let state = self.state.lock();

        log::info!("UnblockedMemoryTracker: {} regions", state.count);
        for (i, entry) in state.entries[..state.count].iter().enumerate() {
            if entry.valid {
                let r = if (entry.attributes & RESOURCE_ATTR_READ) != 0 { "R" } else { "." };
                let w = if (entry.attributes & RESOURCE_ATTR_WRITE) != 0 { "W" } else { "." };
                let x = if (entry.attributes & RESOURCE_ATTR_EXECUTE) != 0 { "X" } else { "." };
                log::info!(
                    "  [{}] 0x{:016x} - 0x{:016x} {}{}{}",
                    i,
                    entry.base_address,
                    entry.end_address(),
                    r, w, x
                );
            }
        }
    }
}

// ============================================================================
// Global Instance
// ============================================================================

/// Global unblocked memory tracker instance.
///
/// This is the singleton that manages all unblocked memory regions for the
/// MM Supervisor. It should be initialized during BSP initialization and
/// used for all memory access validation.
pub static UNBLOCKED_MEMORY_TRACKER: UnblockedMemoryTracker = UnblockedMemoryTracker::new();

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tracker() -> UnblockedMemoryTracker {
        UnblockedMemoryTracker::new()
    }

    #[test]
    fn test_empty_entry() {
        let entry = UnblockedMemoryEntry::empty();
        assert!(!entry.valid);
        assert_eq!(entry.base_address, 0);
        assert_eq!(entry.size, 0);
    }

    #[test]
    fn test_entry_contains() {
        let entry = UnblockedMemoryEntry::new(0x1000, 0x1000, RESOURCE_ATTR_READ);
        
        // Fully contained
        assert!(entry.contains(0x1000, 0x1000));
        assert!(entry.contains(0x1000, 0x800));
        assert!(entry.contains(0x1800, 0x800));
        
        // Partially outside
        assert!(!entry.contains(0x0800, 0x1000)); // Starts before
        assert!(!entry.contains(0x1800, 0x1000)); // Ends after
        
        // Completely outside
        assert!(!entry.contains(0x3000, 0x1000));
    }

    #[test]
    fn test_entry_overlaps() {
        let entry = UnblockedMemoryEntry::new(0x1000, 0x1000, RESOURCE_ATTR_READ);
        
        // Overlapping cases
        assert!(entry.overlaps(0x1000, 0x1000)); // Exact match
        assert!(entry.overlaps(0x0800, 0x1000)); // Starts before, ends inside
        assert!(entry.overlaps(0x1800, 0x1000)); // Starts inside, ends after
        assert!(entry.overlaps(0x0800, 0x2000)); // Completely contains entry
        
        // Non-overlapping
        assert!(!entry.overlaps(0x2000, 0x1000)); // Immediately after
        assert!(!entry.overlaps(0x0000, 0x1000)); // Immediately before
        assert!(!entry.overlaps(0x3000, 0x1000)); // Far after
    }

    #[test]
    fn test_tracker_before_init_complete() {
        let tracker = create_test_tracker();
        
        // Before core init complete, nothing is blocked
        assert!(!tracker.is_memory_blocked(0x1000, 0x1000));
        assert!(!tracker.is_memory_blocked(0x0, 0x100000));
    }

    #[test]
    fn test_tracker_after_init_complete_empty() {
        let tracker = create_test_tracker();
        tracker.set_core_init_complete();
        
        // After init complete with no regions, everything is blocked
        assert!(tracker.is_memory_blocked(0x1000, 0x1000));
    }

    #[test]
    fn test_unblock_memory() {
        let tracker = create_test_tracker();
        
        // Unblock a region
        assert!(tracker.unblock_memory(0x1000, 0x1000, RESOURCE_ATTR_READ | RESOURCE_ATTR_WRITE).is_ok());
        
        tracker.set_core_init_complete();
        
        // Region should be accessible
        assert!(!tracker.is_memory_blocked(0x1000, 0x1000));
        assert!(!tracker.is_memory_blocked(0x1000, 0x800));
        
        // Outside region should be blocked
        assert!(tracker.is_memory_blocked(0x3000, 0x1000));
    }

    #[test]
    fn test_idempotent_unblock() {
        let tracker = create_test_tracker();
        
        // First unblock
        assert!(tracker.unblock_memory(0x1000, 0x1000, RESOURCE_ATTR_READ).is_ok());
        
        // Identical unblock should succeed
        assert!(tracker.unblock_memory(0x1000, 0x1000, RESOURCE_ATTR_READ).is_ok());
        
        // Same region with different attributes should fail
        assert_eq!(
            tracker.unblock_memory(0x1000, 0x1000, RESOURCE_ATTR_READ | RESOURCE_ATTR_WRITE),
            Err(UnblockError::ConflictingAttributes)
        );
    }

    #[test]
    fn test_overlapping_unblock_fails() {
        let tracker = create_test_tracker();
        
        // First unblock
        assert!(tracker.unblock_memory(0x1000, 0x1000, RESOURCE_ATTR_READ).is_ok());
        
        // Overlapping unblock should fail
        assert_eq!(
            tracker.unblock_memory(0x1800, 0x1000, RESOURCE_ATTR_READ),
            Err(UnblockError::ConflictingAttributes)
        );
    }

    #[test]
    fn test_invalid_parameters() {
        let tracker = create_test_tracker();
        tracker.set_core_init_complete();
        
        // Zero size
        assert_eq!(
            tracker.unblock_memory(0x1000, 0, RESOURCE_ATTR_READ),
            Err(UnblockError::InvalidParameter)
        );
        
        // Overflow
        assert_eq!(
            tracker.unblock_memory(u64::MAX, 0x1000, RESOURCE_ATTR_READ),
            Err(UnblockError::AddressOverflow)
        );
    }

    #[test]
    fn test_region_count() {
        let tracker = create_test_tracker();
        
        assert_eq!(tracker.region_count(), 0);
        
        tracker.unblock_memory(0x1000, 0x1000, RESOURCE_ATTR_READ).unwrap();
        assert_eq!(tracker.region_count(), 1);
        
        tracker.unblock_memory(0x3000, 0x1000, RESOURCE_ATTR_WRITE).unwrap();
        assert_eq!(tracker.region_count(), 2);
    }
}
