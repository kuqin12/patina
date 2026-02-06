//! Paging Page Allocator
//!
//! A dedicated page allocator for the paging subsystem that allocates pages for
//! page table structures (PML4, PDPT, PD, PT entries).
//!
//! ## Design
//!
//! This allocator is separate from the generic PageAllocator for two reasons:
//!
//! 1. **Bootstrap problem**: The paging subsystem needs to allocate pages for page
//!    tables, but the generic PageAllocator wants to call into paging to set page
//!    attributes for newly allocated pages. This creates a circular dependency.
//!
//! 2. **Security**: Page table pages require special attributes (Supervisor, RW,
//!    non-executable) and should be tracked separately from general allocations.
//!
//! ## Initialization
//!
//! The paging allocator is initialized with a reserved memory region from SMRAM.
//! This region is exclusively used for page table allocations.
//!
//! ## Integration with Paging
//!
//! After the paging subsystem is fully initialized, the generic PageAllocator can
//! optionally register a callback to apply page table attributes to newly allocated
//! pages via the paging instance.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use core::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use patina_paging::{PtError, page_allocator::PageAllocator as PagingPageAllocator};
use spin::Mutex;

use crate::mm_mem::PAGE_SIZE;

// ============================================================================
// Constants
// ============================================================================

/// Default number of pages to reserve for page table allocations.
/// This should be sufficient for most MM environments (128 pages = 512KB).
pub const DEFAULT_PAGING_POOL_PAGES: usize = 128;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during paging allocator operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingAllocError {
    /// The allocator has not been initialized.
    NotInitialized,
    /// Already initialized.
    AlreadyInitialized,
    /// No free pages available to satisfy the request.
    OutOfMemory,
    /// Invalid alignment requested.
    InvalidAlignment,
    /// The pool region is too small.
    PoolTooSmall,
}

// ============================================================================
// Paging Page Allocator
// ============================================================================

/// A dedicated page allocator for the paging subsystem.
///
/// This allocator uses a simple bump allocator from a reserved pool of pages.
/// It implements the `patina_paging::PageAllocator` trait to be used directly
/// by the paging crate for allocating page table structures.
///
/// ## Thread Safety
///
/// This allocator is thread-safe and can be used from multiple CPUs.
///
/// ## Example
///
/// ```rust,ignore
/// use patina_mm_supervisor_core::paging_allocator::{PagingPageAllocator, DEFAULT_PAGING_POOL_PAGES};
///
/// // During early init, reserve a region from SMRAM for page tables
/// let pool_base = 0x8000_0000u64; // Example base address
/// let pool_pages = DEFAULT_PAGING_POOL_PAGES;
///
/// // Initialize the allocator
/// unsafe {
///     PAGING_ALLOCATOR.init(pool_base, pool_pages)?;
/// }
///
/// // The allocator can now be used by the paging crate
/// ```
pub struct PagingPoolAllocator {
    /// Base address of the pool.
    pool_base: AtomicU64,
    /// Total number of pages in the pool.
    pool_pages: AtomicUsize,
    /// Current allocation offset (bump pointer) in bytes.
    current_offset: AtomicUsize,
    /// Number of pages allocated.
    allocated_pages: AtomicUsize,
    /// Whether the allocator has been initialized.
    initialized: AtomicBool,
    /// Lock for thread safety during allocation.
    lock: Mutex<()>,
}

// SAFETY: The PagingPoolAllocator uses internal locking for thread safety.
unsafe impl Send for PagingPoolAllocator {}
unsafe impl Sync for PagingPoolAllocator {}

impl PagingPoolAllocator {
    /// Creates a new uninitialized paging page allocator.
    pub const fn new() -> Self {
        Self {
            pool_base: AtomicU64::new(0),
            pool_pages: AtomicUsize::new(0),
            current_offset: AtomicUsize::new(0),
            allocated_pages: AtomicUsize::new(0),
            initialized: AtomicBool::new(false),
            lock: Mutex::new(()),
        }
    }

    /// Initializes the paging allocator with a reserved memory region.
    ///
    /// # Arguments
    ///
    /// * `pool_base` - Base physical address of the reserved pool (must be page-aligned)
    /// * `pool_pages` - Number of pages in the pool
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `pool_base` points to a valid memory region in SMRAM
    /// - The region is not used by any other allocator
    /// - The region has at least `pool_pages * PAGE_SIZE` bytes available
    ///
    /// # Errors
    ///
    /// Returns an error if already initialized or if parameters are invalid.
    pub unsafe fn init(&self, pool_base: u64, pool_pages: usize) -> Result<(), PagingAllocError> {
        if self.initialized.load(Ordering::Acquire) {
            return Err(PagingAllocError::AlreadyInitialized);
        }

        if pool_base == 0 || pool_pages == 0 {
            return Err(PagingAllocError::PoolTooSmall);
        }

        if pool_base % PAGE_SIZE as u64 != 0 {
            return Err(PagingAllocError::InvalidAlignment);
        }

        let _guard = self.lock.lock();

        // Zero the pool region
        unsafe {
            core::ptr::write_bytes(pool_base as *mut u8, 0, pool_pages * PAGE_SIZE);
        }

        self.pool_base.store(pool_base, Ordering::Release);
        self.pool_pages.store(pool_pages, Ordering::Release);
        self.current_offset.store(0, Ordering::Release);
        self.allocated_pages.store(0, Ordering::Release);
        self.initialized.store(true, Ordering::Release);

        log::info!(
            "Paging allocator initialized: base=0x{:016x}, pages={} ({} KB)",
            pool_base,
            pool_pages,
            pool_pages * PAGE_SIZE / 1024
        );

        Ok(())
    }

    /// Allocates a page for page table structures.
    ///
    /// # Arguments
    ///
    /// * `align` - Required alignment in bytes (must be a power of 2 and >= PAGE_SIZE)
    /// * `size` - Size in bytes (must be >= PAGE_SIZE)
    /// * `is_root` - Whether this is a root page table (e.g., PML4)
    ///
    /// # Returns
    ///
    /// The physical address of the allocated page, or an error.
    pub fn allocate_page_internal(
        &self,
        align: u64,
        size: u64,
        _is_root: bool,
    ) -> Result<u64, PagingAllocError> {
        if !self.initialized.load(Ordering::Acquire) {
            return Err(PagingAllocError::NotInitialized);
        }

        // Validate alignment (must be at least PAGE_SIZE and power of 2)
        let align = align.max(PAGE_SIZE as u64);
        if !align.is_power_of_two() {
            return Err(PagingAllocError::InvalidAlignment);
        }

        // Validate size (must be at least PAGE_SIZE)
        let size = size.max(PAGE_SIZE as u64);
        let pages_needed = ((size as usize) + PAGE_SIZE - 1) / PAGE_SIZE;

        let _guard = self.lock.lock();

        let pool_base = self.pool_base.load(Ordering::Acquire);
        let pool_pages = self.pool_pages.load(Ordering::Acquire);
        let current_offset = self.current_offset.load(Ordering::Acquire);

        // Calculate the aligned address
        let current_addr = pool_base + current_offset as u64;
        let aligned_addr = (current_addr + align - 1) & !(align - 1);
        let padding = (aligned_addr - current_addr) as usize;
        let total_bytes = padding + (pages_needed * PAGE_SIZE);

        // Check if we have enough space
        if current_offset + total_bytes > pool_pages * PAGE_SIZE {
            log::error!(
                "Paging allocator out of memory: need {} bytes, have {} bytes remaining",
                total_bytes,
                pool_pages * PAGE_SIZE - current_offset
            );
            return Err(PagingAllocError::OutOfMemory);
        }

        // Update the offset
        self.current_offset
            .store(current_offset + total_bytes, Ordering::Release);
        self.allocated_pages
            .fetch_add(pages_needed, Ordering::Release);

        log::trace!(
            "Paging allocator: allocated {} page(s) at 0x{:016x} (align=0x{:x})",
            pages_needed,
            aligned_addr,
            align
        );

        Ok(aligned_addr)
    }

    /// Returns whether the allocator has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    /// Returns the number of pages allocated.
    pub fn allocated_page_count(&self) -> usize {
        self.allocated_pages.load(Ordering::Acquire)
    }

    /// Returns the number of free pages remaining.
    pub fn free_page_count(&self) -> usize {
        if !self.initialized.load(Ordering::Acquire) {
            return 0;
        }

        let pool_pages = self.pool_pages.load(Ordering::Acquire);
        let current_offset = self.current_offset.load(Ordering::Acquire);
        let used_pages = (current_offset + PAGE_SIZE - 1) / PAGE_SIZE;
        pool_pages.saturating_sub(used_pages)
    }

    /// Returns the base address of the pool.
    pub fn pool_base(&self) -> u64 {
        self.pool_base.load(Ordering::Acquire)
    }

    /// Returns the total size of the pool in bytes.
    pub fn pool_size(&self) -> usize {
        self.pool_pages.load(Ordering::Acquire) * PAGE_SIZE
    }
}

// ============================================================================
// Implementation of patina_paging::PageAllocator trait
// ============================================================================

impl PagingPageAllocator for PagingPoolAllocator {
    /// Allocates a page for page table structures.
    ///
    /// This implements the `patina_paging::PageAllocator` trait.
    fn allocate_page(&mut self, align: u64, size: u64, is_root: bool) -> Result<u64, PtError> {
        self.allocate_page_internal(align, size, is_root)
            .map_err(|e| {
                log::error!("Paging allocator error: {:?}", e);
                match e {
                    PagingAllocError::NotInitialized => PtError::InvalidParameter,
                    PagingAllocError::AlreadyInitialized => PtError::InvalidParameter,
                    PagingAllocError::OutOfMemory => PtError::OutOfResources,
                    PagingAllocError::InvalidAlignment => PtError::InvalidParameter,
                    PagingAllocError::PoolTooSmall => PtError::InvalidParameter,
                }
            })
    }
}

// ============================================================================
// Wrapper for Shared Access
// ============================================================================

/// A wrapper around PagingPoolAllocator that allows shared (non-mutable) access
/// while still implementing the PageAllocator trait.
///
/// This is needed because the `patina_paging::PageAllocator` trait requires `&mut self`,
/// but we want to use a global static allocator with interior mutability.
pub struct SharedPagingAllocator {
    /// The underlying allocator.
    inner: UnsafeCell<&'static PagingPoolAllocator>,
}

// SAFETY: The PagingPoolAllocator uses internal locking for thread safety.
unsafe impl Send for SharedPagingAllocator {}
unsafe impl Sync for SharedPagingAllocator {}

impl SharedPagingAllocator {
    /// Creates a new shared paging allocator wrapper.
    pub const fn new(allocator: &'static PagingPoolAllocator) -> Self {
        Self {
            inner: UnsafeCell::new(allocator),
        }
    }
}

impl PagingPageAllocator for SharedPagingAllocator {
    fn allocate_page(&mut self, align: u64, size: u64, is_root: bool) -> Result<u64, PtError> {
        // SAFETY: The underlying PagingPoolAllocator uses internal locking
        let allocator = unsafe { *self.inner.get() };
        allocator
            .allocate_page_internal(align, size, is_root)
            .map_err(|e| {
                log::error!("Paging allocator error: {:?}", e);
                match e {
                    PagingAllocError::NotInitialized => PtError::InvalidParameter,
                    PagingAllocError::AlreadyInitialized => PtError::InvalidParameter,
                    PagingAllocError::OutOfMemory => PtError::OutOfResources,
                    PagingAllocError::InvalidAlignment => PtError::InvalidParameter,
                    PagingAllocError::PoolTooSmall => PtError::InvalidParameter,
                }
            })
    }
}

// ============================================================================
// Global Instance
// ============================================================================

/// Global paging page allocator instance.
///
/// This must be initialized via `init()` before use.
pub static PAGING_ALLOCATOR: PagingPoolAllocator = PagingPoolAllocator::new();

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paging_allocator_not_initialized() {
        let allocator = PagingPoolAllocator::new();
        assert!(!allocator.is_initialized());
        assert_eq!(allocator.free_page_count(), 0);
        assert_eq!(allocator.allocated_page_count(), 0);
    }

    #[test]
    fn test_paging_allocator_init() {
        let allocator = PagingPoolAllocator::new();
        
        // Create a test buffer
        let mut buffer = vec![0u8; 16 * PAGE_SIZE];
        let base = buffer.as_mut_ptr() as u64;
        // Align to page boundary
        let aligned_base = (base + PAGE_SIZE as u64 - 1) & !(PAGE_SIZE as u64 - 1);
        
        unsafe {
            assert!(allocator.init(aligned_base, 8).is_ok());
        }
        
        assert!(allocator.is_initialized());
        assert_eq!(allocator.free_page_count(), 8);
        assert_eq!(allocator.allocated_page_count(), 0);
    }

    #[test]
    fn test_paging_allocator_double_init() {
        let allocator = PagingPoolAllocator::new();
        
        let mut buffer = vec![0u8; 16 * PAGE_SIZE];
        let base = buffer.as_mut_ptr() as u64;
        let aligned_base = (base + PAGE_SIZE as u64 - 1) & !(PAGE_SIZE as u64 - 1);
        
        unsafe {
            assert!(allocator.init(aligned_base, 8).is_ok());
            assert_eq!(
                allocator.init(aligned_base, 8),
                Err(PagingAllocError::AlreadyInitialized)
            );
        }
    }

    #[test]
    fn test_paging_allocator_allocate() {
        let allocator = PagingPoolAllocator::new();
        
        let mut buffer = vec![0u8; 32 * PAGE_SIZE];
        let base = buffer.as_mut_ptr() as u64;
        let aligned_base = (base + PAGE_SIZE as u64 - 1) & !(PAGE_SIZE as u64 - 1);
        
        unsafe {
            allocator.init(aligned_base, 16).unwrap();
        }

        // Allocate a page
        let result = allocator.allocate_page_internal(PAGE_SIZE as u64, PAGE_SIZE as u64, false);
        assert!(result.is_ok());
        let addr = result.unwrap();
        assert_eq!(addr, aligned_base);
        assert_eq!(allocator.allocated_page_count(), 1);

        // Allocate another page
        let result2 = allocator.allocate_page_internal(PAGE_SIZE as u64, PAGE_SIZE as u64, false);
        assert!(result2.is_ok());
        let addr2 = result2.unwrap();
        assert_eq!(addr2, aligned_base + PAGE_SIZE as u64);
        assert_eq!(allocator.allocated_page_count(), 2);
    }
}
