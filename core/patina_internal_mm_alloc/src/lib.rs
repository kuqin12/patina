//! Shared MM Pool Allocator
//!
//! This crate provides a trait-based page allocator abstraction and a generic pool
//! allocator that can be shared between the MM Supervisor Core and MM User Core.
//!
//! ## Design
//!
//! The [`PageAllocatorBackend`] trait abstracts the page allocation mechanism:
//! - The **supervisor** implements it with a direct SMRAM bitmap allocator.
//! - The **user core** implements it by issuing `syscall` instructions that thunk
//!   into the supervisor for page allocation.
//!
//! The [`PoolAllocator`] is a bump-allocator built on top of any `PageAllocatorBackend`.
//! It implements [`GlobalAlloc`] so it can be used as `#[global_allocator]` in both cores.
//!
//! ## Block Management
//!
//! Block metadata is stored **in-band** at the start of each page allocation, forming
//! an intrusive linked list. This means there is no fixed cap on the number of blocks —
//! the allocator grows dynamically as needed by requesting more pages from the backend.
//! When all allocations within a block are freed, the block is unlinked from the list
//! and the pages are returned to the backend.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!
#![no_std]

use core::{
    alloc::{GlobalAlloc, Layout},
    mem,
    ptr,
};

use spin::Mutex;

// ============================================================================
// Constants
// ============================================================================

/// Standard UEFI page size (4 KB).
pub const PAGE_SIZE: usize = 4096;

/// Minimum allocation size for the pool allocator.
const MIN_POOL_ALLOC_SIZE: usize = 16;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during page allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageAllocError {
    /// The allocator has not been initialized.
    NotInitialized,
    /// No free pages available to satisfy the request.
    OutOfMemory,
    /// The requested address is not aligned to page boundary.
    NotAligned,
    /// The address is not within any known SMRAM region.
    InvalidAddress,
    /// The address was not previously allocated.
    NotAllocated,
    /// Too many regions to track.
    TooManyRegions,
    /// A syscall to the supervisor failed.
    SyscallFailed(u64),
}

// ============================================================================
// Page Allocator Backend Trait
// ============================================================================

/// Trait for page-granularity memory allocation.
///
/// Implementors provide the actual page allocation mechanism. The supervisor
/// implements this with direct SMRAM bitmap tracking; the user core implements
/// this by issuing syscalls to the supervisor.
pub trait PageAllocatorBackend: Send + Sync {
    /// Allocates `num_pages` contiguous pages.
    ///
    /// Returns the physical base address of the allocated region on success.
    fn allocate_pages(&self, num_pages: usize) -> Result<u64, PageAllocError>;

    /// Frees `num_pages` contiguous pages starting at `addr`.
    fn free_pages(&self, addr: u64, num_pages: usize) -> Result<(), PageAllocError>;

    /// Returns whether the page allocator has been initialized and is ready for use.
    fn is_initialized(&self) -> bool;
}

// ============================================================================
// Pool Block Header (intrusive linked list node)
// ============================================================================

/// In-band header stored at the beginning of each pool page block.
///
/// By placing the metadata inside the allocated pages themselves, we avoid
/// any fixed-size bookkeeping array. Blocks form a singly-linked list so
/// traversal, insertion, and removal are straightforward.
#[repr(C)]
struct PoolBlockHeader {
    /// Pointer to the next block in the linked list (`null` if this is the tail).
    next: *mut PoolBlockHeader,
    /// Number of pages backing this block (includes the header).
    num_pages: usize,
    /// Current bump offset (in bytes from the block base). Starts just past the header.
    offset: usize,
    /// Number of live allocations served from this block.
    alloc_count: usize,
}

impl PoolBlockHeader {
    /// Base address of this block (== address of the header itself).
    fn base(&self) -> usize {
        self as *const Self as usize
    }

    /// Total usable capacity of this block in bytes.
    fn capacity(&self) -> usize {
        self.num_pages * PAGE_SIZE
    }

    /// Remaining bytes available for bump allocation.
    fn remaining(&self) -> usize {
        self.capacity().saturating_sub(self.offset)
    }

    /// Returns `true` if the given address falls within this block's page range.
    fn contains(&self, addr: usize) -> bool {
        addr >= self.base() && addr < self.base() + self.capacity()
    }

    /// Try to bump-allocate `layout` from this block.
    fn try_alloc(&mut self, layout: Layout) -> Option<*mut u8> {
        let current_ptr = self.base() + self.offset;
        let align = layout.align().max(MIN_POOL_ALLOC_SIZE);
        let aligned_ptr = (current_ptr + align - 1) & !(align - 1);
        let padding = aligned_ptr - current_ptr;
        let total_size = padding + layout.size();

        if total_size > self.remaining() {
            return None;
        }

        self.offset += total_size;
        self.alloc_count += 1;

        Some(aligned_ptr as *mut u8)
    }
}

// ============================================================================
// Pool Allocator
// ============================================================================

/// Pool allocator built on top of a [`PageAllocatorBackend`].
///
/// This allocator provides smaller-granularity allocations by requesting
/// full pages from the backend and subdividing them via bump allocation.
///
/// ## Design
///
/// - Block metadata is stored **in-band** at the start of each page allocation,
///   forming an intrusive singly-linked list. There is no fixed cap on the number
///   of blocks — the allocator grows dynamically.
/// - Uses bump allocation within each block for fast O(1) allocations.
/// - When a block is exhausted, a new page allocation is requested from the backend.
/// - When all allocations within a block are freed, the block is unlinked and its
///   pages are returned to the backend.
///
/// ## Thread Safety
///
/// Uses a spin lock for thread safety and can be used as a global allocator
/// in `no_std` environments.
pub struct PoolAllocator<P: PageAllocatorBackend + 'static> {
    /// Reference to the underlying page allocator.
    page_allocator: &'static P,
    /// Head of the intrusive linked list of pool blocks.
    ///
    /// Protected by `lock`. The raw pointer is `!Send` but the outer struct
    /// provides `Send + Sync` via the lock.
    head: Mutex<*mut PoolBlockHeader>,
}

// SAFETY: The PoolAllocator uses internal locking (spin::Mutex) for all accesses
// to the block linked list. The raw pointer is only dereferenced under the lock.
unsafe impl<P: PageAllocatorBackend> Send for PoolAllocator<P> {}
unsafe impl<P: PageAllocatorBackend> Sync for PoolAllocator<P> {}

impl<P: PageAllocatorBackend> PoolAllocator<P> {
    /// Creates a new pool allocator backed by the given page allocator.
    pub const fn new(page_allocator: &'static P) -> Self {
        Self {
            page_allocator,
            head: Mutex::new(ptr::null_mut()),
        }
    }

    /// Allocate a new page block large enough for `min_size` bytes of payload
    /// and prepend it to the linked list.
    ///
    /// Returns a mutable reference to the new block's header on success.
    ///
    /// # Safety
    ///
    /// Caller must hold `self.head` locked (the lock guard is passed in so
    /// the new block can be linked).
    fn allocate_new_block<'a>(
        &self,
        head: &mut *mut PoolBlockHeader,
        min_size: usize,
    ) -> Option<&'a mut PoolBlockHeader> {
        let header_size = mem::size_of::<PoolBlockHeader>();
        let needed = min_size + header_size;
        let num_pages = ((needed + PAGE_SIZE - 1) / PAGE_SIZE).max(1);

        let base = match self.page_allocator.allocate_pages(num_pages) {
            Ok(addr) => addr,
            Err(e) => {
                log::warn!("Pool allocator: failed to allocate {} pages: {:?}", num_pages, e);
                return None;
            }
        };

        // SAFETY: `base` is a freshly allocated, page-aligned region of at least
        // `num_pages * PAGE_SIZE` bytes. We place our header at offset 0.
        let header = unsafe { &mut *(base as *mut PoolBlockHeader) };
        header.next = *head;
        header.num_pages = num_pages;
        header.offset = header_size; // bump pointer starts right after the header
        header.alloc_count = 0;

        // Prepend to the list.
        *head = header as *mut PoolBlockHeader;

        log::trace!(
            "Pool allocator: new block at {:#018x} ({} pages)",
            base,
            num_pages,
        );

        Some(header)
    }
}

unsafe impl<P: PageAllocatorBackend> GlobalAlloc for PoolAllocator<P> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !self.page_allocator.is_initialized() {
            return ptr::null_mut();
        }

        let mut head = self.head.lock();

        // Walk the linked list, try to bump-alloc from an existing block.
        {
            let mut current = *head;
            while !current.is_null() {
                // SAFETY: `current` was written by us under the same lock.
                let block = unsafe { &mut *current };
                if let Some(ptr) = block.try_alloc(layout) {
                    return ptr;
                }
                current = block.next;
            }
        }

        // No existing block had space — allocate a new one and retry.
        if let Some(block) = self.allocate_new_block(&mut head, layout.size()) {
            if let Some(ptr) = block.try_alloc(layout) {
                return ptr;
            }
        }

        ptr::null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        if ptr.is_null() {
            return;
        }

        let mut head = self.head.lock();
        let addr = ptr as usize;

        let mut prev: *mut PoolBlockHeader = ptr::null_mut();
        let mut current = *head;

        while !current.is_null() {
            // SAFETY: `current` was written by us under the same lock.
            let block = unsafe { &mut *current };

            if block.contains(addr) {
                block.alloc_count = block.alloc_count.saturating_sub(1);

                // If the block is now empty, unlink it and free the pages.
                if block.alloc_count == 0 {
                    let next = block.next;
                    let base = block.base() as u64;
                    let num_pages = block.num_pages;

                    if prev.is_null() {
                        *head = next;
                    } else {
                        // SAFETY: `prev` is a valid block we visited earlier.
                        unsafe { (*prev).next = next };
                    }

                    if let Err(e) = self.page_allocator.free_pages(base, num_pages) {
                        log::warn!(
                            "Pool allocator: failed to free block at {:#018x}: {:?}",
                            base,
                            e,
                        );
                    } else {
                        log::trace!("Pool allocator: freed block at {:#018x} ({} pages)", base, num_pages);
                    }
                }

                return;
            }

            prev = current;
            current = block.next;
        }

        log::warn!(
            "Pool allocator: dealloc called with unknown pointer {:#018x}",
            addr,
        );
    }
}
