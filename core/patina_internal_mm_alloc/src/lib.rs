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
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!
#![no_std]

use core::{
    alloc::{GlobalAlloc, Layout},
    cell::UnsafeCell,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};

use spin::Mutex;

// ============================================================================
// Constants
// ============================================================================

/// Standard UEFI page size (4 KB).
pub const PAGE_SIZE: usize = 4096;

/// Maximum number of pool page blocks we can track dynamically.
const MAX_POOL_BLOCKS: usize = 64;

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
// Pool Block Tracking
// ============================================================================

/// Tracks a single pool page block allocated from the page allocator.
#[derive(Clone, Copy)]
struct PoolBlock {
    /// Base address of this pool block.
    base: u64,
    /// Number of pages in this block.
    num_pages: usize,
    /// Current offset into the block for bump allocation.
    offset: usize,
    /// Number of active allocations from this block.
    alloc_count: usize,
}

impl PoolBlock {
    const fn new() -> Self {
        Self {
            base: 0,
            num_pages: 0,
            offset: 0,
            alloc_count: 0,
        }
    }

    fn is_valid(&self) -> bool {
        self.base != 0 && self.num_pages > 0
    }

    fn capacity(&self) -> usize {
        self.num_pages * PAGE_SIZE
    }

    fn remaining(&self) -> usize {
        self.capacity().saturating_sub(self.offset)
    }

    /// Try to allocate from this block using bump allocation.
    fn try_alloc(&mut self, layout: Layout) -> Option<*mut u8> {
        if !self.is_valid() {
            return None;
        }

        // Calculate aligned offset
        let current_ptr = self.base as usize + self.offset;
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

    /// Record a deallocation from this block.
    fn record_dealloc(&mut self) {
        if self.alloc_count > 0 {
            self.alloc_count -= 1;
        }
    }

    /// Returns true if this block has no active allocations and can be freed.
    fn is_empty(&self) -> bool {
        self.alloc_count == 0
    }
}

// ============================================================================
// Pool Allocator
// ============================================================================

/// Pool allocator built on top of a [`PageAllocatorBackend`].
///
/// This allocator provides smaller-granularity allocations by allocating
/// pages from the backend and subdividing them for pool allocations.
///
/// ## Design
///
/// - Uses bump allocation within each pool block for fast allocations
/// - Tracks multiple pool blocks for dynamic growth
/// - When a block is exhausted, allocates new pages
/// - Blocks are freed back to the page allocator when all allocations are released
///
/// ## Thread Safety
///
/// This allocator uses a spin lock for thread safety and can be used as a
/// global allocator in multi-threaded environments.
pub struct PoolAllocator<P: PageAllocatorBackend + 'static> {
    /// Reference to the underlying page allocator.
    page_allocator: &'static P,
    /// Pool blocks for tracking allocated pages.
    blocks: UnsafeCell<[PoolBlock; MAX_POOL_BLOCKS]>,
    /// Number of valid blocks.
    block_count: AtomicUsize,
    /// Lock for thread safety.
    lock: Mutex<()>,
}

// SAFETY: The PoolAllocator uses internal locking for thread safety.
unsafe impl<P: PageAllocatorBackend> Send for PoolAllocator<P> {}
unsafe impl<P: PageAllocatorBackend> Sync for PoolAllocator<P> {}

impl<P: PageAllocatorBackend> PoolAllocator<P> {
    /// Creates a new pool allocator using the given page allocator backend.
    pub const fn new(page_allocator: &'static P) -> Self {
        const EMPTY_BLOCK: PoolBlock = PoolBlock::new();
        Self {
            page_allocator,
            blocks: UnsafeCell::new([EMPTY_BLOCK; MAX_POOL_BLOCKS]),
            block_count: AtomicUsize::new(0),
            lock: Mutex::new(()),
        }
    }

    /// Allocates a new pool block from the page allocator.
    fn allocate_new_block(&self, min_size: usize) -> Option<usize> {
        let _guard = self.lock.lock();

        // SAFETY: We have exclusive access via the lock
        let blocks = unsafe { &mut *self.blocks.get() };
        let block_count = self.block_count.load(Ordering::Acquire);

        if block_count >= MAX_POOL_BLOCKS {
            log::warn!("Pool allocator: maximum block count reached");
            return None;
        }

        // Calculate number of pages needed (at least 1)
        let num_pages = ((min_size + PAGE_SIZE - 1) / PAGE_SIZE).max(1);

        // Allocate pages from the backend
        let base = match self.page_allocator.allocate_pages(num_pages) {
            Ok(addr) => addr,
            Err(e) => {
                log::warn!("Pool allocator: failed to allocate pages: {:?}", e);
                return None;
            }
        };

        // Initialize the new block
        let block_index = block_count;
        blocks[block_index] = PoolBlock {
            base,
            num_pages,
            offset: 0,
            alloc_count: 0,
        };

        self.block_count.store(block_count + 1, Ordering::Release);

        log::trace!(
            "Pool allocator: allocated new block {} at 0x{:016x} ({} pages)",
            block_index,
            base,
            num_pages
        );

        Some(block_index)
    }

    /// Tries to reclaim empty blocks back to the page allocator.
    fn try_reclaim_empty_blocks(&self) {
        let _guard = self.lock.lock();

        // SAFETY: We have exclusive access via the lock
        let blocks = unsafe { &mut *self.blocks.get() };
        let block_count = self.block_count.load(Ordering::Acquire);

        for i in 0..block_count {
            if blocks[i].is_valid() && blocks[i].is_empty() {
                // Free this block back to the page allocator
                if let Err(e) = self
                    .page_allocator
                    .free_pages(blocks[i].base, blocks[i].num_pages)
                {
                    log::warn!(
                        "Pool allocator: failed to free block {} back to page allocator: {:?}",
                        i,
                        e
                    );
                } else {
                    log::trace!("Pool allocator: reclaimed block {} at 0x{:016x}", i, blocks[i].base);
                    // Mark block as invalid
                    blocks[i] = PoolBlock::new();
                }
            }
        }
    }

    /// Finds which block contains the given address.
    fn find_block_for_address(&self, ptr: *mut u8) -> Option<usize> {
        // SAFETY: We're only reading, caller should ensure proper synchronization
        let blocks = unsafe { &*self.blocks.get() };
        let block_count = self.block_count.load(Ordering::Acquire);
        let addr = ptr as u64;

        for i in 0..block_count {
            let block = &blocks[i];
            if block.is_valid() {
                let block_end = block.base + block.capacity() as u64;
                if addr >= block.base && addr < block_end {
                    return Some(i);
                }
            }
        }

        None
    }
}

unsafe impl<P: PageAllocatorBackend> GlobalAlloc for PoolAllocator<P> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !self.page_allocator.is_initialized() {
            return ptr::null_mut();
        }

        let _guard = self.lock.lock();

        // SAFETY: We have exclusive access via the lock
        let blocks = unsafe { &mut *self.blocks.get() };
        let block_count = self.block_count.load(Ordering::Acquire);

        // First, try to allocate from an existing block
        for i in 0..block_count {
            if blocks[i].is_valid() {
                if let Some(ptr) = blocks[i].try_alloc(layout) {
                    return ptr;
                }
            }
        }

        // Need to drop the guard before allocating a new block
        drop(_guard);

        // No existing block has space, allocate a new block
        if let Some(block_index) = self.allocate_new_block(layout.size()) {
            let _guard = self.lock.lock();
            // SAFETY: We have exclusive access via the lock
            let blocks = unsafe { &mut *self.blocks.get() };

            if let Some(ptr) = blocks[block_index].try_alloc(layout) {
                return ptr;
            }
        }

        ptr::null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        if ptr.is_null() {
            return;
        }

        let _guard = self.lock.lock();

        // SAFETY: We have exclusive access via the lock
        let blocks = unsafe { &mut *self.blocks.get() };

        if let Some(block_index) = self.find_block_for_address(ptr) {
            blocks[block_index].record_dealloc();

            // Optionally try to reclaim empty blocks
            if blocks[block_index].is_empty() {
                drop(_guard);
                self.try_reclaim_empty_blocks();
            }
        } else {
            log::warn!(
                "Pool allocator: dealloc called with unknown pointer 0x{:016x}",
                ptr as u64
            );
        }
    }
}
