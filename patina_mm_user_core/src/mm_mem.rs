//! MM User Core Memory Allocator
//!
//! Provides a [`SyscallPageAllocator`] that implements [`PageAllocatorBackend`]
//! by issuing `syscall` instructions to the MM Supervisor for page allocation
//! and deallocation.
//!
//! The pool allocator from `patina_internal_mm_alloc` is re-used on top of this
//! backend, and wired up as the `#[global_allocator]`.
//!
//! ## Syscall ABI
//!
//! The MM Supervisor exposes page allocation via the following syscall indices
//! (defined in SysCallLib.h / `SyscallIndex` enum in the supervisor):
//!
//! | Syscall     | RAX       | RDX (arg1)     | R8 (arg2)      | R9 (arg3)   |
//! |-------------|-----------|----------------|----------------|-------------|
//! | AllocPage   | `0x10004` | alloc_type (0) | mem_type (6)   | page_count  |
//! | FreePage    | `0x10005` | address        | page_count     | 0           |
//!
//! The supervisor returns:
//! - RAX: result value (allocated address for AllocPage, 0 for FreePage)
//! - RDX: EFI status (0 = success)
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use core::sync::atomic::{AtomicBool, Ordering};

use patina_internal_mm_alloc::{PageAllocError, PageAllocatorBackend, PoolAllocator};
use patina_internal_mm_common::SyscallIndex;

// ============================================================================
// AllocateType constants (matching EFI_ALLOCATE_TYPE)
// ============================================================================

/// `AllocateAnyPages` — allocate any available pages.
const ALLOCATE_ANY_PAGES: u64 = 0;

// ============================================================================
// Memory type constants (matching EFI_MEMORY_TYPE)
// ============================================================================

/// `EfiRuntimeServicesData` — the memory type used for MM pool allocations.
const RUNTIME_SERVICES_DATA: u64 = 6;

// ============================================================================
// Syscall helpers
// ============================================================================

/// Result of a raw syscall to the supervisor.
#[derive(Debug, Clone, Copy)]
struct RawSyscallResult {
    /// Value returned in RAX (e.g., allocated address).
    value: u64,
    /// Status returned in RDX (EFI_STATUS).
    status: u64,
}

/// Issue a `syscall` instruction to the MM Supervisor.
///
/// # ABI
///
/// - RAX = call_index
/// - RDX = arg1
/// - R8  = arg2
/// - R9  = arg3
///
/// On return:
/// - RAX = result value
/// - RDX = status
///
/// # Safety
///
/// This is inherently unsafe — it transfers control to the supervisor and
/// the arguments must be valid for the specific syscall index.
#[cfg(target_arch = "x86_64")]
unsafe fn raw_syscall(call_index: u64, arg1: u64, arg2: u64, arg3: u64) -> RawSyscallResult {
    let value: u64;
    let status: u64;

    // The `syscall` instruction uses:
    //   RAX = syscall number
    //   RCX = return address (set by CPU on syscall entry, clobbered)
    //   R11 = RFLAGS (set by CPU on syscall entry, clobbered)
    //   RDX = arg1 (also used for status return)
    //   R8  = arg2
    //   R9  = arg3
    //
    // On return from the supervisor:
    //   RAX = result value
    //   RDX = status
    unsafe {
        core::arch::asm!(
            "syscall",
            inlateout("rax") call_index => value,
            inlateout("rdx") arg1 => status,
            in("r8") arg2,
            in("r9") arg3,
            // RCX and R11 are clobbered by the `syscall` instruction.
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }

    RawSyscallResult { value, status }
}

// ============================================================================
// Communication buffer validation
// ============================================================================

/// Validate that a given memory range is a valid MM communication buffer by
/// issuing the `MmIsCommBuffer` syscall to the supervisor.
///
/// Returns `true` if the supervisor confirms the range falls entirely within
/// the user communication buffer region.
pub fn is_comm_buffer(address: u64, size: u64) -> bool {
    let result = unsafe { raw_syscall(SyscallIndex::MmIsCommBuffer.as_u64(), address, size, 0) };
    result.value != 0
}

// ============================================================================
// SyscallPageAllocator
// ============================================================================

/// A page allocator backend that issues `syscall` instructions to the MM Supervisor.
///
/// This is used as the [`PageAllocatorBackend`] for the MM User Core's
/// [`PoolAllocator`] and global allocator.
///
/// ## Initialization
///
/// Call [`SyscallPageAllocator::set_initialized`] after the user core has been
/// set up and is ready to issue syscalls (i.e., during `StartUserCore` handling,
/// before driver dispatch begins).
pub struct SyscallPageAllocator {
    /// Whether the allocator has been activated. Before this is set, all
    /// allocations will fail immediately. This prevents accidental allocations
    /// before the syscall interface is ready.
    initialized: AtomicBool,
}

// SAFETY: SyscallPageAllocator uses an atomic flag and the syscall interface is
// re-entrant from the BSP.
unsafe impl Send for SyscallPageAllocator {}
unsafe impl Sync for SyscallPageAllocator {}

impl SyscallPageAllocator {
    /// Creates a new uninitialized syscall page allocator.
    pub const fn new() -> Self {
        Self {
            initialized: AtomicBool::new(false),
        }
    }

    /// Marks the allocator as ready. Must be called after the syscall interface
    /// is available (i.e., early in `StartUserCore` handling).
    pub fn set_initialized(&self) {
        self.initialized.store(true, Ordering::Release);
        log::info!("SyscallPageAllocator initialized — heap is now available.");
    }
}

impl PageAllocatorBackend for SyscallPageAllocator {
    fn allocate_pages(&self, num_pages: usize) -> Result<u64, PageAllocError> {
        if !self.initialized.load(Ordering::Acquire) {
            return Err(PageAllocError::NotInitialized);
        }

        if num_pages == 0 {
            return Err(PageAllocError::OutOfMemory);
        }

        let result = unsafe {
            raw_syscall(
                SyscallIndex::AllocPage.as_u64(),
                ALLOCATE_ANY_PAGES,
                RUNTIME_SERVICES_DATA,
                num_pages as u64,
            )
        };

        if result.status != 0 {
            log::warn!(
                "SyscallPageAllocator: AllocPage({} pages) failed with status 0x{:x}",
                num_pages,
                result.status
            );
            return Err(PageAllocError::SyscallFailed(result.status));
        }

        log::trace!(
            "SyscallPageAllocator: allocated {} page(s) at 0x{:016x}",
            num_pages,
            result.value
        );

        Ok(result.value)
    }

    fn free_pages(&self, addr: u64, num_pages: usize) -> Result<(), PageAllocError> {
        if !self.initialized.load(Ordering::Acquire) {
            return Err(PageAllocError::NotInitialized);
        }

        unsafe {
            raw_syscall(
                SyscallIndex::FreePage.as_u64(),
                addr,
                num_pages as u64,
                0,
            )
        };

        log::trace!(
            "SyscallPageAllocator: freed {} page(s) at 0x{:016x}",
            num_pages,
            addr
        );

        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }
}

// ============================================================================
// Global Allocator Instance
// ============================================================================

/// Global page allocator instance for the user core.
///
/// This issues syscalls to the supervisor for actual page allocation.
/// Call [`SYSCALL_PAGE_ALLOCATOR.set_initialized()`] during `StartUserCore`
/// to enable the heap.
pub static SYSCALL_PAGE_ALLOCATOR: SyscallPageAllocator = SyscallPageAllocator::new();

/// Global pool allocator instance.
///
/// Uses the shared [`PoolAllocator`] from `patina_internal_mm_alloc`,
/// backed by [`SyscallPageAllocator`] for page allocation via syscalls.
#[global_allocator]
static GLOBAL_ALLOCATOR: PoolAllocator<SyscallPageAllocator> =
    PoolAllocator::new(&SYSCALL_PAGE_ALLOCATOR);
