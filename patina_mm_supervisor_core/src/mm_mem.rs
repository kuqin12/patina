//! MM Supervisor Core Page and Pool Allocators
//!
//! Provides a page-granularity memory allocator and a pool allocator for the MM Supervisor Core.
//!
//! ## Page Allocator
//!
//! When the one-time initialization routine is called, it will mark the blocks reported under
//! `gEfiSmmSmramMemoryGuid` or `gEfiMmPeiMmramMemoryReserveGuid` in the HOB list accordingly.
//! Blocks that have the `EFI_ALLOCATED` bit set in the `RegionState` field will be marked as allocated,
//! indicating they are in use. All other blocks will be marked as free.
//!
//! The page allocator is fully dynamic:
//! - No fixed limit on number of SMRAM regions
//! - No fixed limit on pages per region (supports up to 4GB per region)
//! - Bookkeeping is stored in SMRAM itself
//!
//! The page allocator provides:
//! - `allocate_pages(num_pages)` - Allocate contiguous pages
//! - `free_pages(addr, num_pages)` - Free previously allocated pages
//!
//! ## Pool Allocator
//!
//! Built on top of the page allocator, the pool allocator provides smaller-granularity allocations.
//! It allocates pages from the page allocator and subdivides them for pool allocations.
//! When a pool page is exhausted, more pages are allocated as needed.
//!
//! The pool allocator implements the `GlobalAlloc` trait for use as a global allocator.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use core::{
    alloc::{GlobalAlloc, Layout},
    cell::UnsafeCell,
    ffi::c_void,
    mem::size_of,
    ptr,
    slice,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use patina::pi::hob::{Hob, PhaseHandoffInformationTable};
use r_efi::efi;
use spin::Mutex;
use patina_paging::{MemoryAttributes, PageTable};

// ============================================================================
// Constants
// ============================================================================

/// Standard UEFI page size (4 KB).
pub const PAGE_SIZE: usize = 4096;

/// Bits per byte.
const BITS_PER_BYTE: usize = 8;

/// Maximum number of pool page blocks we can track dynamically.
/// This allows for dynamic bookkeeping of allocated pool pages.
const MAX_POOL_BLOCKS: usize = 64;

/// Minimum allocation size for the pool allocator.
const MIN_POOL_ALLOC_SIZE: usize = 16;

/// EFI_ALLOCATED bit in RegionState.
pub const EFI_ALLOCATED: u64 = 0x0000000000000010;

// GUID for gEfiSmmSmramMemoryGuid
// { 0x6dadf1d1, 0xd4cc, 0x4910, { 0xbb, 0x6e, 0x82, 0xb1, 0xfd, 0x80, 0xff, 0x3d }}
pub const SMM_SMRAM_MEMORY_GUID: efi::Guid = efi::Guid::from_fields(
    0x6dadf1d1,
    0xd4cc,
    0x4910,
    0xbb,
    0x6e,
    &[0x82, 0xb1, 0xfd, 0x80, 0xff, 0x3d],
);

// GUID for gEfiMmPeiMmramMemoryReserveGuid
// { 0x0703f912, 0xbf8d, 0x4e2a, { 0xbe, 0x07, 0xab, 0x27, 0x25, 0x25, 0xc5, 0x92 }}
pub const MM_PEI_MMRAM_MEMORY_RESERVE_GUID: efi::Guid = efi::Guid::from_fields(
    0x0703f912,
    0xbf8d,
    0x4e2a,
    0xbe,
    0x07,
    &[0xab, 0x27, 0x25, 0x25, 0xc5, 0x92],
);

// ============================================================================
// Error Types
// ============================================================================

/// Type of memory allocation - distinguishes supervisor-internal vs user/driver allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AllocationType {
    /// Supervisor-internal allocation (e.g., for core data structures).
    /// These are typically never freed and may have stricter protections.
    Supervisor = 0,
    /// User/driver allocation (e.g., for MM driver requests).
    /// These can be allocated and freed by external code.
    User = 1,
}

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
}

// ============================================================================
// SMRAM Descriptor (matches EFI_SMRAM_DESCRIPTOR)
// ============================================================================

/// SMRAM descriptor structure matching EFI_SMRAM_DESCRIPTOR.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SmramDescriptor {
    /// Physical start address of the SMRAM region.
    pub physical_start: efi::PhysicalAddress,
    /// CPU start address (may differ from physical for remapping).
    pub cpu_start: efi::PhysicalAddress,
    /// Size of the SMRAM region in bytes.
    pub physical_size: u64,
    /// Region state flags (EFI_ALLOCATED, etc.).
    pub region_state: u64,
}

// ============================================================================
// SMRAM Reserve HOB structure
// ============================================================================

/// SMRAM reserve descriptor count structure.
/// This is the data that immediately follows a GuidHob with SMM_SMRAM_MEMORY_GUID
/// or MM_PEI_MMRAM_MEMORY_RESERVE_GUID.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SmramReserveHobData {
    /// Number of SMRAM descriptors that follow.
    pub number_of_smram_regions: u32,
    /// Reserved for alignment.
    pub reserved: u32,
    // SmramDescriptor array follows immediately after
}

// ============================================================================
// Memory Region Tracking (Dynamic)
// ============================================================================

/// Metadata for a single SMRAM region.
/// This struct is stored in the bookkeeping pages, not statically.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RegionInfo {
    /// Base physical address of the region.
    pub base: u64,
    /// Total number of pages in this region.
    pub total_pages: usize,
    /// Starting bit index in the global allocation bitmap.
    pub bitmap_start_bit: usize,
}

/// Internal state for the page allocator, stored in bookkeeping pages.
#[repr(C)]
struct AllocatorState {
    /// Number of regions.
    region_count: usize,
    /// Total number of pages across all regions.
    total_pages: usize,
    /// Number of pages used for bookkeeping.
    bookkeeping_pages: usize,
    /// Base address of bookkeeping memory.
    bookkeeping_base: u64,
    // Followed by:
    // - RegionInfo array (region_count entries)
    // - Allocation bitmap (total_pages bits, rounded up to bytes)
    // - Type bitmap (total_pages bits, rounded up to bytes)
}

// ============================================================================
// Page Allocator (Dynamic)
// ============================================================================

/// Page-granularity allocator for SMRAM memory.
///
/// This allocator is fully dynamic:
/// - No fixed limit on number of SMRAM regions
/// - No fixed limit on pages per region (supports up to 4GB per region)
/// - Bookkeeping data structures are allocated from SMRAM itself
///
/// ## Initialization
///
/// During initialization, the allocator:
/// 1. Scans HOBs to count regions and total pages
/// 2. Calculates bookkeeping space needed
/// 3. Reserves pages from the first available region for bookkeeping
/// 4. Initializes bitmaps in the reserved pages
pub struct PageAllocator {
    /// Pointer to the allocator state (stored in SMRAM).
    state: UnsafeCell<*mut AllocatorState>,
    /// Lock for thread safety.
    lock: Mutex<()>,
    /// Whether the allocator has been initialized.
    initialized: AtomicBool,
}

// SAFETY: The PageAllocator uses internal locking for thread safety.
unsafe impl Send for PageAllocator {}
unsafe impl Sync for PageAllocator {}

impl PageAllocator {
    /// Creates a new uninitialized page allocator.
    pub const fn new() -> Self {
        Self {
            state: UnsafeCell::new(ptr::null_mut()),
            lock: Mutex::new(()),
            initialized: AtomicBool::new(false),
        }
    }

    /// Calculates the number of pages needed for bookkeeping.
    ///
    /// Bookkeeping includes:
    /// - AllocatorState header
    /// - RegionInfo array
    /// - Allocation bitmap (1 bit per page)
    /// - Type bitmap (1 bit per page)
    fn calculate_bookkeeping_pages(region_count: usize, total_pages: usize) -> usize {
        let header_size = size_of::<AllocatorState>();
        let regions_size = region_count * size_of::<RegionInfo>();
        let bitmap_bytes = (total_pages + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
        let total_bytes = header_size + regions_size + bitmap_bytes * 2; // alloc + type bitmaps
        (total_bytes + PAGE_SIZE - 1) / PAGE_SIZE
    }

    /// Gets the regions array from the state.
    unsafe fn get_regions(&self) -> &[RegionInfo] {
        unsafe {
            let state = *self.state.get();
            if state.is_null() {
                return &[];
            }
            let region_count = (*state).region_count;
            let regions_ptr = (state as *const u8).add(size_of::<AllocatorState>()) as *const RegionInfo;
            slice::from_raw_parts(regions_ptr, region_count)
        }
    }

    /// Gets the regions array mutably from the state.
    unsafe fn get_regions_mut(&self) -> &mut [RegionInfo] {
        unsafe {
            let state = *self.state.get();
            if state.is_null() {
                return &mut [];
            }
            let region_count = (*state).region_count;
            let regions_ptr = (state as *mut u8).add(size_of::<AllocatorState>()) as *mut RegionInfo;
            slice::from_raw_parts_mut(regions_ptr, region_count)
        }
    }

    /// Gets the allocation bitmap from the state.
    unsafe fn get_alloc_bitmap(&self) -> &[u8] {
        unsafe {
            let state = *self.state.get();
            if state.is_null() {
                return &[];
            }
            let region_count = (*state).region_count;
            let total_pages = (*state).total_pages;
            let bitmap_bytes = (total_pages + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
            let bitmap_ptr = (state as *const u8)
                .add(size_of::<AllocatorState>())
                .add(region_count * size_of::<RegionInfo>());
            slice::from_raw_parts(bitmap_ptr, bitmap_bytes)
        }
    }

    /// Gets the allocation bitmap mutably from the state.
    unsafe fn get_alloc_bitmap_mut(&self) -> &mut [u8] {
        unsafe {
            let state = *self.state.get();
            if state.is_null() {
                return &mut [];
            }
            let region_count = (*state).region_count;
            let total_pages = (*state).total_pages;
            let bitmap_bytes = (total_pages + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
            let bitmap_ptr = (state as *mut u8)
                .add(size_of::<AllocatorState>())
                .add(region_count * size_of::<RegionInfo>());
            slice::from_raw_parts_mut(bitmap_ptr, bitmap_bytes)
        }
    }

    /// Gets the type bitmap from the state.
    unsafe fn get_type_bitmap(&self) -> &[u8] {
        unsafe {
            let state = *self.state.get();
            if state.is_null() {
                return &[];
            }
            let region_count = (*state).region_count;
            let total_pages = (*state).total_pages;
            let bitmap_bytes = (total_pages + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
            let type_bitmap_ptr = (state as *const u8)
                .add(size_of::<AllocatorState>())
                .add(region_count * size_of::<RegionInfo>())
                .add(bitmap_bytes);
            slice::from_raw_parts(type_bitmap_ptr, bitmap_bytes)
        }
    }

    /// Gets the type bitmap mutably from the state.
    unsafe fn get_type_bitmap_mut(&self) -> &mut [u8] {
        unsafe {
            let state = *self.state.get();
            if state.is_null() {
                return &mut [];
            }
            let region_count = (*state).region_count;
            let total_pages = (*state).total_pages;
            let bitmap_bytes = (total_pages + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
            let type_bitmap_ptr = (state as *mut u8)
                .add(size_of::<AllocatorState>())
                .add(region_count * size_of::<RegionInfo>())
                .add(bitmap_bytes);
            slice::from_raw_parts_mut(type_bitmap_ptr, bitmap_bytes)
        }
    }

    /// Checks if a global bit index is allocated.
    unsafe fn is_bit_allocated(&self, bit_index: usize) -> bool {
        unsafe {
            let bitmap = self.get_alloc_bitmap();
            let byte_index = bit_index / BITS_PER_BYTE;
            let bit_offset = bit_index % BITS_PER_BYTE;
            if byte_index >= bitmap.len() {
                return true; // Out of bounds = allocated
            }
            (bitmap[byte_index] & (1 << bit_offset)) != 0
        }
    }

    /// Gets the allocation type for a global bit index.
    unsafe fn get_bit_type(&self, bit_index: usize) -> AllocationType {
        unsafe {
            let bitmap = self.get_type_bitmap();
            let byte_index = bit_index / BITS_PER_BYTE;
            let bit_offset = bit_index % BITS_PER_BYTE;
            if byte_index >= bitmap.len() {
                return AllocationType::Supervisor;
            }
            if (bitmap[byte_index] & (1 << bit_offset)) != 0 {
                AllocationType::User
            } else {
                AllocationType::Supervisor
            }
        }
    }

    /// Sets a bit as allocated with the given type.
    unsafe fn set_bit_allocated(&self, bit_index: usize, alloc_type: AllocationType) {
        unsafe {
            let alloc_bitmap = self.get_alloc_bitmap_mut();
            let type_bitmap = self.get_type_bitmap_mut();
            let byte_index = bit_index / BITS_PER_BYTE;
            let bit_offset = bit_index % BITS_PER_BYTE;

            if byte_index < alloc_bitmap.len() {
                alloc_bitmap[byte_index] |= 1 << bit_offset;
                match alloc_type {
                    AllocationType::User => {
                        type_bitmap[byte_index] |= 1 << bit_offset;
                    }
                    AllocationType::Supervisor => {
                        type_bitmap[byte_index] &= !(1 << bit_offset);
                    }
                }
            }
        }
    }

    /// Clears a bit (marks as free).
    unsafe fn set_bit_free(&self, bit_index: usize) {
        unsafe {
            let alloc_bitmap = self.get_alloc_bitmap_mut();
            let type_bitmap = self.get_type_bitmap_mut();
            let byte_index = bit_index / BITS_PER_BYTE;
            let bit_offset = bit_index % BITS_PER_BYTE;

            if byte_index < alloc_bitmap.len() {
                alloc_bitmap[byte_index] &= !(1 << bit_offset);
                type_bitmap[byte_index] &= !(1 << bit_offset);
            }
        }
    }

    /// Finds which region contains an address and returns (region_index, page_index_in_region).
    unsafe fn find_region_for_address(&self, addr: u64) -> Option<(usize, usize)> {
        unsafe {
            let regions = self.get_regions();
            for (i, region) in regions.iter().enumerate() {
                let region_end = region.base + (region.total_pages as u64 * PAGE_SIZE as u64);
                if addr >= region.base && addr < region_end {
                    let page_in_region = ((addr - region.base) / PAGE_SIZE as u64) as usize;
                    return Some((i, page_in_region));
                }
            }
            None
        }
    }

    /// Converts a region index and page-in-region to a global bit index.
    unsafe fn region_page_to_bit(&self, region_index: usize, page_in_region: usize) -> usize {
        unsafe {
            let regions = self.get_regions();
            if region_index < regions.len() {
                regions[region_index].bitmap_start_bit + page_in_region
            } else {
                0
            }
        }
    }

    /// Initializes the page allocator from the HOB list.
    ///
    /// This function:
    /// 1. Scans HOBs to count regions and total pages
    /// 2. Finds the first non-allocated region for bookkeeping
    /// 3. Reserves pages for bookkeeping structures
    /// 4. Initializes the bitmaps
    ///
    /// # Safety
    ///
    /// The caller must ensure that `hob_list` points to a valid HOB list.
    pub unsafe fn init_from_hob_list(&self, hob_list: *const c_void) -> Result<(), PageAllocError> {
        if hob_list.is_null() {
            return Err(PageAllocError::NotInitialized);
        }

        let _guard = self.lock.lock();

        // Get the HOB list iterator
        let hob_list_info = unsafe {
            (hob_list as *const PhaseHandoffInformationTable)
                .as_ref()
                .ok_or(PageAllocError::NotInitialized)?
        };

        let hob = Hob::Handoff(hob_list_info);

        // First pass: count regions and total pages
        let mut region_count = 0usize;
        let mut total_pages = 0usize;
        let mut first_free_region_base: Option<u64> = None;
        let mut first_free_region_size: u64 = 0;

        // Temporary storage for region info (we'll copy to SMRAM later)
        // Using a reasonable stack limit - actual regions stored in SMRAM
        const MAX_TEMP_REGIONS: usize = 256;
        let mut temp_regions: [(u64, u64, bool); MAX_TEMP_REGIONS] = [(0, 0, false); MAX_TEMP_REGIONS];

        for current_hob in &hob {
            if let Hob::GuidHob(guid_hob, data) = current_hob {
                if guid_hob.name == SMM_SMRAM_MEMORY_GUID
                    || guid_hob.name == MM_PEI_MMRAM_MEMORY_RESERVE_GUID
                {
                    log::info!("Found SMRAM memory HOB with GUID {:?}", guid_hob.name);

                    if data.len() < size_of::<SmramReserveHobData>() {
                        continue;
                    }

                    let reserve_data = unsafe { &*(data.as_ptr() as *const SmramReserveHobData) };
                    let descriptor_count = reserve_data.number_of_smram_regions as usize;

                    let descriptors_ptr = unsafe {
                        data.as_ptr().add(size_of::<SmramReserveHobData>()) as *const SmramDescriptor
                    };

                    for i in 0..descriptor_count {
                        if region_count >= MAX_TEMP_REGIONS {
                            log::warn!("Too many SMRAM regions for temp storage, increase MAX_TEMP_REGIONS");
                            break;
                        }

                        let descriptor = unsafe { &*descriptors_ptr.add(i) };
                        let pre_allocated = (descriptor.region_state & EFI_ALLOCATED) != 0;
                        let pages = (descriptor.physical_size as usize) / PAGE_SIZE;

                        log::info!(
                            "SMRAM Region {}: base=0x{:016x}, size=0x{:x}, pages={}, state=0x{:x}, allocated={}",
                            region_count,
                            descriptor.physical_start,
                            descriptor.physical_size,
                            pages,
                            descriptor.region_state,
                            pre_allocated
                        );

                        temp_regions[region_count] = (
                            descriptor.physical_start,
                            descriptor.physical_size,
                            pre_allocated,
                        );

                        // Track first non-allocated region for bookkeeping
                        if first_free_region_base.is_none() && !pre_allocated {
                            first_free_region_base = Some(descriptor.physical_start);
                            first_free_region_size = descriptor.physical_size;
                        }

                        total_pages += pages;
                        region_count += 1;
                    }
                }
            }
        }

        if region_count == 0 {
            log::error!("No SMRAM regions found in HOB list");
            return Err(PageAllocError::NotInitialized);
        }

        // Calculate bookkeeping space needed
        let bookkeeping_pages = Self::calculate_bookkeeping_pages(region_count, total_pages);

        log::info!(
            "Allocator needs {} pages for bookkeeping ({} regions, {} total pages)",
            bookkeeping_pages,
            region_count,
            total_pages
        );

        // Find space for bookkeeping
        let bookkeeping_base = first_free_region_base.ok_or_else(|| {
            log::error!("No free SMRAM region available for bookkeeping");
            PageAllocError::OutOfMemory
        })?;

        if (bookkeeping_pages * PAGE_SIZE) as u64 > first_free_region_size {
            log::error!("First free region too small for bookkeeping");
            return Err(PageAllocError::OutOfMemory);
        }

        log::info!(
            "Using 0x{:016x} for bookkeeping ({} pages)",
            bookkeeping_base,
            bookkeeping_pages
        );

        // Initialize the state structure in SMRAM
        let state_ptr = bookkeeping_base as *mut AllocatorState;
        unsafe {
            // Zero the bookkeeping pages first
            ptr::write_bytes(bookkeeping_base as *mut u8, 0, bookkeeping_pages * PAGE_SIZE);

            // Write the header
            (*state_ptr).region_count = region_count;
            (*state_ptr).total_pages = total_pages;
            (*state_ptr).bookkeeping_pages = bookkeeping_pages;
            (*state_ptr).bookkeeping_base = bookkeeping_base;

            // Store state pointer
            *self.state.get() = state_ptr;
        }

        // Initialize region info
        let mut bitmap_start_bit = 0usize;
        {
            let regions = unsafe { self.get_regions_mut() };
            for (i, region) in regions.iter_mut().enumerate() {
                let (base, size, _) = temp_regions[i];
                let pages = (size as usize) / PAGE_SIZE;
                region.base = base;
                region.total_pages = pages;
                region.bitmap_start_bit = bitmap_start_bit;
                bitmap_start_bit += pages;
            }
        }

        // Mark pre-allocated regions and bookkeeping pages as allocated
        for i in 0..region_count {
            let (base, size, pre_allocated) = temp_regions[i];
            let pages = (size as usize) / PAGE_SIZE;

            if pre_allocated {
                // Mark entire region as allocated (supervisor)
                let regions = unsafe { self.get_regions() };
                let start_bit = regions[i].bitmap_start_bit;
                for p in 0..pages {
                    unsafe { self.set_bit_allocated(start_bit + p, AllocationType::Supervisor) };
                }
            } else if base == bookkeeping_base {
                // Mark bookkeeping pages as allocated (supervisor)
                let regions = unsafe { self.get_regions() };
                let start_bit = regions[i].bitmap_start_bit;
                for p in 0..bookkeeping_pages {
                    unsafe { self.set_bit_allocated(start_bit + p, AllocationType::Supervisor) };
                }
            }
        }

        self.initialized.store(true, Ordering::Release);

        log::info!(
            "Page allocator initialized: {} region(s), {} total pages, {} free pages",
            region_count,
            total_pages,
            self.free_page_count()
        );

        Ok(())
    }

    /// Allocates contiguous pages from SMRAM for supervisor use.
    pub fn allocate_pages(&self, num_pages: usize) -> Result<u64, PageAllocError> {
        self.allocate_pages_with_type(num_pages, AllocationType::Supervisor)
    }

    /// Allocates contiguous pages from SMRAM with the specified allocation type.
    ///
    /// For `Supervisor` allocations, the allocated region is marked as supervisor-owned
    /// data pages (R/W, non-executable) in the page table.
    pub fn allocate_pages_with_type(
        &self,
        num_pages: usize,
        alloc_type: AllocationType,
    ) -> Result<u64, PageAllocError> {
        if !self.initialized.load(Ordering::Acquire) {
            return Err(PageAllocError::NotInitialized);
        }

        if num_pages == 0 {
            return Err(PageAllocError::OutOfMemory);
        }

        let allocated_addr = {
            let _guard = self.lock.lock();

            // SAFETY: We have exclusive access via the lock
            unsafe {
                let regions = self.get_regions();

                let mut found: Option<u64> = None;
                // Try each region
                'outer: for region in regions.iter() {
                    // First-fit search for contiguous pages
                    let mut run_start = 0usize;
                    let mut run_length = 0usize;

                    for page_in_region in 0..region.total_pages {
                        let bit_index = region.bitmap_start_bit + page_in_region;
                        if self.is_bit_allocated(bit_index) {
                            run_start = page_in_region + 1;
                            run_length = 0;
                        } else {
                            run_length += 1;
                            if run_length == num_pages {
                                // Found a suitable run, allocate it
                                for p in run_start..run_start + num_pages {
                                    let bit = region.bitmap_start_bit + p;
                                    self.set_bit_allocated(bit, alloc_type);
                                }
                                let addr = region.base + (run_start as u64 * PAGE_SIZE as u64);
                                log::trace!(
                                    "Allocated {} {:?} page(s) at 0x{:016x}",
                                    num_pages,
                                    alloc_type,
                                    addr
                                );
                                found = Some(addr);
                                break 'outer;
                            }
                        }
                    }
                }

                found
            }
        }; // lock is dropped here

        let addr = allocated_addr.ok_or(PageAllocError::OutOfMemory)?;

        // For supervisor allocations, update page table attributes to mark as
        // supervisor-owned data pages (R/W/NX/S), otherwise they would
        // default to user data (R/W/NX/U).
        self.apply_data_page_attributes(addr, num_pages, alloc_type);

        Ok(addr)
    }

    /// Applies supervisor page table attributes to a newly allocated region.
    ///
    /// Marks pages as supervisor-owned data pages: Read/Write + Non-Executable (NX).
    /// This ensures supervisor data cannot be executed, providing W^X enforcement.
    ///
    /// If the global page table is not yet initialized (e.g., during early boot),
    /// this is a no-op with a warning.
    fn apply_data_page_attributes(&self, addr: u64, num_pages: usize, _alloc_type: AllocationType) {

        let size = (num_pages * PAGE_SIZE) as u64;
        let mut pt_guard = crate::PAGE_TABLE.lock();
        if let Some(ref mut pt) = *pt_guard {
            // Data pages: R/W (no ReadOnly) + NX (ExecuteProtect)
            let mut attributes = MemoryAttributes::ExecuteProtect;
            
            if _alloc_type == AllocationType::Supervisor {
                // For Supervisor allocations, we additionally want the U/S bit cleared (Supervisor-only).
                attributes = attributes | MemoryAttributes::Special; // Ensure not writable by user code
            }

            if let Err(e) = pt.map_memory_region(addr, size, attributes) {
                log::error!(
                    "Failed to set supervisor page attributes for 0x{:016x} ({} pages): {:?}",
                    addr,
                    num_pages,
                    e
                );
            } else {
                log::trace!(
                    "Marked 0x{:016x} ({} pages) as supervisor R/W+NX",
                    addr,
                    num_pages,
                );
            }
        } else {
            log::warn!(
                "Page table not initialized, skipping attribute update for 0x{:016x}",
                addr
            );
        }
    }

    /// Applies restrictive page table attributes to freed pages.
    ///
    /// Marks pages as completely inaccessible: Supervisor + ReadProtect + ExecuteProtect (NX).
    /// This prevents any read, write, or execute access to freed memory, mitigating
    /// use-after-free vulnerabilities.
    ///
    /// If the global page table is not yet initialized (e.g., during early boot),
    /// this is a no-op with a warning.
    fn apply_freed_page_attributes(&self, addr: u64, num_pages: usize) {

        let size = (num_pages * PAGE_SIZE) as u64;
        let mut pt_guard = crate::PAGE_TABLE.lock();
        if let Some(ref mut pt) = *pt_guard {
            // Freed pages: ReadProtect (not present) + NX (no execute) + ReadOnly (no write)
            // This makes the pages completely inaccessible.
            if let Err(e) = pt.unmap_memory_region(addr, size) {
                log::error!(
                    "Failed to set freed page attributes for 0x{:016x} ({} pages): {:?}",
                    addr,
                    num_pages,
                    e
                );
            } else {
                log::trace!(
                    "Marked 0x{:016x} ({} pages) as inaccessible (RP+NX+RO+S)",
                    addr,
                    num_pages,
                );
            }
        } else {
            log::warn!(
                "Page table not initialized, skipping freed page attribute update for 0x{:016x}",
                addr
            );
        }
    }

    /// Frees previously allocated pages.
    ///
    /// After freeing, the pages are marked as inaccessible in the page table
    /// (Supervisor + ReadProtect + ExecuteProtect) to prevent use-after-free.
    pub fn free_pages(&self, addr: u64, num_pages: usize) -> Result<(), PageAllocError> {
        if !self.initialized.load(Ordering::Acquire) {
            return Err(PageAllocError::NotInitialized);
        }

        if addr % PAGE_SIZE as u64 != 0 {
            return Err(PageAllocError::NotAligned);
        }

        {
            let _guard = self.lock.lock();

            unsafe {
                let (region_index, page_in_region) = self
                    .find_region_for_address(addr)
                    .ok_or(PageAllocError::InvalidAddress)?;

                let regions = self.get_regions();
                let region = &regions[region_index];

                // Verify all pages are allocated
                for p in 0..num_pages {
                    let bit = region.bitmap_start_bit + page_in_region + p;
                    if !self.is_bit_allocated(bit) {
                        return Err(PageAllocError::NotAllocated);
                    }
                }

                // Free the pages
                for p in 0..num_pages {
                    let bit = region.bitmap_start_bit + page_in_region + p;
                    self.set_bit_free(bit);
                }

                log::trace!("Freed {} page(s) at 0x{:016x}", num_pages, addr);
            }
        } // lock is dropped here

        // Mark freed pages as inaccessible in the page table.
        self.apply_freed_page_attributes(addr, num_pages);

        Ok(())
    }

    /// Frees previously allocated pages, verifying the allocation type matches.
    ///
    /// After freeing, the pages are marked as inaccessible in the page table
    /// (Supervisor + ReadProtect + ExecuteProtect) to prevent use-after-free.
    pub fn free_pages_checked(
        &self,
        addr: u64,
        num_pages: usize,
        expected_type: AllocationType,
    ) -> Result<(), PageAllocError> {
        if !self.initialized.load(Ordering::Acquire) {
            return Err(PageAllocError::NotInitialized);
        }

        if addr % PAGE_SIZE as u64 != 0 {
            return Err(PageAllocError::NotAligned);
        }

        {
            let _guard = self.lock.lock();

            unsafe {
                let (region_index, page_in_region) = self
                    .find_region_for_address(addr)
                    .ok_or(PageAllocError::InvalidAddress)?;

                let regions = self.get_regions();
                let region = &regions[region_index];

                // Verify all pages are allocated with expected type
                for p in 0..num_pages {
                    let bit = region.bitmap_start_bit + page_in_region + p;
                    if !self.is_bit_allocated(bit) {
                        return Err(PageAllocError::NotAllocated);
                    }
                    if self.get_bit_type(bit) != expected_type {
                        log::warn!(
                            "Type mismatch at 0x{:016x}: expected {:?}, got {:?}",
                            addr + (p as u64 * PAGE_SIZE as u64),
                            expected_type,
                            self.get_bit_type(bit)
                        );
                        return Err(PageAllocError::InvalidAddress);
                    }
                }

                // Free the pages
                for p in 0..num_pages {
                    let bit = region.bitmap_start_bit + page_in_region + p;
                    self.set_bit_free(bit);
                }

                log::trace!("Freed {} {:?} page(s) at 0x{:016x}", num_pages, expected_type, addr);
            }
        } // lock is dropped here

        // Mark freed pages as inaccessible in the page table.
        self.apply_freed_page_attributes(addr, num_pages);

        Ok(())
    }

    /// Returns the total number of free pages across all regions.
    pub fn free_page_count(&self) -> usize {
        if !self.initialized.load(Ordering::Acquire) {
            return 0;
        }

        unsafe {
            let state = *self.state.get();
            if state.is_null() {
                return 0;
            }
            let total_pages = (*state).total_pages;
            let mut free = 0;
            for bit in 0..total_pages {
                if !self.is_bit_allocated(bit) {
                    free += 1;
                }
            }
            free
        }
    }

    /// Returns the number of pages allocated for a specific type.
    pub fn allocated_page_count(&self, alloc_type: AllocationType) -> usize {
        if !self.initialized.load(Ordering::Acquire) {
            return 0;
        }

        let _guard = self.lock.lock();

        unsafe {
            let state = *self.state.get();
            if state.is_null() {
                return 0;
            }
            let total_pages = (*state).total_pages;
            let mut count = 0;
            for bit in 0..total_pages {
                if self.is_bit_allocated(bit) && self.get_bit_type(bit) == alloc_type {
                    count += 1;
                }
            }
            count
        }
    }

    /// Returns the allocation type for a given address.
    pub fn get_allocation_type(&self, addr: u64) -> Option<AllocationType> {
        if !self.initialized.load(Ordering::Acquire) {
            return None;
        }

        let _guard = self.lock.lock();

        unsafe {
            let (region_index, page_in_region) = self.find_region_for_address(addr)?;
            let bit = self.region_page_to_bit(region_index, page_in_region);
            if self.is_bit_allocated(bit) {
                Some(self.get_bit_type(bit))
            } else {
                None
            }
        }
    }

    /// Returns whether the allocator has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    /// Returns the total number of pages across all regions.
    pub fn total_page_count(&self) -> usize {
        if !self.initialized.load(Ordering::Acquire) {
            return 0;
        }
        unsafe {
            let state = *self.state.get();
            if state.is_null() {
                0
            } else {
                (*state).total_pages
            }
        }
    }

    /// Returns the number of regions.
    pub fn region_count(&self) -> usize {
        if !self.initialized.load(Ordering::Acquire) {
            return 0;
        }
        unsafe {
            let state = *self.state.get();
            if state.is_null() {
                0
            } else {
                (*state).region_count
            }
        }
    }

    pub fn is_region_inside_mmram(&self, addr: u64, size: u64) -> bool {
        if !self.initialized.load(Ordering::Acquire) {
            return false;
        }

        let _guard = self.lock.lock();

        unsafe {
            let regions = self.get_regions();
            for region in regions.iter() {
                let region_end = region.base + (region.total_pages as u64 * PAGE_SIZE as u64);
                if addr >= region.base && (addr + size) <= region_end {
                    return true;
                }
            }
            false
        }
    }
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
// Global Pool Allocator
// ============================================================================

/// Pool allocator built on top of the page allocator.
///
/// This allocator provides smaller-granularity allocations by allocating
/// pages from the page allocator and subdividing them for pool allocations.
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
pub struct PoolAllocator {
    /// Reference to the underlying page allocator.
    page_allocator: &'static PageAllocator,
    /// Pool blocks for tracking allocated pages.
    blocks: UnsafeCell<[PoolBlock; MAX_POOL_BLOCKS]>,
    /// Number of valid blocks.
    block_count: AtomicUsize,
    /// Lock for thread safety.
    lock: Mutex<()>,
}

// SAFETY: The PoolAllocator uses internal locking for thread safety.
unsafe impl Send for PoolAllocator {}
unsafe impl Sync for PoolAllocator {}

impl PoolAllocator {
    /// Creates a new pool allocator using the given page allocator.
    pub const fn new(page_allocator: &'static PageAllocator) -> Self {
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

        // Allocate pages from the page allocator
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

unsafe impl GlobalAlloc for PoolAllocator {
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
            // (could be done less frequently for performance)
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

// ============================================================================
// Global Allocator Instance
// ============================================================================

/// Global page allocator instance.
///
/// This must be initialized via `init_from_hob_list` before use.
pub static PAGE_ALLOCATOR: PageAllocator = PageAllocator::new();

/// Global pool allocator instance.
///
/// This uses the global `PAGE_ALLOCATOR` and can be set as the `#[global_allocator]`.
#[global_allocator]
static GLOBAL_ALLOCATOR: PoolAllocator = PoolAllocator::new(&PAGE_ALLOCATOR);
