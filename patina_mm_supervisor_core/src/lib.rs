//! MM Supervisor Core
//!
//! A pure Rust implementation of the MM Supervisor Core for standalone MM mode environments.
//!
//! This crate provides the core functionality for running a supervisor in MM (Management Mode)
//! that orchestrates incoming requests on the BSP while APs wait in a holding pen.
//!
//! ## Architecture
//!
//! The entry point is executed on all cores:
//! - **BSP**: Performs one-time initialization and enters the request serving loop
//! - **APs**: Enter a holding pen and poll mailboxes for commands from BSP
//!
//! ## Memory Model
//!
//! This is a core component that manages its own memory. It does **not** use heap allocation.
//! All structures use fixed-size arrays with compile-time constants provided via const generics.
//!
//! ## Example
//!
//! ```rust,ignore
//! use patina_mm_supervisor_core::*;
//!
//! struct MyPlatform;
//!
//! impl PlatformInfo for MyPlatform {
//!     type CpuInfo = Self;
//!     const MAX_CPU_COUNT: usize = 8;
//!     const MAX_HANDLERS: usize = 32;
//! }
//!
//! impl CpuInfo for MyPlatform {
//!     fn ap_poll_timeout_us() -> u64 { 1000 }
//! }
//!
//! static SUPERVISOR: MmSupervisorCore<MyPlatform> = MmSupervisorCore::new();
//! ```
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!
#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]
#![cfg(target_arch = "x86_64")]
#![feature(coverage_attribute)]
#![feature(generic_const_exprs)]

mod cpu;
mod mailbox;
pub mod mm_mem;
pub mod paging_allocator;
mod request_handler;
pub mod unblock_memory;

pub use cpu::{ApState, CpuInfo, CpuManager};
pub use mailbox::{ApCommand, ApMailbox, ApResponse, MailboxManager};
pub use mm_mem::{
    AllocationType, PageAllocator, PoolAllocator, PageAllocError, SmramDescriptor,
    PAGE_SIZE, PAGE_ALLOCATOR,
    SMM_SMRAM_MEMORY_GUID, MM_PEI_MMRAM_MEMORY_RESERVE_GUID,
};
pub use paging_allocator::{
    PagingPoolAllocator, PagingAllocError, SharedPagingAllocator,
    PAGING_ALLOCATOR, DEFAULT_PAGING_POOL_PAGES,
};
pub use request_handler::{RequestContext, RequestHandler, RequestResult, RequestDispatcher};
pub use unblock_memory::{
    UnblockedMemoryTracker, UnblockedMemoryEntry, UnblockError,
    UNBLOCKED_MEMORY_TRACKER,
};

use core::{
    arch::{global_asm, asm},
    ffi::c_void,
    num::NonZeroUsize,
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicU32, Ordering},
};

use patina::pi::hob::{Hob, PhaseHandoffInformationTable};
use patina_paging::{PagingType, x64::X64PageTable};
use r_efi::efi;

use patina_mm_policy::{walk_page_table, MemDescriptorV1_0};

// GUID for gMmSupervisorHobMemoryAllocModuleGuid
// { 0x3efafe72, 0x3dbf, 0x4341, { 0xad, 0x04, 0x1c, 0xb6, 0xe8, 0xb6, 0x8e, 0x5e }}
/// GUID used in MemoryAllocationModule HOBs to identify MM Supervisor module allocations.
pub const MM_SUPERVISOR_HOB_MEMORY_ALLOC_MODULE_GUID: efi::Guid = efi::Guid::from_fields(
    0x3efafe72,
    0x3dbf,
    0x4341,
    0xad,
    0x04,
    &[0x1c, 0xb6, 0xe8, 0xb6, 0x8e, 0x5e],
);

// GUID for gMmSupervisorUserGuid
// { 0x30d1cc3f, 0xc1db, 0x41ed, { 0xb1, 0x13, 0xab, 0xce, 0x21, 0xb0, 0x2b, 0xce }}
/// GUID identifying the MM Supervisor User module.
pub const MM_SUPERVISOR_USER_GUID: efi::Guid = efi::Guid::from_fields(
    0x30d1cc3f,
    0xc1db,
    0x41ed,
    0xb1,
    0x13,
    &[0xab, 0xce, 0x21, 0xb0, 0x2b, 0xce],
);

// GUID for gMmSupervisorPassDownHobGuid
// { 0x3f2d2d1a, 0x7c6a, 0x4e2e, { 0x91, 0x2e, 0x5c, 0x4f, 0x5b, 0x8c, 0x2a, 0x9d } }
/// GUID for the MM Supervisor PassDown HOB.
pub const MM_SUPV_PASS_DOWN_HOB_GUID: efi::Guid = efi::Guid::from_fields(
    0x3f2d2d1a,
    0x7c6a,
    0x4e2e,
    0x91,
    0x2e,
    &[0x5c, 0x4f, 0x5b, 0x8c, 0x2a, 0x9d],
);

/// MM Supervisor PassDown HOB Revision
pub const MM_SUPV_PASS_DOWN_HOB_REVISION: u32 = 1;

/// MM Supervisor PassDown HOB Data Structure
///
/// This structure contains various buffer pointers and sizes passed from
/// the PEI phase to the MM Supervisor.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct MmSupvPassDownHobData {
    /// Revision of this HOB structure
    pub revision: u32,
    /// Reserved for future use
    pub reserved: u32,
    /// Base address of CPL3 stack for MM Supervisor
    pub mm_supervisor_cpl3_stack_base: u64,
    /// Per-CPU stack size for CPL3
    pub mm_supervisor_cpl3_per_core_stack_size: u64,
    /// MM Supervisor CPU private data base address
    pub mm_supv_cpu_private: u64,
    /// Size of MM Supervisor CPU private data
    pub mm_supv_cpu_private_size: u64,
    /// MM Supervisor MP sync data base address
    pub mm_supv_mp_sync_data: u64,
    /// Size of MM Supervisor MP sync data
    pub mm_supv_mp_sync_data_size: u64,
    /// MM Supervisor communication buffer base address
    pub mm_supv_comm_buffer: u64,
    /// MM Supervisor internal communication buffer base address
    pub mm_supv_comm_buffer_internal: u64,
    /// Size of MM Supervisor communication buffer
    pub mm_supv_comm_buffer_size: u64,
    /// MM User communication buffer base address
    pub mm_user_comm_buffer: u64,
    /// MM User internal communication buffer base address
    pub mm_user_comm_buffer_internal: u64,
    /// Size of MM User communication buffer
    pub mm_user_comm_buffer_size: u64,
    /// MM Supervisor status buffer base address
    pub mm_supv_status_buffer: u64,
    /// MM Supervisor to User buffer base address
    pub mm_supv_to_user_buffer: u64,
    /// Size of MM Supervisor to User buffer
    pub mm_supv_to_user_buffer_size: u64,
    /// MM Supervisor GDT buffer base address
    pub mm_supv_gdt_buffer: u64,
    /// Size of MM Supervisor GDT buffer
    pub mm_supv_gdt_buffer_size: u64,
    /// Step size of MM Supervisor GDT buffer per CPU
    pub mm_supv_gdt_step_size: u64,
    /// MM Initialized buffer base address
    pub mm_initialized_buffer: u64,
    /// MM Supervisor firmware policy buffer base address
    pub mm_supv_firmware_policy_buffer: u64,
    /// Size of MM Supervisor firmware policy buffer
    pub mm_supv_firmware_policy_buffer_size: u64,
    /// MM Supervisor memory policy buffer base address
    pub mm_supv_memory_policy_buffer: u64,
    /// Size of MM Supervisor memory policy buffer
    pub mm_supv_memory_policy_buffer_size: u64,
}

/// Errors that can occur during policy initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyInitError {
    /// The HOB list pointer is null.
    NullHobList,
    /// PassDown HOB not found.
    PassDownHobNotFound,
    /// Invalid PassDown HOB revision.
    InvalidRevision { found: u32, expected: u32 },
    /// Firmware policy buffer is null or empty.
    NullFirmwarePolicyBuffer,
    /// Invalid policy data.
    InvalidPolicyData,
}

use spin::Once;
use patina_internal_cpu::interrupts::Interrupts;

global_asm!(include_str!("entry_point.asm"));

/// A trait to be implemented by the platform to provide configuration values and types to be used
/// by the MM Supervisor Core.
///
/// ## Example
///
/// ```rust,ignore
/// use patina_mm_supervisor_core::*;
///
/// struct ExamplePlatform;
///
/// impl CpuInfo for ExamplePlatform {
///     fn ap_poll_timeout_us() -> u64 { 1000 }
/// }
///
/// impl PlatformInfo for ExamplePlatform {
///     type CpuInfo = Self;
///     const MAX_CPU_COUNT: usize = 8;
///     const MAX_HANDLERS: usize = 32;
/// }
/// ```
#[cfg_attr(test, mockall::automock(type CpuInfo = MockCpuInfo;))]
pub trait PlatformInfo: 'static {
    /// The platform's CPU information and configuration.
    type CpuInfo: CpuInfo;

    /// Maximum number of CPUs supported by the platform.
    /// This is used to size the CPU manager and mailbox arrays.
    const MAX_CPU_COUNT: usize;

    /// Maximum number of request handlers that can be registered.
    const MAX_HANDLERS: usize;
}

/// Static reference to the MM Supervisor Core instance.
///
/// This is set during the `entry_point` call and provides global access to the supervisor.
static __SUPERVISOR: Once<NonZeroUsize> = Once::new();

/// Counter for tracking CPU arrivals at the entry point.
static CPU_ARRIVAL_COUNT: AtomicU32 = AtomicU32::new(0);

/// Flag indicating that initialization is complete.
static BSP_INIT_COMPLETE: AtomicBool = AtomicBool::new(false);

/// The policy object is initialized once during BSP initialization and provides access to the security policy
/// for the MM Supervisor. It is stored in a static variable for global access.
/// The policy gate is initialized from the firmware policy buffer provided in the PassDown HOB.
static POLICY_GATE: Once<patina_mm_policy::PolicyGate> = Once::new();

/// The MM Supervisor Core responsible for managing the standalone MM environment.
///
/// This struct is generic over the [`PlatformInfo`] trait, which provides platform-specific
/// configuration including compile-time constants for array sizes.
///
/// The supervisor manages:
/// - BSP initialization and request handling
/// - AP management through the holding pen and mailbox system
/// - Request dispatching and response handling
///
/// ## Memory Model
///
/// This struct does not perform heap allocation. All internal structures use fixed-size
/// arrays based on the `MAX_CPU_COUNT` and `MAX_HANDLERS` constants from [`PlatformInfo`].
///
/// ## Usage
///
/// Create a static instance of the supervisor and call `entry_point` from all cores:
///
/// ```rust,ignore
/// use patina_mm_supervisor_core::*;
///
/// static SUPERVISOR: MmSupervisorCore<MyPlatform> = MmSupervisorCore::new();
///
/// #[no_mangle]
/// pub extern "efiapi" fn mm_entry(hob_list: *const c_void) -> ! {
///     SUPERVISOR.entry_point(hob_list)
/// }
/// ```
pub struct MmSupervisorCore<P: PlatformInfo>
where
    [(); P::MAX_CPU_COUNT]:,
    [(); P::MAX_HANDLERS]:,
{
    /// Manager for CPU-related operations.
    cpu_manager: CpuManager<{ P::MAX_CPU_COUNT }>,
    /// Manager for AP mailboxes.
    mailbox_manager: MailboxManager<{ P::MAX_CPU_COUNT }>,
    /// Request dispatcher for handling incoming requests.
    request_dispatcher: RequestDispatcher<{ P::MAX_HANDLERS }>,
    /// Flag indicating if the core has been initialized.
    initialized: AtomicBool,
    /// Phantom data for the platform type.
    _phantom: core::marker::PhantomData<P>,
}

// SAFETY: The MmSupervisorCore is designed to be shared across threads with proper synchronization.
unsafe impl<P: PlatformInfo> Send for MmSupervisorCore<P>
where
    [(); P::MAX_CPU_COUNT]:,
    [(); P::MAX_HANDLERS]:,
{
}
unsafe impl<P: PlatformInfo> Sync for MmSupervisorCore<P>
where
    [(); P::MAX_CPU_COUNT]:,
    [(); P::MAX_HANDLERS]:,
{
}

fn is_buffer_inside_mmram(base: u64, size: u64) -> bool {
    // we will go over the page allocator to see if this region falls inside any of the MMRAM regions
    mm_mem::PAGE_ALLOCATOR.is_region_inside_mmram(base, size)
}

/// Read CR3 register.
fn read_cr3() -> u64 {
    let mut _value = 0u64;

    #[cfg(all(not(test), target_arch = "x86_64"))]
    {
        // SAFETY: inline asm is inherently unsafe because Rust can't reason about it.
        // In this case we are reading the CR3 register, which is a safe operation.
        unsafe {
            asm!("mov {}, cr3", out(reg) _value, options(nostack, preserves_flags));
        }
    }

    _value
}

#[coverage(off)]
impl<P: PlatformInfo> MmSupervisorCore<P>
where
    [(); P::MAX_CPU_COUNT]:,
    [(); P::MAX_HANDLERS]:,
{
    /// Creates a new instance of the MM Supervisor Core.
    ///
    /// This is a const fn that performs no heap allocation.
    pub const fn new() -> Self {
        Self {
            cpu_manager: CpuManager::new(),
            mailbox_manager: MailboxManager::new(),
            request_dispatcher: RequestDispatcher::new(),
            initialized: AtomicBool::new(false),
            _phantom: core::marker::PhantomData,
        }
    }

    /// Sets the static supervisor instance for global access.
    ///
    /// Returns true if the address was successfully stored, false if already set.
    #[must_use]
    fn set_instance(&'static self) -> bool {
        let physical_address = NonNull::from_ref(self).expose_provenance();
        &physical_address == __SUPERVISOR.call_once(|| physical_address)
    }

    /// Gets the static MM Supervisor Core instance for global access.
    #[allow(unused)]
    pub(crate) fn instance<'a>() -> &'a Self {
        // SAFETY: The pointer is guaranteed to be valid as set_instance ensures single initialization.
        unsafe {
            NonNull::<Self>::with_exposed_provenance(
                *__SUPERVISOR.get().expect("MM Supervisor Core is not initialized."),
            )
            .as_ref()
        }
    }

    /// The entry point for the MM Supervisor Core.
    ///
    /// This function is called on all cores (BSP and APs). The BSP performs initialization
    /// and enters the request serving loop, while APs enter the holding pen.
    ///
    /// # Arguments
    ///
    /// * `hob_list` - Pointer to the HOB (Hand-Off Block) list passed from the pre-MM phase.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The supervisor instance was already set
    /// - The HOB list pointer is null
    pub fn entry_point(&'static self, cpu_index: usize, hob_list: *const c_void) -> ! {
        // Get the current CPU's APIC ID to determine if we're BSP or AP
        let cpu_id = cpu::get_current_cpu_id();

        // Track CPU arrival
        let arrival_order = CPU_ARRIVAL_COUNT.fetch_add(1, Ordering::SeqCst);

        // The first CPU to arrive is considered the BSP
        let is_bsp = arrival_order == 0;

        if is_bsp {
            // BSP path: Initialize the supervisor
            assert!(self.set_instance(), "MM Supervisor Core instance was already set!");
            assert!(!hob_list.is_null(), "MM Supervisor Core requires a non-null HOB list pointer.");

            log::info!("MM Supervisor Core v{}", env!("CARGO_PKG_VERSION"));
            log::info!("BSP (CPU {}) starting initialization...", cpu_id);

            // Register BSP with CPU manager
            self.cpu_manager.register_cpu(cpu_id, true);

            // Perform platform-specific initialization
            self.bsp_init(hob_list);

            // Mark as initialized
            self.initialized.store(true, Ordering::Release);

            // Signal that initialization is complete
            BSP_INIT_COMPLETE.store(true, Ordering::Release);

            log::info!("BSP initialization complete, entering request serving loop...");

            // Enter the main request serving loop
            self.bsp_request_loop()
        } else {
            // AP path: Wait for BSP to complete initialization, then enter holding pen
            log::trace!("AP (CPU {}) waiting for BSP initialization...", cpu_id);

            // Spin until BSP completes initialization
            while !BSP_INIT_COMPLETE.load(Ordering::Acquire) {
                core::hint::spin_loop();
            }

            // Register this AP with the CPU manager
            self.cpu_manager.register_cpu(cpu_id, false);

            log::trace!("AP (CPU {}) entering holding pen...", cpu_id);

            // Enter the holding pen
            self.ap_holding_pen(cpu_id)
        }
    }

    /// BSP-specific initialization.
    ///
    /// This is called only on the BSP after basic setup is complete.
    fn bsp_init(&'static self, hob_list: *const c_void) {
        log::info!("BSP performing one-time initialization...");

        let mut interrupt_manager = Interrupts::new();
        interrupt_manager.initialize().unwrap_or_else(|err| {
            panic!("Failed to initialize Interrupt Manager: {:?}", err);
        });

        // // For debugging: Dump the HOB list
        // // SAFETY: The HOB list pointer is provided by the MM IPL and is guaranteed to be valid at this point.
        // unsafe {
        //     mm_mem::dump_hob_list(hob_list);
        // }

        // Initialize the page allocator from the HOB list
        // This finds all SMRAM regions and sets up memory tracking
        // SAFETY: hob_list is provided by the MM IPL and is guaranteed to be valid
        unsafe {
            if let Err(e) = mm_mem::PAGE_ALLOCATOR.init_from_hob_list(hob_list) {
                log::error!("Failed to initialize page allocator: {:?}", e);
            }
        }

        // Reserve pages from the page allocator for paging structures.
        // This is done before paging is initialized to avoid circular dependency.
        unsafe {
            match mm_mem::PAGE_ALLOCATOR.allocate_pages(paging_allocator::DEFAULT_PAGING_POOL_PAGES) {
                Ok(paging_pool_base) => {
                    log::info!(
                        "Reserved {} pages at 0x{:016x} for paging structures",
                        paging_allocator::DEFAULT_PAGING_POOL_PAGES,
                        paging_pool_base
                    );
                    // Initialize the paging allocator with the reserved pool
                    if let Err(e) = paging_allocator::PAGING_ALLOCATOR.init(
                        paging_pool_base,
                        paging_allocator::DEFAULT_PAGING_POOL_PAGES,
                    ) {
                        log::error!("Failed to initialize paging allocator: {:?}", e);
                    }
                }
                Err(e) => {
                    log::error!("Failed to reserve pages for paging structures: {:?}", e);
                }
            }
        }

        let mut paging_alloc = paging_allocator::SharedPagingAllocator::new(&paging_allocator::PAGING_ALLOCATOR);
        let paging = X64PageTable::new(paging_alloc, PagingType::Paging4Level);

        // Discover the MM Supervisor User module entry point from the HOB list.
        // We look for EFI_HOB_TYPE_MEMORY_ALLOCATION HOBs that have:
        // - MemoryAllocationHeader.Name == gMmSupervisorHobMemoryAllocModuleGuid
        // - ModuleName == gMmSupervisorUserGuid
        // SAFETY: hob_list is provided by the MM IPL and is guaranteed to be valid
        let user_entry_point = unsafe { self.discover_user_module_entry(hob_list) };
        if let Some(entry) = user_entry_point {
            log::info!("Discovered MM User module entry point: 0x{:016x}", entry);
            // TODO: Store this entry point for later invocation
        } else {
            log::warn!("MM User module entry point not found in HOB list");
        }

        // Initialize the policy gate from the PassDown HOB.
        // This discovers the firmware policy buffer and initializes the policy gate.
        // SAFETY: hob_list is provided by the MM IPL and is guaranteed to be valid
        unsafe {
            if let Err(e) = self.init_policy_from_hob_list(hob_list) {
                log::error!("Failed to initialize policy gate: {:?}", e);
            }
        }

        // Allocate buffer for descriptors
        // let mut buffer = [MemDescriptorV1_0::default(); 1024];


        // TODO: Initialize request handler infrastructure

        log::trace!("BSP one-time initialization complete.");
    }

    /// Discovers the MM Supervisor User module entry point from the HOB list.
    ///
    /// This function iterates through the HOB list looking for `MemoryAllocationModule` HOBs
    /// that match the MM Supervisor memory allocation module GUID and have the MM Supervisor
    /// User GUID as their module name.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `hob_list` points to a valid HOB list.
    ///
    /// # Returns
    ///
    /// The entry point address of the user module if found, or `None` otherwise.
    unsafe fn discover_user_module_entry(&self, hob_list: *const c_void) -> Option<u64> {
        if hob_list.is_null() {
            return None;
        }

        // Get the HOB list header
        let hob_list_info = unsafe {
            (hob_list as *const PhaseHandoffInformationTable).as_ref()?
        };

        let hob = Hob::Handoff(hob_list_info);

        // Iterate through the HOB list looking for MemoryAllocationModule HOBs
        for current_hob in &hob {
            if let Hob::MemoryAllocationModule(mem_alloc_mod) = current_hob {
                // Check if this is an MM Supervisor module allocation
                // (MemoryAllocationHeader.Name == gMmSupervisorHobMemoryAllocModuleGuid)
                if mem_alloc_mod.alloc_descriptor.name == MM_SUPERVISOR_HOB_MEMORY_ALLOC_MODULE_GUID {
                    log::debug!(
                        "Found MM Supervisor module HOB: module_name={:?}, entry_point=0x{:016x}",
                        mem_alloc_mod.module_name,
                        mem_alloc_mod.entry_point
                    );

                    // Check if this is the User module (ModuleName == gMmSupervisorUserGuid)
                    if mem_alloc_mod.module_name == MM_SUPERVISOR_USER_GUID {
                        log::info!(
                            "Found MM User module: entry_point=0x{:016x}, base=0x{:016x}, size=0x{:x}",
                            mem_alloc_mod.entry_point,
                            mem_alloc_mod.alloc_descriptor.memory_base_address,
                            mem_alloc_mod.alloc_descriptor.memory_length
                        );
                        return Some(mem_alloc_mod.entry_point);
                    }
                }
            }
        }

        None
    }

    /// Initializes the policy gate from the PassDown HOB.
    ///
    /// This function iterates through the HOB list looking for the PassDown HOB,
    /// extracts the firmware policy buffer pointer, and initializes the policy gate.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `hob_list` points to a valid HOB list.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the policy gate was successfully initialized, or an error otherwise.
    /// TODO: Remove the passdown hob eventually!!!!!
    unsafe fn init_policy_from_hob_list(&self, hob_list: *const c_void) -> Result<(), PolicyInitError> {
        if hob_list.is_null() {
            return Err(PolicyInitError::NullHobList);
        }

        // Get the HOB list header
        let hob_list_info = unsafe {
            (hob_list as *const PhaseHandoffInformationTable)
                .as_ref()
                .ok_or(PolicyInitError::NullHobList)?
        };

        let hob = Hob::Handoff(hob_list_info);

        // Walk through HOBs to find the PassDown HOB
        for current_hob in &hob {
            if let Hob::GuidHob(guid_hob, data) = current_hob {
                if guid_hob.name == MM_SUPV_PASS_DOWN_HOB_GUID {
                    log::info!("Found MM Supervisor PassDown HOB");

                    // Verify data size
                    if data.len() < core::mem::size_of::<MmSupvPassDownHobData>() {
                        log::error!(
                            "PassDown HOB data too small: {} < {}",
                            data.len(),
                            core::mem::size_of::<MmSupvPassDownHobData>()
                        );
                        return Err(PolicyInitError::InvalidPolicyData);
                    }

                    // Cast to PassDown HOB data structure
                    let pass_down = unsafe { &*(data.as_ptr() as *const MmSupvPassDownHobData) };

                    // Copy packed struct fields to local variables to avoid unaligned access
                    // SAFETY: read_unaligned is used because MmSupvPassDownHobData is packed
                    let revision = unsafe { core::ptr::addr_of!(pass_down.revision).read_unaligned() };
                    let firmware_policy_buffer = unsafe { core::ptr::addr_of!(pass_down.mm_supv_firmware_policy_buffer).read_unaligned() };
                    let firmware_policy_buffer_size = unsafe { core::ptr::addr_of!(pass_down.mm_supv_firmware_policy_buffer_size).read_unaligned() };
                    let memory_policy_buffer = unsafe { core::ptr::addr_of!(pass_down.mm_supv_memory_policy_buffer).read_unaligned() };
                    let memory_policy_buffer_size = unsafe { core::ptr::addr_of!(pass_down.mm_supv_memory_policy_buffer_size).read_unaligned() };

                    // Validate revision
                    if revision != MM_SUPV_PASS_DOWN_HOB_REVISION {
                        log::error!(
                            "Invalid PassDown HOB revision: {} (expected {})",
                            revision,
                            MM_SUPV_PASS_DOWN_HOB_REVISION
                        );
                        return Err(PolicyInitError::InvalidRevision {
                            found: revision,
                            expected: MM_SUPV_PASS_DOWN_HOB_REVISION,
                        });
                    }

                    log::info!(
                        "PassDown HOB: FirmwarePolicyBuffer=0x{:x}, Size=0x{:x}",
                        firmware_policy_buffer,
                        firmware_policy_buffer_size
                    );
                    log::info!(
                        "PassDown HOB: MemoryPolicyBuffer=0x{:x}, Size=0x{:x}",
                        memory_policy_buffer,
                        memory_policy_buffer_size
                    );

                    // Validate firmware policy buffer
                    if firmware_policy_buffer == 0
                        || firmware_policy_buffer_size == 0
                    {
                        log::error!("Firmware policy buffer is null or empty");
                        return Err(PolicyInitError::NullFirmwarePolicyBuffer);
                    }

                    // Initialize the policy gate with the firmware policy buffer
                    let policy_ptr = firmware_policy_buffer as *const u8;
                    // SAFETY: We validated that policy_ptr is non-null above and comes from
                    // the PassDown HOB which is set up by the MM IPL.
                    match unsafe { patina_mm_policy::PolicyGate::new(policy_ptr) } {
                        Ok(gate) => {
                            log::info!("Policy gate initialized successfully");
                            // TODO: Store the policy gate for later use
                            // For now, dump the policy for debugging
                            // SAFETY: policy_ptr points to valid policy data as validated above.
                            unsafe { patina_mm_policy::dump_policy(policy_ptr) };
                            // Store the initialized policy gate in the static variable for global access
                            POLICY_GATE.call_once(|| gate);
                        }
                        Err(e) => {
                            log::error!("Failed to create policy gate: {:?}", e);
                            return Err(PolicyInitError::InvalidPolicyData);
                        }
                    }

                    // Read CR3 from hardware
                    let cr3: u64 = read_cr3();

                    // Walk page table and generate memory policy
                    let count = unsafe {
                        walk_page_table(
                            cr3,
                            memory_policy_buffer as *mut MemDescriptorV1_0,
                            memory_policy_buffer_size as usize,
                            |base, size| is_buffer_inside_mmram(base, size), // Your MMRAM check
                        )
                    };

                    if let Ok(descriptor_count) = count {
                        log::info!("Successfully generated {} memory policy descriptors", descriptor_count);

                        // Initialize the unblocked memory tracker from the generated descriptors
                        // SAFETY: memory_policy_buffer points to valid MemDescriptorV1_0 array
                        // with descriptor_count entries, as we just filled it via walk_page_table
                        if let Err(e) = unsafe {
                            unblock_memory::UNBLOCKED_MEMORY_TRACKER.init_from_buffer(
                                memory_policy_buffer as *const MemDescriptorV1_0,
                                descriptor_count,
                            )
                        } {
                            log::error!("Failed to initialize unblocked memory tracker: {:?}", e);
                        } else {
                            log::info!("Unblocked memory tracker initialized");
                            // Dump regions for debugging
                            unblock_memory::UNBLOCKED_MEMORY_TRACKER.dump_regions();
                        }
                    } else {
                        log::error!("Failed to generate memory policy descriptors: {:?}", count.err());
                    }

                    log::info!("Generated {} memory policy descriptors", count.unwrap_or(0));

                    return Ok(());
                }
            }
        }

        log::error!("PassDown HOB not found in HOB list");
        Err(PolicyInitError::PassDownHobNotFound)
    }

    /// The main request serving loop for the BSP.
    ///
    /// The BSP sits in this loop, processing incoming requests and dispatching
    /// work to APs as needed.
    fn bsp_request_loop(&'static self) -> ! {
        log::trace!("BSP entering request serving loop...");

        loop {
            // Check for incoming requests
            // In a real implementation, this would check the communication buffer
            // and dispatch handlers for incoming MM requests.

            // Process any pending work
            self.process_pending_requests();

            // Brief pause to avoid spinning too aggressively
            core::hint::spin_loop();
        }
    }

    /// Process pending requests from the communication buffer.
    fn process_pending_requests(&self) {
        // TODO: Check communication buffer for incoming requests
        // TODO: Dispatch to registered handlers via self.request_dispatcher
        // TODO: Optionally distribute work to APs via mailbox
    }

    /// The holding pen for APs.
    ///
    /// APs wait here, polling their mailbox for commands from the BSP.
    fn ap_holding_pen(&'static self, cpu_id: u32) -> ! {
        log::trace!("AP (CPU {}) in holding pen, polling mailbox...", cpu_id);

        loop {
            // Check mailbox for commands
            if let Some(command) = self.mailbox_manager.check_mailbox(cpu_id) {
                log::trace!("AP (CPU {}) received command: {:?}", cpu_id, command);

                // Execute the command
                let response = self.execute_ap_command(cpu_id, command);

                // Post the response
                self.mailbox_manager.post_response(cpu_id, response);
            }

            // Pause to avoid spinning too aggressively
            // In a production system, this might use MWAIT or HLT
            for _ in 0..1000 {
                core::hint::spin_loop();
            }
        }
    }

    /// Execute a command received by an AP.
    fn execute_ap_command(&self, cpu_id: u32, command: ApCommand) -> ApResponse {
        match command {
            ApCommand::Nop => {
                log::trace!("AP (CPU {}) executing NOP", cpu_id);
                ApResponse::Success
            }
            ApCommand::Execute { handler_id, context } => {
                log::trace!("AP (CPU {}) executing handler {}", cpu_id, handler_id);
                // TODO: Look up and execute the handler
                let _ = context;
                ApResponse::Success
            }
            ApCommand::Shutdown => {
                log::trace!("AP (CPU {}) received shutdown command", cpu_id);
                ApResponse::Success
            }
        }
    }

    /// Register a request handler with the supervisor.
    ///
    /// Handlers are invoked when matching requests are received.
    ///
    /// Returns `true` if the handler was registered, `false` if the handler table is full.
    pub fn register_handler(&self, handler: &'static dyn RequestHandler) -> bool {
        self.request_dispatcher.register(handler)
    }

    /// Get the CPU manager.
    pub fn cpu_manager(&self) -> &CpuManager<{ P::MAX_CPU_COUNT }> {
        &self.cpu_manager
    }

    /// Get the mailbox manager.
    pub fn mailbox_manager(&self) -> &MailboxManager<{ P::MAX_CPU_COUNT }> {
        &self.mailbox_manager
    }

    /// Get the request dispatcher.
    pub fn request_dispatcher(&self) -> &RequestDispatcher<{ P::MAX_HANDLERS }> {
        &self.request_dispatcher
    }

    /// Send a command to a specific AP.
    ///
    /// Returns `Ok(())` if the command was successfully posted to the mailbox,
    /// or `Err(())` if the AP is not available or the mailbox is full.
    pub fn send_ap_command(&self, cpu_id: u32, command: ApCommand) -> Result<(), ()> {
        self.mailbox_manager.send_command(cpu_id, command)
    }

    /// Wait for a response from a specific AP.
    ///
    /// Returns the response from the AP, or `None` if timeout or error.
    pub fn wait_ap_response(&self, cpu_id: u32, timeout_us: u64) -> Option<ApResponse> {
        self.mailbox_manager.wait_response(cpu_id, timeout_us)
    }

    /// Check if the supervisor has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }
}

impl<P: PlatformInfo> Default for MmSupervisorCore<P>
where
    [(); P::MAX_CPU_COUNT]:,
    [(); P::MAX_HANDLERS]:,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlatform;

    impl CpuInfo for TestPlatform {}

    impl PlatformInfo for TestPlatform {
        type CpuInfo = Self;
        const MAX_CPU_COUNT: usize = 4;
        const MAX_HANDLERS: usize = 8;
    }

    #[test]
    fn test_cpu_info_defaults() {
        assert_eq!(<TestPlatform as CpuInfo>::ap_poll_timeout_us(), 1000);
    }

    #[test]
    fn test_supervisor_creation() {
        let _supervisor: MmSupervisorCore<TestPlatform> = MmSupervisorCore::new();
        // Just verify it compiles and creates without panic
    }

    #[test]
    fn test_supervisor_is_const() {
        // Verify we can create a static instance (no heap allocation)
        static _SUPERVISOR: MmSupervisorCore<TestPlatform> = MmSupervisorCore::new();
    }
}
