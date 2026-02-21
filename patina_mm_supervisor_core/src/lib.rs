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
pub mod perf_timer;
pub mod privilege_mgmt;
pub mod save_state;
pub mod supervisor_handlers;
pub mod unblock_memory;

pub use cpu::{ApState, CpuInfo, CpuManager, get_current_cpu_id, is_bsp};
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
pub use unblock_memory::{
    UnblockedMemoryTracker, UnblockedMemoryEntry, UnblockError,
    UNBLOCKED_MEMORY_TRACKER,
};
pub use privilege_mgmt::{
    SyscallInterface,
    invoke_demoted_routine,
};
pub use supervisor_handlers::{
    SupervisorMmiHandler, SUPERVISOR_MMI_HANDLERS,
};

use core::{
    arch::{asm, global_asm}, ffi::c_void, num::NonZeroUsize, panic, ptr::NonNull, sync::atomic::{AtomicBool, AtomicU32, Ordering}
};

use patina::pi::hob::{Hob, PhaseHandoffInformationTable};
use patina_paging::{MemoryAttributes, PageTable, PagingType, x64::X64PageTable};
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
    /// Size of the MMI entry point structure (for validating against expected size in supervisor)
    pub mmi_entrypoint_size: u64,
    /// Base address of the BSP MM
    pub bsp_mm_base_address: u64,
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

use spin::{Mutex, Once};
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

/// Flag indicating that BSP one-time initialization is complete.
static BSP_INIT_COMPLETE: AtomicBool = AtomicBool::new(false);

/// Pointer to the per-core initialized buffer from the PassDown HOB.
/// Each core has a 64-bit slot at `buffer_base + (cpu_index * 8)`.
/// A non-zero value indicates the core has completed initialization.
static MM_INITIALIZED_BUFFER: Once<u64> = Once::new();

/// Counter for tracking how many cores have completed their per-core init.
static PER_CORE_INIT_COUNT: AtomicU32 = AtomicU32::new(0);

/// The policy object is initialized once during BSP initialization and provides access to the security policy
/// for the MM Supervisor. It is stored in a static variable for global access.
/// The policy gate is initialized from the firmware policy buffer provided in the PassDown HOB.
pub(crate) static POLICY_GATE: Once<patina_mm_policy::PolicyGate> = Once::new();

/// Global page table instance for managing page attributes.
///
/// Initialized during BSP init from the active CR3 register. This allows the supervisor
/// to modify page table attributes (e.g., marking supervisor pages as R/W + NX) when
/// allocating memory.
pub(crate) static PAGE_TABLE: Mutex<Option<X64PageTable<SharedPagingAllocator>>> = Mutex::new(None);

/// Result of a page table ownership query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PageOwnership {
    /// The page is user-accessible (U/S = 1, SpecialPurpose clear).
    User,
    /// The page is supervisor-only (U/S = 0, SpecialPurpose set).
    Supervisor,
}

/// Aligns an address and size to page boundaries for page table queries.
///
/// Rounds the address down to the nearest page boundary and adjusts the size
/// upward so the entire original range `[address, address+size)` is covered.
///
/// Returns `(page_aligned_address, page_aligned_size)`.
#[inline]
fn page_align_range(address: u64, size: u64) -> (u64, u64) {
    const PAGE_MASK: u64 = (PAGE_SIZE as u64) - 1;
    let aligned_start = address & !PAGE_MASK;
    let end = address.saturating_add(size);
    let aligned_end = end.saturating_add(PAGE_MASK) & !PAGE_MASK;
    (aligned_start, aligned_end.saturating_sub(aligned_start))
}

/// Queries the page table to determine whether an address is mapped and accessible.
///
/// The address and size are page-aligned before querying (rounded down / up respectively).
///
/// Returns `Ok(true)` if the address is mapped (regardless of privilege level),
/// `Ok(false)` if the page table is not initialized, or `Err(PtError)` if the
/// query fails (e.g., unmapped address).
pub(crate) fn is_address_mapped(address: u64, size: u64) -> Result<bool, patina_paging::PtError> {
    let (aligned_addr, aligned_size) = page_align_range(address, size);
    let page_table = PAGE_TABLE.lock();
    match page_table.as_ref() {
        Some(pt) => {
            pt.query_memory_region(aligned_addr, aligned_size)?;
            Ok(true)
        }
        None => Ok(false),
    }
}

/// Queries the page table to determine the ownership (user vs supervisor) of an address.
///
/// The address and size are page-aligned before querying (rounded down / up respectively).
///
/// Checks the `SpecialPurpose` attribute which maps to the U/S bit on X64:
///   - `SpecialPurpose` set  => `PageOwnership::Supervisor` (U/S = 0)
///   - `SpecialPurpose` clear => `PageOwnership::User` (U/S = 1)
///
/// Returns `None` if the page table is not initialized or the address is unmapped.
pub(crate) fn query_address_ownership(address: u64, size: u64) -> Option<PageOwnership> {
    let (aligned_addr, aligned_size) = page_align_range(address, size);
    let page_table = PAGE_TABLE.lock();
    let pt = page_table.as_ref()?;
    let attrs = pt.query_memory_region(aligned_addr, aligned_size).ok()?;
    if attrs.contains(MemoryAttributes::SpecialPurpose) {
        Some(PageOwnership::Supervisor)
    } else {
        Some(PageOwnership::User)
    }
}

// ============================================================================
// Communication Buffer Pointers (from PassDown HOB)
// ============================================================================

/// Communication buffer configuration extracted from PassDown HOB.
#[derive(Debug, Clone, Copy, Default)]
pub struct CommBufferConfig {
    /// MM Supervisor communication buffer (external interface).
    pub supv_comm_buffer: u64,
    /// MM Supervisor internal communication buffer.
    pub supv_comm_buffer_internal: u64,
    /// Size of supervisor communication buffer.
    pub supv_comm_buffer_size: u64,
    /// MM User communication buffer (external interface).
    pub user_comm_buffer: u64,
    /// MM User internal communication buffer.
    pub user_comm_buffer_internal: u64,
    /// Size of user communication buffer.
    pub user_comm_buffer_size: u64,
    /// MM Supervisor status buffer (indicates target: supervisor or user).
    pub status_buffer: u64,
    /// MM Supervisor to User buffer.
    pub supv_to_user_buffer: u64,
    /// Size of Supervisor to User buffer.
    pub supv_to_user_buffer_size: u64,
}

/// Communication buffer configuration initialized from PassDown HOB.
pub(crate) static COMM_BUFFER_CONFIG: Once<CommBufferConfig> = Once::new();

/// User module entry point discovered from HOB list.
static USER_ENTRY_POINT: Once<u64> = Once::new();

/// Pointer to the SMM_CPU_PRIVATE_DATA structure from the PassDown HOB.
/// This is used to access the SmmCoreEntryContext for user request dispatch.
static SMM_CPU_PRIVATE: Once<u64> = Once::new();

/// Type-erased function pointer for AP startup dispatch.
///
/// This is set during [`MmSupervisorCore`] initialization. The function is monomorphized
/// for the concrete `PlatformInfo` type so the syscall dispatcher can invoke it without
/// knowing the platform's const-generic parameters.
///
/// Signature: `fn(cpu_index: u64, procedure: u64, argument: u64) -> u64`
///
/// Returns 0 on success, or an EFI status code on failure.
pub(crate) static AP_STARTUP_FN: Once<fn(u64, u64, u64) -> u64> = Once::new();

/// MM Communication Buffer Status Structure.
/// Matches the C structure MM_COMM_BUFFER_STATUS from MmCommBuffer.h
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MmCommBufferStatus {
    /// Whether the data in the fixed MM communication buffer is valid when entering from non-MM to MM.
    pub is_comm_buffer_valid: u8, // BOOLEAN in C is u8 in Rust
    /// The channel used to communicate with MM (true = Supervisor, false = User).
    pub talk_to_supervisor: u8, // BOOLEAN in C is u8 in Rust
    /// The return status when returning from MM to non-MM.
    pub return_status: u64,
    /// The size in bytes of the output buffer when returning from MM to non-MM.
    pub return_buffer_size: u64,
}

/// EFI_SMM_RESERVED_SMRAM_REGION structure.
///
/// Describes a reserved SMRAM region that cannot be used for the SMRAM heap.
/// Matches the C structure from PI specification.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfiSmmReservedSmramRegion {
    /// Starting address of the reserved SMRAM area.
    pub smram_reserved_start: u64,
    /// Number of bytes occupied by the reserved SMRAM area.
    pub smram_reserved_size: u64,
}

/// EFI_MM_ENTRY_CONTEXT structure.
///
/// Processor information and functionality needed by MM Foundation.
/// Matches the C `EFI_MM_ENTRY_CONTEXT` / `EFI_SMM_ENTRY_CONTEXT` from PI specification.
///
/// Layout (x86_64, all fields 8 bytes):
/// - `mm_startup_this_ap`: Function pointer for `EFI_MM_STARTUP_THIS_AP`
/// - `currently_executing_cpu`: Index of the processor executing the MM Foundation
/// - `number_of_cpus`: Total number of possible processors in the platform (1-based)
/// - `cpu_save_state_size`: Pointer to array of save state sizes per CPU
/// - `cpu_save_state`: Pointer to array of CPU save state pointers
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfiMmEntryContext {
    /// Function pointer for EFI_MM_STARTUP_THIS_AP.
    pub mm_startup_this_ap: u64,
    /// Index of the currently executing CPU.
    pub currently_executing_cpu: u64,
    /// Total number of CPUs (1-based).
    pub number_of_cpus: u64,
    /// Pointer to array of per-CPU save state sizes.
    pub cpu_save_state_size: u64,
    /// Pointer to array of per-CPU save state pointers.
    pub cpu_save_state: u64,
}

/// SMM_CPU_PRIVATE_DATA structure.
///
/// Private structure for the SMM CPU module, passed from PEI via the PassDown HOB.
/// Matches the C `SMM_CPU_PRIVATE_DATA` layout from MpService.h.
///
/// Layout (x86_64):
/// ```text
/// Offset  Field
/// 0x00    signature (UINTN)
/// 0x08    smm_cpu_handle (EFI_HANDLE)
/// 0x10    processor_info (ptr)
/// 0x18    cpu_save_state_size (ptr)
/// 0x20    cpu_save_state (ptr)
/// 0x28    smm_reserved_smram_region[1] (16 bytes)
/// 0x38    smm_core_entry_context (40 bytes, inline)
/// 0x60    smm_core_entry (fn ptr)
/// 0x68    smm_user_entry (fn ptr)
/// 0x70    ap_wrapper_func (ptr)
/// 0x78    token_list (ptr)
/// 0x80    first_free_token (ptr)
/// Total:  0x88 bytes
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SmmCpuPrivateData {
    /// Signature ('scpu').
    pub signature: u64,
    /// SMM CPU handle.
    pub smm_cpu_handle: u64,
    /// Pointer to processor information array.
    pub processor_info: u64,
    /// Pointer to per-CPU save state size array.
    pub cpu_save_state_size: u64,
    /// Pointer to per-CPU save state pointer array.
    pub cpu_save_state: u64,
    /// Reserved SMRAM region descriptor (single element array).
    pub smm_reserved_smram_region: EfiSmmReservedSmramRegion,
    /// Inline entry context structure (40 bytes).
    pub smm_core_entry_context: EfiMmEntryContext,
    /// Supervisor core entry point function pointer.
    pub smm_core_entry: u64,
    /// User core entry point function pointer.
    pub smm_user_entry: u64,
    /// AP wrapper function pointer.
    pub ap_wrapper_func: u64,
    /// Token list pointer.
    pub token_list: u64,
    /// First free token pointer.
    pub first_free_token: u64,
}

/// EFI_MM_COMMUNICATE_HEADER structure.
///
/// Communication buffer header used by the MM Communicate protocol.
/// The data payload immediately follows this header.
///
/// Layout:
/// - `header_guid`: 16 bytes - GUID identifying the handler
/// - `message_length`: 8 bytes - size of `Data` in bytes (does not include header size)
///
/// Note: Although the C definition uses `#pragma pack(1)`, the fields are naturally aligned
/// (16-byte GUID + 8-byte u64), so `#[repr(C)]` produces an identical layout of 24 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfiMmCommunicateHeader {
    /// GUID identifying the target handler for this communication.
    pub header_guid: efi::Guid,
    /// Size of the data payload in bytes (does not include this header).
    pub message_length: u64,
    // Variable-length data follows at offset 24 (0x18)
}

impl EfiMmCommunicateHeader {
    /// Size of the header (offset to the start of the data payload).
    pub const HEADER_SIZE: usize = core::mem::size_of::<Self>();
}

/// Request target derived from MM_COMM_BUFFER_STATUS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestTarget {
    /// No pending request (buffer not valid).
    None,
    /// Request targets the Supervisor.
    Supervisor,
    /// Request targets the User module.
    User,
}

impl From<&MmCommBufferStatus> for RequestTarget {
    fn from(status: &MmCommBufferStatus) -> Self {
        if status.is_comm_buffer_valid == 0 {
            RequestTarget::User
        } else if status.talk_to_supervisor != 0 {
            RequestTarget::Supervisor
        } else {
            RequestTarget::User
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserCommandType {
    /// Command to initiate the user level core
    StartUserCore,
    /// Command to execute a user level request from the supervisor
    UserRequest,
    /// Command to implement a user AP procedure
    UserApProcedure,
}


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
    /// Syscall interface for privilege transitions.
    syscall_interface: SyscallInterface<{ P::MAX_CPU_COUNT }>,
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

pub(crate) fn is_buffer_inside_mmram(base: u64, size: u64) -> bool {
    // we will go over the page allocator to see if this region falls inside any of the MMRAM regions
    mm_mem::PAGE_ALLOCATOR.is_region_inside_mmram(base, size)
}

/// Read CR3 register.
pub(crate) fn read_cr3() -> u64 {
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

use x86_64::structures::DescriptorTablePointer;

// ============================================================================
// SMI Handler Fixup Constants
// ============================================================================

/// Offset from SMBASE where the SMI handler code is located.
const SMM_HANDLER_OFFSET: u64 = 0x8000;

/// Index into the Fixup64 array for the SMI handler IDTR pointer.
const FIXUP64_SMI_HANDLER_IDTR: usize = 5;

/// Per-core MMI entry structure header.
///
/// This packed structure is embedded at the end of the SMI handler binary template.
/// It contains offsets (relative to the header start) to fixup arrays that the
/// relocation code uses to patch per-CPU values into the binary.
///
/// Layout matches the C `PER_CORE_MMI_ENTRY_STRUCT_HDR` from SeaResponder.h.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct PerCoreMmiEntryStructHdr {
    /// Header version (4 for version 4).
    header_version: u32,
    /// Offset from header start to FixUpStruct array.
    fixup_struct_offset: u8,
    /// Number of FixUpStruct array entries.
    fixup_struct_num: u8,
    /// Offset from header start to Fixup64 array.
    fixup64_offset: u8,
    /// Number of Fixup64 array entries.
    fixup64_num: u8,
    /// Offset from header start to Fixup32 array.
    fixup32_offset: u8,
    /// Number of Fixup32 array entries.
    fixup32_num: u8,
    /// Offset from header start to Fixup8 array.
    fixup8_offset: u8,
    /// Number of Fixup8 array entries.
    fixup8_num: u8,
    /// SMI entry binary version.
    binary_version: u16,
    /// SPL value for SMI entry binary.
    spl_value: u32,
    /// Reserved for future use.
    reserved: u32,
}

/// Read the current IDT Register (IDTR) via the `SIDT` instruction.
///
/// Returns a [`DescriptorTablePointer`] containing the IDT base and limit.
fn read_idtr() -> DescriptorTablePointer {
    let mut descriptor = DescriptorTablePointer {
        limit: 0,
        base: x86_64::VirtAddr::zero(),
    };

    #[cfg(all(not(test), target_arch = "x86_64"))]
    {
        // SAFETY: SIDT stores the 10-byte IDTR pseudo-descriptor to the specified
        // memory location. This is a read-only operation on CPU state.
        unsafe {
            asm!(
                "sidt [{}]",
                in(reg) &mut descriptor as *mut DescriptorTablePointer,
                options(nostack, preserves_flags)
            );
        }
    }

    descriptor
}

// ============================================================================
// Per-Core Initialization Status Helpers
// ============================================================================

/// Checks if a specific core has completed initialization.
///
/// Reads the 64-bit slot at `mm_initialized_buffer + (cpu_index * 8)`.
/// A non-zero value indicates the core has completed initialization.
fn is_core_initialized(cpu_index: usize) -> bool {
    if let Some(&buffer_base) = MM_INITIALIZED_BUFFER.get() {
        if buffer_base == 0 {
            return false;
        }
        let slot_ptr = (buffer_base as usize + cpu_index) as *const u8;
        // SAFETY: The buffer is provided by the MM IPL and is guaranteed to be valid.
        // Each core only reads its own slot or slots of other cores.
        let value = unsafe { core::ptr::read_volatile(slot_ptr) };
        value != 0
    } else {
        false
    }
}

/// Marks a specific core as initialized.
///
/// Writes a non-zero value to the 64-bit slot at `mm_initialized_buffer + (cpu_index * 8)`.
fn mark_core_initialized(cpu_index: usize) {
    if let Some(&buffer_base) = MM_INITIALIZED_BUFFER.get() {
        if buffer_base == 0 {
            log::error!("MM initialized buffer is null, cannot mark core {} as initialized", cpu_index);
            return;
        }
        let slot_ptr = (buffer_base as usize + cpu_index) as *mut u8;
        // SAFETY: The buffer is provided by the MM IPL and is guaranteed to be valid.
        // Each core writes only to its own slot.
        unsafe { core::ptr::write_volatile(slot_ptr, 1) };
        log::trace!("Core {} marked as initialized at 0x{:016x}", cpu_index, slot_ptr as u64);
    } else {
        log::error!("MM initialized buffer not set, cannot mark core {} as initialized", cpu_index);
    }
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
            syscall_interface: SyscallInterface::new(),
            initialized: AtomicBool::new(false),
            _phantom: core::marker::PhantomData,
        }
    }

    /// Sets the static supervisor instance for global access.
    ///
    /// Returns true if the address was successfully stored, false if already set.
    /// Also registers the type-erased AP startup function pointer.
    #[must_use]
    fn set_instance(&'static self) -> bool {
        let physical_address = NonNull::from_ref(self).expose_provenance();
        let stored = &physical_address == __SUPERVISOR.call_once(|| physical_address);
        if stored {
            // Register the monomorphized AP startup function for this platform
            AP_STARTUP_FN.call_once(|| Self::start_ap_procedure_trampoline);
        }
        stored
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
    /// 
    /// # Returns
    /// 
    /// On the first call (initialization phase), this function returns after init is complete.
    /// On subsequent calls, BSP enters the request loop and APs enter the holding pen (neither returns).
    pub fn entry_point(&'static self, cpu_index: usize, hob_list: *const c_void) {
        // Get the current CPU's APIC ID
        let cpu_id = cpu::get_current_cpu_id();

        // Determine if we're BSP by checking IA32_APIC_BASE MSR
        let is_bsp = cpu::is_bsp();

        // Check if this core has already completed initialization (per-core check)
        if is_core_initialized(cpu_index) {
            // Subsequent entry: go directly to request loop or holding pen (does not return)
            self.enter_runtime(cpu_id);

            return;
        }

        // First entry: initialization phase
        if is_bsp {
            // BSP path: Initialize the supervisor
            assert!(self.set_instance(), "MM Supervisor Core instance was already set!");
            assert!(!hob_list.is_null(), "MM Supervisor Core requires a non-null HOB list pointer.");

            log::info!("MM Supervisor Core v{}", env!("CARGO_PKG_VERSION"));
            log::info!("BSP (CPU {}, index {}) starting one-time initialization...", cpu_id, cpu_index);

            // Register BSP with CPU manager
            self.cpu_manager.register_cpu(cpu_id, true);

            // Perform BSP-only one-time initialization (this sets up MM_INITIALIZED_BUFFER)
            self.bsp_init(hob_list);

            // Dispatch to the user level entry point discovered from the HOB list (if found)
            let user_entry = match USER_ENTRY_POINT.get() {
                Some(&entry) if entry != 0 => entry,
                _ => {
                    log::error!("User entry point not configured, cannot demote");
                    return;
                }
            };

            let cpl3_stack = match self.syscall_interface.get_cpl3_stack(cpu_index) {
                Ok(stack) => stack,
                Err(e) => {
                    log::error!("Failed to get CPL3 stack for CPU {}: {:?}", cpu_index, e);
                    return;
                }
            };

            // SAFETY: We are transitioning from the supervisor (CPL0) to the user module (CPL3) for the first time.
            // The entry point and stack have been validated and set up during initialization, and the user module is
            // will be responsible for validating any further inputs.
            let ret = unsafe {
                invoke_demoted_routine (
                    cpu_index,
                    user_entry,
                    cpl3_stack,
                    3,
                    UserCommandType::StartUserCore as u64,
                    hob_list,
                    0)
            };
            log::info!("Returned from user entry point with value: 0x{:016x}", ret);

            // Mark BSP init as complete so APs can proceed
            self.initialized.store(true, Ordering::Release);
            BSP_INIT_COMPLETE.store(true, Ordering::Release);

            log::info!("BSP one-time initialization complete.");
        } else {
            // AP path: Wait for BSP to complete one-time initialization
            log::trace!("AP (CPU {}, index {}) waiting for BSP initialization...", cpu_id, cpu_index);

            // Spin until BSP completes initialization
            while !BSP_INIT_COMPLETE.load(Ordering::Acquire) {
                core::hint::spin_loop();
            }

            // Register this AP with the CPU manager
            self.cpu_manager.register_cpu(cpu_id, false);
        }

        // All cores perform per-core initialization
        self.per_core_init(cpu_id, is_bsp);

        // Mark this core as initialized in the per-core buffer
        mark_core_initialized(cpu_index);

        // Track that this core has completed per-core init
        let init_count = PER_CORE_INIT_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
        log::trace!("CPU {} (index {}) completed per-core init ({} cores initialized)", cpu_id, cpu_index, init_count);

        // BSP waits for all registered CPUs to complete per-core init before returning
        if is_bsp {
            let expected_cpus = self.cpu_manager.registered_count();
            while PER_CORE_INIT_COUNT.load(Ordering::Acquire) < expected_cpus as u32 {
                core::hint::spin_loop();
            }

            log::info!("All {} cores completed initialization, returning to caller.", expected_cpus);
        }

        // First entry returns to caller after init is complete
        // (Each core has already marked itself as initialized via mark_core_initialized)
    }

    /// BSP-specific initialization.
    ///
    /// This is called only on the BSP after basic setup is complete.
    fn bsp_init(&'static self, hob_list: *const c_void) {
        log::info!("BSP performing one-time initialization...");

        // Initialize the performance timer early so all subsequent code
        // can use real TSC-based timeouts.
        perf_timer::init(P::CpuInfo::perf_timer_frequency().unwrap_or(0));

        let mut interrupt_manager = Interrupts::new();
        interrupt_manager.initialize().unwrap_or_else(|err| {
            panic!("Failed to initialize Interrupt Manager: {:?}", err);
        });

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

        // Initialize the global page table from the active CR3.
        // This allows the supervisor to modify page attributes on newly allocated pages.
        let cr3 = read_cr3();
        let paging_alloc = paging_allocator::SharedPagingAllocator::new(&paging_allocator::PAGING_ALLOCATOR);
        let page_table = unsafe {
            X64PageTable::from_existing(cr3, paging_alloc, PagingType::Paging4Level)
        }.expect("Failed to create page table from active CR3");
        *PAGE_TABLE.lock() = Some(page_table);
        log::info!("Page table initialized from CR3=0x{:016x}", cr3);

        // Discover the MM Supervisor User module entry point from the HOB list.
        // We look for EFI_HOB_TYPE_MEMORY_ALLOCATION HOBs that have:
        // - MemoryAllocationHeader.Name == gMmSupervisorHobMemoryAllocModuleGuid
        // - ModuleName == gMmSupervisorUserGuid
        // SAFETY: hob_list is provided by the MM IPL and is guaranteed to be valid
        let user_entry_point = unsafe { self.discover_user_module_entry(hob_list) };
        if let Some(entry) = user_entry_point {
            log::info!("Discovered MM User module entry point: 0x{:016x}", entry);
            // Store entry point in static for use during request processing
            USER_ENTRY_POINT.call_once(|| entry);
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

        log::trace!("BSP one-time initialization complete.");
    }

    /// Patches the SMI handler's IDT descriptor to point to Rust interrupt handlers.
    ///
    /// Navigates the per-core MMI entry fixup structure embedded at the end of the
    /// SMI handler binary to locate the `gSmiHandlerIdtr` pointer, then overwrites
    /// it with the current IDT descriptor (base + limit).
    ///
    /// # Arguments
    ///
    /// * `mmi_entry_size` - Size of the MMI entry binary from the PassDown HOB.
    fn patch_smi_handler_idt(mmbase: u64, mmi_entry_size: u64) {
        if mmi_entry_size == 0 {
            log::warn!("MMI entry size is 0 in PassDown HOB, cannot navigate fixup structure");
            return;
        }

        // SAFETY: Reading SMBASE MSR is safe during BSP init in SMM context.
        let mut smbase = unsafe { cpu::read_msr(cpu::IA32_MSR_SMBASE) }.unwrap_or_else(|err| {
            panic!("Failed to read IA32_MSR_SMBASE: {:?}", err);
        });

        if smbase == 0 {
            smbase = mmbase;
        }

        let mmi_entry_base = smbase + SMM_HANDLER_OFFSET;
        log::info!("MMI entry at 0x{:016x} with size 0x{:x}", mmi_entry_base, mmi_entry_size);

        // The last u32 in the MMI entry binary is the total fixup structure size.
        let whole_struct_size_addr = mmi_entry_base + mmi_entry_size - 4;
        // SAFETY: whole_struct_size_addr points into the SMI handler template in SMRAM.
        let whole_struct_size = unsafe {
            core::ptr::read_unaligned(whole_struct_size_addr as *const u32)
        };

        // The structure header starts before the trailing size field.
        let hdr_addr = (mmi_entry_base + mmi_entry_size - 4 - whole_struct_size as u64)
            as *const PerCoreMmiEntryStructHdr;
        // SAFETY: hdr_addr points to the packed fixup header within the SMI handler binary.
        let hdr = unsafe { core::ptr::read_unaligned(hdr_addr) };

        let hdr_version = hdr.header_version;
        let f64_offset = hdr.fixup64_offset;
        let f64_num = hdr.fixup64_num;
        log::trace!(
            "Fixup header at 0x{:016x}: version={}, fixup64_offset={}, fixup64_num={}",
            hdr_addr as u64, hdr_version, f64_offset, f64_num
        );

        // Validate the Fixup64 array has the IDTR entry.
        if (FIXUP64_SMI_HANDLER_IDTR as u8) >= f64_num {
            log::error!(
                "Fixup64 array too small: need index {} but only {} entries",
                FIXUP64_SMI_HANDLER_IDTR, f64_num
            );
            return;
        }

        // Navigate to the Fixup64 array and read the IDTR entry.
        let fixup64_base = (hdr_addr as u64 + f64_offset as u64) as *const u64;
        // SAFETY: fixup64_base + index is within the fixup array in the SMI handler binary.
        let idt_desc_addr = unsafe {
            core::ptr::read_unaligned(fixup64_base.add(FIXUP64_SMI_HANDLER_IDTR))
        };

        if idt_desc_addr == 0 {
            log::warn!("Fixup64[{}] (SMI_HANDLER_IDTR) is null", FIXUP64_SMI_HANDLER_IDTR);
            return;
        }

        // Overwrite the IA32_DESCRIPTOR with our Rust IDT's base/limit.
        let idt_desc_ptr = idt_desc_addr as *mut DescriptorTablePointer;
        let idtr = read_idtr();

        // SAFETY: idt_desc_ptr points to an IA32_DESCRIPTOR allocated by the C relocation
        // code via AllocateCodePages(1). Both DescriptorTablePointer (packed(2)) and
        // IA32_DESCRIPTOR (packed(1)) have the same 10-byte {u16, u64} layout.
        unsafe { core::ptr::write_unaligned(idt_desc_ptr, idtr) };

        log::info!(
            "Patched SMI handler IDT descriptor at 0x{:016x}: base=0x{:016x}, limit=0x{:04x}",
            idt_desc_addr, idtr.base.as_u64(), idtr.limit
        );
    }

    /// Per-core initialization.
    ///
    /// This is called on every core (BSP and APs) during the first entry.
    /// Use this for setting up per-CPU state like syscall MSRs, GS base, etc.
    fn per_core_init(&'static self, cpu_id: u32, is_bsp: bool) {
        let core_type = if is_bsp { "BSP" } else { "AP" };
        log::trace!("{} (CPU {}) performing per-core initialization...", core_type, cpu_id);

        // TODO: Set up per-CPU GDT/TSS if needed
        // TODO: Set up per-CPU interrupt stacks
        // TODO: Initialize per-CPU data structures

        log::trace!("{} (CPU {}) per-core initialization complete.", core_type, cpu_id);
    }

    /// Enter runtime mode (called on subsequent entries after init is complete).
    ///
    /// Implements the MP synchronization protocol:
    /// 1. APs check in by setting their state to `InHoldingPen` and entering the holding pen
    /// 2. BSP waits (with timeout) for all registered APs to arrive
    /// 3. BSP processes the pending request via `bsp_request_loop`
    /// 4. BSP broadcasts `Return` to all APs so they exit the holding pen
    /// 5. BSP waits for all AP responses before returning
    fn enter_runtime(&'static self, cpu_id: u32) {
        let is_bsp = cpu::is_bsp();

        if is_bsp {
            log::trace!("BSP (CPU {}) waiting for APs to arrive...", cpu_id);

            // Wait for all registered APs to check in (set state to InHoldingPen)
            let expected_aps = self.cpu_manager.registered_count().saturating_sub(1) as usize;
            self.wait_for_ap_arrival(expected_aps);

            // All APs (or timeout) - proceed with request processing
            log::trace!("BSP (CPU {}) entering request serving routine...", cpu_id);
            self.bsp_request_loop(cpu_id as usize);

            // BSP is done handling the request - broadcast Return to all APs
            log::trace!("BSP (CPU {}) broadcasting Return to all APs...", cpu_id);
            let sent = self.mailbox_manager.broadcast_command(ApCommand::Return);
            log::trace!("BSP (CPU {}) sent Return to {} APs, waiting for acknowledgement...", cpu_id, sent);

            // TODO: Wait for all APs to acknowledge the Return command
            const RETURN_TIMEOUT_US: u64 = 100_000; // 100 ms
            let responded = self.mailbox_manager.wait_all_responses(RETURN_TIMEOUT_US);
            log::trace!(
                "BSP (CPU {}) done: {}/{} APs acknowledged Return",
                cpu_id, responded, sent
            );
        } else {
            // AP: check in by marking state, then enter holding pen
            self.cpu_manager.set_ap_state(cpu_id, cpu::ApState::InHoldingPen);
            log::trace!("AP (CPU {}) checked in, entering holding pen...", cpu_id);
            self.ap_holding_pen(cpu_id);
        }
    }

    /// Waits for APs to arrive with a timeout.
    ///
    /// Spins until the expected number of APs have set their state to `InHoldingPen`,
    /// or the timeout expires (whichever comes first).
    fn wait_for_ap_arrival(&self, expected_aps: usize) {
        if expected_aps == 0 {
            return;
        }

        const AP_ARRIVAL_TIMEOUT_US: u64 = 100_000; // 100 ms

        let all_arrived = perf_timer::spin_until(AP_ARRIVAL_TIMEOUT_US, || {
            self.cpu_manager.count_aps_in_state(cpu::ApState::InHoldingPen) >= expected_aps
        });

        if all_arrived {
            log::trace!("All {} APs arrived", expected_aps);
        } else {
            let arrived = self.cpu_manager.count_aps_in_state(cpu::ApState::InHoldingPen);
            log::warn!(
                "AP arrival timeout: {}/{} APs arrived, proceeding with available cores",
                arrived, expected_aps
            );
        }
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
                    // SAFETY: Direct access to read the addresses from the hob data.
                    let revision = unsafe { core::ptr::addr_of!(pass_down.revision).read() };
                    let mm_initialized_buffer = unsafe { core::ptr::addr_of!(pass_down.mm_initialized_buffer).read() };
                    let firmware_policy_buffer = unsafe { core::ptr::addr_of!(pass_down.mm_supv_firmware_policy_buffer).read() };
                    let firmware_policy_buffer_size = unsafe { core::ptr::addr_of!(pass_down.mm_supv_firmware_policy_buffer_size).read() };
                    let memory_policy_buffer = unsafe { core::ptr::addr_of!(pass_down.mm_supv_memory_policy_buffer).read() };
                    let memory_policy_buffer_size = unsafe { core::ptr::addr_of!(pass_down.mm_supv_memory_policy_buffer_size).read() };

                    // Extract communication buffer pointers
                    let supv_comm_buffer = unsafe { core::ptr::addr_of!(pass_down.mm_supv_comm_buffer).read() };
                    let supv_comm_buffer_internal = unsafe { core::ptr::addr_of!(pass_down.mm_supv_comm_buffer_internal).read() };
                    let supv_comm_buffer_size = unsafe { core::ptr::addr_of!(pass_down.mm_supv_comm_buffer_size).read() };
                    let user_comm_buffer = unsafe { core::ptr::addr_of!(pass_down.mm_user_comm_buffer).read() };
                    let user_comm_buffer_internal = unsafe { core::ptr::addr_of!(pass_down.mm_user_comm_buffer_internal).read() };
                    let user_comm_buffer_size = unsafe { core::ptr::addr_of!(pass_down.mm_user_comm_buffer_size).read() };
                    let status_buffer = unsafe { core::ptr::addr_of!(pass_down.mm_supv_status_buffer).read() };
                    let supv_to_user_buffer = unsafe { core::ptr::addr_of!(pass_down.mm_supv_to_user_buffer).read() };
                    let supv_to_user_buffer_size = unsafe { core::ptr::addr_of!(pass_down.mm_supv_to_user_buffer_size).read() };
                    let cpl3_stack_buffer = unsafe { core::ptr::addr_of!(pass_down.mm_supervisor_cpl3_stack_base).read() };
                    let cpl3_stack_buffer_size = unsafe { core::ptr::addr_of!(pass_down.mm_supervisor_cpl3_per_core_stack_size).read() };
                    let mmi_entry_size = unsafe { core::ptr::addr_of!(pass_down.mmi_entrypoint_size).read() };
                    let mmbase = unsafe { core::ptr::addr_of!(pass_down.bsp_mm_base_address).read() };

                    // Patch the SMI entry IDT descriptor to point to our interrupt handlers.
                    // The C relocation code patches each CPU's SMI handler template with fixup
                    // arrays. The Fixup64 array at index FIXUP64_SMI_HANDLER_IDTR contains the
                    // address of an IA32_DESCRIPTOR (gSmiHandlerIdtr). On SMI entry, the assembly
                    // loads that address and does `lidt [rax]`. We navigate the fixup structure
                    // in the BSP's SMI handler to find this pointer, then overwrite the descriptor
                    // with our Rust IDT's base/limit.
                    Self::patch_smi_handler_idt(mmbase, mmi_entry_size);

                    // Extract CPU private data pointer
                    let cpu_private = unsafe { core::ptr::addr_of!(pass_down.mm_supv_cpu_private).read() };

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

                    // Store the per-core initialized buffer address for use by all cores
                    if mm_initialized_buffer != 0 {
                        MM_INITIALIZED_BUFFER.call_once(|| mm_initialized_buffer);
                        log::info!("MM Initialized buffer set to 0x{:016x}", mm_initialized_buffer);
                    } else {
                        log::warn!("MM Initialized buffer is null in PassDown HOB");
                    }

                    // Store CPU private data pointer for SmmCoreEntryContext access
                    if cpu_private != 0 {
                        SMM_CPU_PRIVATE.call_once(|| cpu_private);
                        log::info!("SMM CPU Private data at 0x{:016x}", cpu_private);
                    } else {
                        log::warn!("SMM CPU Private data pointer is null in PassDown HOB");
                    }

                    // Store communication buffer configuration.
                    COMM_BUFFER_CONFIG.call_once(|| CommBufferConfig {
                        supv_comm_buffer,
                        supv_comm_buffer_internal,
                        supv_comm_buffer_size,
                        user_comm_buffer,
                        user_comm_buffer_internal,
                        user_comm_buffer_size,
                        status_buffer,
                        supv_to_user_buffer,
                        supv_to_user_buffer_size,
                    });
                    log::info!(
                        "Comm buffers: supv=0x{:x}/0x{:x} size=0x{:x}, user=0x{:x}/0x{:x} size=0x{:x}, status=0x{:x}",
                        supv_comm_buffer, supv_comm_buffer_internal, supv_comm_buffer_size,
                        user_comm_buffer, user_comm_buffer_internal, user_comm_buffer_size,
                        status_buffer
                    );

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
                        Ok(mut gate) => {
                            log::info!("Policy gate initialized successfully");
                            // SAFETY: policy_ptr points to valid policy data as validated above.
                            unsafe { patina_mm_policy::dump_policy(policy_ptr) };

                            // Configure the memory policy buffer on the gate so that
                            // take_snapshot / verify_snapshot / fetch_n_update_policy
                            // can use it.
                            let mem_policy_max_count = memory_policy_buffer_size as usize
                                / core::mem::size_of::<MemDescriptorV1_0>();
                            gate.set_memory_policy_buffer(
                                memory_policy_buffer as *mut MemDescriptorV1_0,
                                mem_policy_max_count,
                            );

                            // Store the initialized policy gate in the static variable for global access
                            POLICY_GATE.call_once(|| gate);
                        }
                        Err(e) => {
                            log::error!("Failed to create policy gate: {:?}", e);
                            return Err(PolicyInitError::InvalidPolicyData);
                        }
                    }

                    // Init syscall interface
                    self.syscall_interface.init(
                        self.cpu_manager.max_cpus(),
                        cpl3_stack_buffer,
                        cpl3_stack_buffer_size.try_into().unwrap_or_else(
                            |err| panic!("Invalid CPL3 stack buffer size: {:?}", err)
                        ),
                    ).unwrap_or_else(|err| {
                        panic!("Failed to initialize syscall interface: {:?}", err);
                    });

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
    /// It manages other CPUs and processes pending requests from the communication buffer.
    ///
    /// This function reads the MM_COMM_BUFFER_STATUS structure to determine if there's a pending request
    /// and whether it targets the Supervisor or User module.
    ///
    /// - If targeting User: copies user comm buffer to internal, then demotes to user entry point
    /// - If targeting Supervisor: dispatches to the request dispatcher
    fn bsp_request_loop(&self, cpu_index: usize) {
        // Get communication buffer configuration
        let config = match COMM_BUFFER_CONFIG.get() {
            Some(c) => c,
            None => {
                // Not yet initialized, nothing to process
                return;
            }
        };

        // Check status buffer for pending request
        if config.status_buffer == 0 {
            return;
        }

        // Read the MM_COMM_BUFFER_STATUS structure
        // SAFETY: status_buffer is provided by MM IPL and is guaranteed valid
        let status = unsafe {
            core::ptr::read_volatile(config.status_buffer as *const MmCommBufferStatus)
        };
        let target = RequestTarget::from(&status);

        log::trace!(
            "Processing request: valid={}, talk_to_supervisor={}, target={:?}",
            status.is_comm_buffer_valid,
            status.talk_to_supervisor,
            target
        );

        match target {
            RequestTarget::None => {
                // No pending request
            }
            RequestTarget::User => {
                // Request targets the User module
                self.process_user_request(config, &status, cpu_index);
            }
            RequestTarget::Supervisor => {
                // Request targets the Supervisor
                self.process_supervisor_request(config, &status, cpu_index);
            }
        }
    }

    /// Process a request targeting the User module.
    ///
    /// This function implements the user-mode MMI dispatch pathway:
    /// 1. Builds a fresh `EfiMmEntryContext` with the current CPU index and CPU count
    /// 2. Copies the `EfiMmEntryContext` into the supervisor-to-user data buffer
    /// 3. Appends the `MmCommBufferStatus` immediately after the context
    /// 4. For synchronous MMIs, copies the user comm buffer to the internal copy
    /// 5. Demotes to the user entry point via `invoke_demoted_routine`
    /// 6. On return, copies back the user comm buffer and reads the updated status
    fn process_user_request(&self, config: &CommBufferConfig, status: &MmCommBufferStatus, cpu_index: usize) {
        log::trace!("Processing User request...");

        // Validate buffers
        if config.user_comm_buffer == 0 || config.user_comm_buffer_internal == 0 {
            log::error!("User communication buffer not configured");
            return;
        }

        if config.supv_to_user_buffer == 0 {
            log::error!("Supervisor-to-user data buffer not configured");
            return;
        }

        // Get user entry point
        let user_entry = match USER_ENTRY_POINT.get() {
            Some(&entry) if entry != 0 => entry,
            _ => {
                log::error!("User entry point not configured, cannot demote");
                return;
            }
        };

        // Demote to user entry point to process the request
        let cpl3_stack = match self.syscall_interface.get_cpl3_stack(cpu_index) {
            Ok(stack) => stack,
            Err(e) => {
                log::error!("Failed to get CPL3 stack for CPU {}: {:?}", cpu_index, e);
                return;
            }
        };

        // Build a fresh EfiMmEntryContext with only the fields the user actually needs.
        // The legacy C structure carried pointers (mm_startup_this_ap, cpu_save_state,
        // cpu_save_state_size) that are meaningless in the Rust supervisor model — the
        // user module accesses those services through syscalls instead.
        let entry_context = EfiMmEntryContext {
            mm_startup_this_ap: 0,
            currently_executing_cpu: cpu_index as u64,
            number_of_cpus: self.cpu_manager.registered_count() as u64,
            cpu_save_state_size: 0,
            cpu_save_state: 0,
        };

        log::info!(
            "Built EfiMmEntryContext: currently_executing_cpu={}, number_of_cpus={}",
            entry_context.currently_executing_cpu,
            entry_context.number_of_cpus
        );

        // Copy the EfiMmEntryContext + MmCommBufferStatus into the supervisor-to-user
        // data buffer so the user can read processor information after demotion.
        let context_size = core::mem::size_of::<EfiMmEntryContext>();
        let status_size = core::mem::size_of::<MmCommBufferStatus>();

        // Validate the supervisor-to-user buffer is large enough for context + status
        if (config.supv_to_user_buffer_size as usize) < context_size + status_size {
            log::error!(
                "Supervisor-to-user buffer too small: {} < {} (context) + {} (status)",
                config.supv_to_user_buffer_size,
                context_size,
                status_size
            );
            return;
        }

        // SAFETY: supv_to_user_buffer is valid and large enough, verified above.
        unsafe {
            // Copy the EfiMmEntryContext to the start of the supervisor-to-user buffer
            core::ptr::copy_nonoverlapping(
                &entry_context as *const EfiMmEntryContext as *const u8,
                config.supv_to_user_buffer as *mut u8,
                context_size,
            );

            // Copy the MmCommBufferStatus right after the context
            core::ptr::copy_nonoverlapping(
                status as *const MmCommBufferStatus as *const u8,
                (config.supv_to_user_buffer as *mut u8).add(context_size),
                status_size,
            );
        }

        // Determine whether this is synchronous or asynchronous request
        let sync_mmi = status.is_comm_buffer_valid;

        if sync_mmi != 0 {
            // Copy user buffer to user internal buffer for processing in Ring 3
            // SAFETY: Buffers are provided by MM IPL and are guaranteed valid
            unsafe {
                core::ptr::copy_nonoverlapping(
                    config.user_comm_buffer as *const u8,
                    config.user_comm_buffer_internal as *mut u8,
                    config.user_comm_buffer_size as usize,
                );
            }
            log::trace!(
                "Copied {} bytes from user buffer 0x{:x} to internal 0x{:x}",
                config.user_comm_buffer_size,
                config.user_comm_buffer,
                config.user_comm_buffer_internal
            );
        }

        // Invoke the demoted user entry point with:
        //   arg1: UserCommandType::UserRequest (command type)
        //   arg2: supv_to_user_buffer (pointer to EfiMmEntryContext + MmCommBufferStatus)
        //   arg3: sizeof(EfiMmEntryContext) (size of the context portion)
        let ret = unsafe {
            invoke_demoted_routine(
                cpu_index,
                user_entry,
                cpl3_stack,
                3,
                UserCommandType::UserRequest as u64,
                config.supv_to_user_buffer,
                context_size as u64,
            )
        };
        log::info!("Returned from user request with value: 0x{}", ret);

        // Copy the response from the internal buffer back to the user buffer
        // SAFETY: Buffers are provided by MM IPL and are guaranteed valid
        if sync_mmi != 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    config.user_comm_buffer_internal as *const u8,
                    config.user_comm_buffer as *mut u8,
                    config.user_comm_buffer_size as usize,
                );
            }
        }

        // Read the updated MmCommBufferStatus back from the supervisor-to-user buffer
        // (the user may have modified return_status and return_buffer_size)
        // SAFETY: supv_to_user_buffer is valid and the status is at offset context_size
        let returned_status = unsafe {
            core::ptr::read(
                (config.supv_to_user_buffer as *const u8).add(context_size) as *const MmCommBufferStatus,
            )
        };

        // Write the returned status back to the supervisor's status buffer, clearing
        // is_comm_buffer_valid to indicate processing is complete
        // SAFETY: status_buffer is valid and writable
        unsafe {
            let status_ptr = config.status_buffer as *mut MmCommBufferStatus;
            let mut final_status = returned_status;
            final_status.is_comm_buffer_valid = 0;
            core::ptr::write_volatile(status_ptr, final_status);
        }
    }

    /// Process a request targeting the Supervisor.
    ///
    /// Parses the [`EfiMmCommunicateHeader`] from the supervisor communication buffer,
    /// matches the header GUID against the [`SUPERVISOR_MMI_HANDLERS`] distributed slice,
    /// and invokes the first matching handler. Handlers are registered at build time,
    /// allowing platforms to link in additional handlers without modifying the core.
    ///
    /// ## Dispatch Flow
    ///
    /// 1. Zero the internal buffer and copy the external supervisor buffer into it
    /// 2. Parse the `EfiMmCommunicateHeader` (GUID + message length) from the internal buffer
    /// 3. Validate message length does not exceed the buffer size
    /// 4. Iterate [`SUPERVISOR_MMI_HANDLERS`] to find a handler matching the header GUID
    /// 5. Call the handler with a pointer to the data payload and mutable size
    /// 6. Update the status buffer with return status and total response size
    /// 7. Copy the internal buffer back to the external buffer
    fn process_supervisor_request(&self, config: &CommBufferConfig, status: &MmCommBufferStatus, cpu_index: usize) {
        log::trace!("Processing Supervisor request on CPU {}...", cpu_index);

        // Validate buffers
        if config.supv_comm_buffer == 0 || config.supv_comm_buffer_internal == 0 {
            log::error!("Supervisor communication buffer not configured");
            return;
        }

        let buffer_size = config.supv_comm_buffer_size as usize;

        // Zero the internal buffer then copy the external supervisor buffer into it
        // SAFETY: Buffers are provided by MM IPL and are guaranteed valid and non-overlapping
        unsafe {
            core::ptr::write_bytes(config.supv_comm_buffer_internal as *mut u8, 0, buffer_size);
            core::ptr::copy_nonoverlapping(
                config.supv_comm_buffer as *const u8,
                config.supv_comm_buffer_internal as *mut u8,
                buffer_size,
            );
        }

        // Parse the EfiMmCommunicateHeader from the internal buffer
        if buffer_size < EfiMmCommunicateHeader::HEADER_SIZE {
            log::error!(
                "Supervisor buffer too small for communicate header: {} < {}",
                buffer_size,
                EfiMmCommunicateHeader::HEADER_SIZE
            );
            self.write_supv_status(config, status, efi::Status::BAD_BUFFER_SIZE, 0);
            return;
        }

        // SAFETY: We verified the buffer is large enough for the header.
        // The header is packed so we use read_unaligned.
        let header = unsafe {
            core::ptr::read_unaligned(config.supv_comm_buffer_internal as *const EfiMmCommunicateHeader)
        };

        let message_length = header.message_length as usize;

        // Validate message length doesn't exceed the buffer
        if message_length > buffer_size.saturating_sub(EfiMmCommunicateHeader::HEADER_SIZE) {
            log::error!(
                "Message length 0x{:x} exceeds available buffer space 0x{:x}",
                message_length,
                buffer_size - EfiMmCommunicateHeader::HEADER_SIZE
            );
            self.write_supv_status(config, status, efi::Status::BAD_BUFFER_SIZE, 0);
            return;
        }

        // Compute pointer to the data payload (after the header)
        let data_ptr = unsafe {
            (config.supv_comm_buffer_internal as *mut u8).add(EfiMmCommunicateHeader::HEADER_SIZE)
        };
        let mut data_size = message_length;

        // Dispatch: iterate the SUPERVISOR_MMI_HANDLERS distributed slice to find a match
        let handler_guid = header.header_guid;
        let mut dispatch_status = efi::Status::NOT_FOUND;

        for handler in SUPERVISOR_MMI_HANDLERS.iter() {
            if handler.handler_guid == handler_guid {
                log::trace!(
                    "Dispatching supervisor request to handler '{}' (GUID: {:?})",
                    handler.name,
                    handler.handler_guid
                );
                dispatch_status = (handler.handle)(data_ptr, &mut data_size);
                break;
            }
        }

        if dispatch_status == efi::Status::NOT_FOUND {
            log::warn!(
                "No handler found for supervisor request GUID: {:?}",
                handler_guid
            );
        }

        // Compute the total response size (header + data) for the copy-back
        let total_response_size = data_size + EfiMmCommunicateHeader::HEADER_SIZE;

        // Copy the (possibly modified) internal buffer back to the external buffer
        if total_response_size <= buffer_size {
            // SAFETY: Both buffers are valid and total_response_size is within bounds
            unsafe {
                core::ptr::copy_nonoverlapping(
                    config.supv_comm_buffer_internal as *const u8,
                    config.supv_comm_buffer as *mut u8,
                    total_response_size,
                );
            }
        } else {
            log::error!(
                "Response size 0x{:x} exceeds buffer capacity 0x{:x}",
                total_response_size,
                buffer_size
            );
        }
        log::info!(
            "Copied {} bytes from internal buffer 0x{:x} back to external 0x{:x}",
            total_response_size,
            config.supv_comm_buffer_internal,
            config.supv_comm_buffer
        );

        // Update the status buffer with return status and response size
        let return_status = if dispatch_status == efi::Status::SUCCESS {
            efi::Status::SUCCESS
        } else {
            efi::Status::NOT_FOUND
        };
        self.write_supv_status(config, status, return_status, total_response_size as u64);
    }

    /// Write the supervisor status buffer after processing a supervisor request.
    ///
    /// Clears `is_comm_buffer_valid` and `talk_to_supervisor`, sets return status and size.
    fn write_supv_status(
        &self,
        config: &CommBufferConfig,
        _status: &MmCommBufferStatus,
        return_status: efi::Status,
        return_buffer_size: u64,
    ) {
        // SAFETY: status_buffer is valid and writable, set up by MM IPL
        unsafe {
            let status_ptr = config.status_buffer as *mut MmCommBufferStatus;
            let updated = MmCommBufferStatus {
                is_comm_buffer_valid: 0,
                talk_to_supervisor: 0,
                return_status: return_status.as_usize() as u64,
                return_buffer_size,
            };
            core::ptr::write_volatile(status_ptr, updated);
            // Dump the content from the status_ptr
            let dumped_status = core::ptr::read_volatile(status_ptr);
            log::info!("written to supervisor status buffer at 0x{:x}", status_ptr as usize);
            log::info!(
                "Updated supervisor status buffer: is_comm_buffer_valid={}, talk_to_supervisor={}, return_status=0x{:x}, return_buffer_size=0x{:x}",
                dumped_status.is_comm_buffer_valid,
                dumped_status.talk_to_supervisor,
                dumped_status.return_status,
                dumped_status.return_buffer_size
            );
        }
    }

    /// The holding pen for APs.
    ///
    /// APs wait here, polling their mailbox for commands from the BSP.
    /// The loop exits when the AP receives a `Return` command.
    fn ap_holding_pen(&'static self, cpu_id: u32) {
        log::trace!("AP (CPU {}) in holding pen, polling mailbox...", cpu_id);

        loop {
            // Check mailbox for commands
            if let Some(command) = self.mailbox_manager.check_mailbox(cpu_id) {
                log::trace!("AP (CPU {}) received command: {:?}", cpu_id, command);

                // Execute the command
                let response = self.execute_ap_command(cpu_id, &command);

                // Post the response
                self.mailbox_manager.post_response(cpu_id, response);

                // Break out of the holding pen on Return
                if matches!(command, ApCommand::Return) {
                    log::trace!("AP (CPU {}) exiting holding pen", cpu_id);
                    break;
                }
            }
        }
    }

    /// Execute a command received by an AP.
    fn execute_ap_command(&self, cpu_id: u32, command: &ApCommand) -> ApResponse {
        match *command {
            ApCommand::RunProcedure { procedure, argument } => {
                self.run_procedure_on_ap(cpu_id, procedure, argument)
            }
            ApCommand::Return => {
                log::trace!("AP (CPU {}) received return command", cpu_id);
                ApResponse::Success
            }
        }
    }

    /// Run a procedure on an AP, demoting to user mode if the procedure is in user-owned range.
    ///
    /// This is the AP-side handler for `ApCommand::RunProcedure`. It mirrors the C
    /// `ProcedureWrapper` logic: inspects the procedure pointer ownership and either
    /// calls it directly (supervisor-owned) or demotes to Ring 3 (user-owned).
    fn run_procedure_on_ap(&self, cpu_id: u32, procedure: u64, argument: u64) -> ApResponse {
        log::trace!(
            "AP (CPU {}) running procedure 0x{:x} with arg 0x{:x}",
            cpu_id, procedure, argument
        );

        // Determine if the procedure is in user-owned (Ring 3) range by querying the
        // page table via the centralized helper.
        let is_user_range = match query_address_ownership(procedure, core::mem::size_of::<usize>() as u64) {
            Some(PageOwnership::User) => true,
            Some(PageOwnership::Supervisor) => false,
            None => {
                log::error!(
                    "AP (CPU {}) failed to query ownership for 0x{:x} (unmapped or page table not ready)",
                    cpu_id, procedure
                );
                return ApResponse::Error(efi::Status::DEVICE_ERROR.as_usize() as u32);
            }
        };

        if is_user_range {
            // Resolve the cpu_index (slot index) for this APIC ID
            let cpu_index = match self.cpu_manager.find_cpu_index(cpu_id) {
                Some(idx) => idx,
                None => {
                    log::error!("AP (CPU {}) has no registered slot, cannot demote", cpu_id);
                    return ApResponse::Error(efi::Status::DEVICE_ERROR.as_usize() as u32);
                }
            };

            // Get the CPL3 stack for this CPU
            let cpl3_stack = match self.syscall_interface.get_cpl3_stack(cpu_index) {
                Ok(stack) => stack,
                Err(e) => {
                    log::error!("AP (CPU {}) failed to get CPL3 stack: {:?}", cpu_id, e);
                    return ApResponse::Error(efi::Status::DEVICE_ERROR.as_usize() as u32);
                }
            };

            let user_entry = match USER_ENTRY_POINT.get() {
                Some(&entry) if entry != 0 => entry,
                _ => {
                    log::error!("User entry point not configured, cannot demote AP (CPU {})", cpu_id);
                    return ApResponse::Error(efi::Status::DEVICE_ERROR.as_usize() as u32);
                }
            };

            // Demote to user mode and call the procedure
            // The procedure signature is: void (EFIAPI *)(void *ProcedureArgument)
            log::trace!(
                "AP (CPU {}) demoting to user: proc=0x{:x}, stack=0x{:x}, arg=0x{:x}",
                cpu_id, procedure, cpl3_stack, argument
            );

            let _ret = unsafe {
                invoke_demoted_routine(
                    cpu_index,
                    user_entry,
                    cpl3_stack,
                    3,
                    UserCommandType::UserApProcedure as u64,
                    procedure,
                    argument,
                )
            };

            log::trace!("AP (CPU {}) returned from demoted procedure: 0x{:x}", cpu_id, _ret);
            ApResponse::Success
        } else {
            // Supervisor-owned: call directly in Ring 0
            log::trace!("AP (CPU {}) calling supervisor procedure directly at 0x{:x}", cpu_id, procedure);

            // SAFETY: The BSP validated the procedure pointer before dispatching.
            // The procedure follows the EFI AP_PROCEDURE calling convention.
            type EfiApProcedure = unsafe extern "efiapi" fn(*mut core::ffi::c_void);
            let proc_fn: EfiApProcedure = unsafe { core::mem::transmute(procedure) };
            unsafe { proc_fn(argument as *mut core::ffi::c_void) };

            ApResponse::Success
        }
    }

    /// Type-erased trampoline for AP startup, called from the syscall dispatcher.
    ///
    /// This function is monomorphized for the concrete `P: PlatformInfo` type
    /// and stored as a `fn(u64, u64, u64) -> u64` in [`AP_STARTUP_FN`].
    ///
    /// # Arguments
    ///
    /// * `cpu_index` - The EFI processor index (slot index) of the target AP
    /// * `procedure` - The procedure function pointer to execute on the AP
    /// * `argument` - The argument to pass to the procedure
    ///
    /// # Returns
    ///
    /// 0 on success, or an EFI status code (as u64) on failure.
    fn start_ap_procedure_trampoline(cpu_index: u64, procedure: u64, argument: u64) -> u64 {
        let core = Self::instance();
        core.start_ap_procedure(cpu_index, procedure, argument)
    }

    /// Validate and dispatch a procedure to a specific AP.
    ///
    /// Performs validation checks similar to the C `InternalSmmStartupThisAp`:
    /// 1. CPU index is within range of registered CPUs
    /// 2. CPU at that index is present (registered)
    /// 3. CPU is not the BSP
    /// 4. Procedure pointer is non-null
    /// 5. Sends the command via the mailbox (fails if AP is busy)
    /// 6. Waits for the AP to complete (blocking)
    fn start_ap_procedure(&self, cpu_index: u64, procedure: u64, argument: u64) -> u64 {
        let cpu_index = cpu_index as usize;

        // 1. Validate CPU index is within registered count
        let registered = self.cpu_manager.registered_count();
        if cpu_index >= registered {
            log::error!(
                "START_AP: CpuIndex({}) >= registered_count({})",
                cpu_index, registered
            );
            return efi::Status::INVALID_PARAMETER.as_usize() as u64;
        }

        // 2. Look up the APIC ID for this index
        let cpu_id = match self.cpu_manager.get_cpu_id_by_index(cpu_index) {
            Some(id) => id,
            None => {
                log::error!("START_AP: CpuIndex({}) has no registered CPU", cpu_index);
                return efi::Status::INVALID_PARAMETER.as_usize() as u64;
            }
        };

        // 3. Check that the target is not the BSP
        if self.cpu_manager.is_bsp(cpu_id) {
            log::error!("START_AP: CpuIndex({}) is the BSP, cannot start as AP", cpu_index);
            return efi::Status::INVALID_PARAMETER.as_usize() as u64;
        }

        // 4. Validate procedure pointer is non-null
        if procedure == 0 {
            log::error!("START_AP: Null procedure pointer");
            return efi::Status::INVALID_PARAMETER.as_usize() as u64;
        }

        // 5. Send the RunProcedure command to the AP via mailbox
        //    This will fail if the AP's mailbox is not empty (AP is busy).
        let command = ApCommand::RunProcedure { procedure, argument };
        if let Err(()) = self.mailbox_manager.send_command(cpu_id, command) {
            log::error!(
                "START_AP: AP (CPU {}, index {}) is busy or mailbox unavailable",
                cpu_id, cpu_index
            );
            return efi::Status::INVALID_PARAMETER.as_usize() as u64;
        }

        log::trace!(
            "START_AP: Dispatched proc=0x{:x} arg=0x{:x} to CPU {} (index {})",
            procedure, argument, cpu_id, cpu_index
        );

        // 6. Wait for the AP to complete (blocking mode)
        //    Use a generous timeout (10 seconds = 10_000_000 microseconds)
        const AP_TIMEOUT_US: u64 = 10_000_000;
        match self.mailbox_manager.wait_response(cpu_id, AP_TIMEOUT_US) {
            Some(ApResponse::Success) => {
                log::trace!("START_AP: AP (CPU {}) completed successfully", cpu_id);
                efi::Status::SUCCESS.as_usize() as u64
            }
            Some(ApResponse::Error(code)) => {
                log::error!("START_AP: AP (CPU {}) returned error: 0x{:x}", cpu_id, code);
                code as u64
            }
            Some(ApResponse::Busy) => {
                log::error!("START_AP: AP (CPU {}) reported busy", cpu_id);
                efi::Status::NOT_READY.as_usize() as u64
            }
            Some(ApResponse::None) | None => {
                log::error!("START_AP: AP (CPU {}) timed out or no response", cpu_id);
                efi::Status::TIMEOUT.as_usize() as u64
            }
        }
    }

    /// Get the CPU manager.
    pub fn cpu_manager(&self) -> &CpuManager<{ P::MAX_CPU_COUNT }> {
        &self.cpu_manager
    }

    /// Get the mailbox manager.
    pub fn mailbox_manager(&self) -> &MailboxManager<{ P::MAX_CPU_COUNT }> {
        &self.mailbox_manager
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
