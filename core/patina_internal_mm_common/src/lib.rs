//! Shared type definitions for MM supervisor and user cores.
//!
//! This crate provides the communication structures and enumerations that define
//! the ABI between the supervisor (ring 0) and user (ring 3) MM modules.

#![no_std]

use core::mem;

use r_efi::efi;

// =============================================================================
// GUIDs
// =============================================================================

/// GUID for the MM communication buffer HOB (`gMmCommBufferHobGuid`).
///
/// `{ 0x6c2a2520, 0x0131, 0x4aee, { 0xa7, 0x50, 0xcc, 0x38, 0x4a, 0xac, 0xe8, 0xc6 } }`
pub const MM_COMM_BUFFER_HOB_GUID: efi::Guid = efi::Guid::from_fields(
    0x6c2a2520,
    0x0131,
    0x4aee,
    0xa7,
    0x50,
    &[0xcc, 0x38, 0x4a, 0xac, 0xe8, 0xc6],
);

// =============================================================================
// Communication Structures
// =============================================================================

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

/// MM Communication Buffer Status Structure.
///
/// Matches the C structure `MM_COMM_BUFFER_STATUS` from MmCommBuffer.h.
/// The supervisor writes this to indicate request validity and direction;
/// both supervisor and user core read/update it during request handling.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MmCommBufferStatus {
    /// Whether the data in the fixed MM communication buffer is valid when entering from non-MM to MM.
    pub is_comm_buffer_valid: u8,
    /// The channel used to communicate with MM (1 = Supervisor, 0 = User).
    pub talk_to_supervisor: u8,
    /// The return status when returning from MM to non-MM.
    pub return_status: u64,
    /// The size in bytes of the output buffer when returning from MM to non-MM.
    pub return_buffer_size: u64,
}

/// EFI_MM_COMMUNICATE_HEADER structure.
///
/// Communication buffer header used by the MM Communicate protocol.
/// The data payload immediately follows this header.
///
/// Layout:
/// - `header_guid`: 16 bytes — GUID identifying the handler
/// - `message_length`: 8 bytes — size of `Data` in bytes (does not include header size)
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
    pub const HEADER_SIZE: usize = mem::size_of::<Self>();
}

/// MM Common Buffer HOB Data Structure.
///
/// Describes the communication buffer region passed via HOB from PEI to MM.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct MmCommonBufferHobData {
    /// Physical start address of the common region.
    pub physical_start: u64,
    /// Number of pages in the communication buffer region.
    pub number_of_pages: u64,
    /// Pointer to `MmCommBufferStatus` structure.
    pub status_buffer: u64,
}

// =============================================================================
// Command Types
// =============================================================================

/// Command types passed from the supervisor to the user core via `invoke_demoted_routine`.
///
/// Discriminant values are part of the supervisor↔user ABI and must not change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum UserCommandType {
    /// Initialize the user core: walk HOBs, discover drivers, dispatch.
    StartUserCore = 0,
    /// Handle a runtime MMI request: parse communication buffer and dispatch handlers.
    UserRequest = 1,
    /// Execute a procedure on an AP.
    UserApProcedure = 2,
}

impl TryFrom<u64> for UserCommandType {
    type Error = u64;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(UserCommandType::StartUserCore),
            1 => Ok(UserCommandType::UserRequest),
            2 => Ok(UserCommandType::UserApProcedure),
            other => Err(other),
        }
    }
}

// =============================================================================
// Syscall Indices
// =============================================================================

/// Syscall indices for the MM Supervisor ↔ User Core syscall interface.
///
/// These match the definitions in SysCallLib.h and define the ABI used when
/// Ring 3 code issues a `syscall` instruction to the Ring 0 supervisor.
///
/// ## ABI
///
/// - RAX = call index ([`SyscallIndex`])
/// - RDX = arg1
/// - R8  = arg2
/// - R9  = arg3
///
/// On return:
/// - RAX = result value
/// - RDX = status (EFI_STATUS)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum SyscallIndex {
    /// Read MSR - Arg1: MSR index, Returns: MSR value
    RdMsr = 0x0000,
    /// Write MSR - Arg1: MSR index, Arg2: value
    WrMsr = 0x0001,
    /// CLI - Clear interrupts
    Cli = 0x0002,
    /// IO Read - Arg1: port, Arg2: width
    IoRead = 0x0003,
    /// IO Write - Arg1: port, Arg2: width, Arg3: value
    IoWrite = 0x0004,
    /// WBINVD - Write back and invalidate cache
    Wbinvd = 0x0005,
    /// HLT - Halt processor
    Hlt = 0x0006,
    /// Save State Read - Arg1: register, Arg2: CPU index
    SaveStateRead = 0x0007,
    /// Maximum value for legacy syscall indices
    LegacyMax = 0xFFFF,
    /// Allocate Pages - Arg1: alloc_type, Arg2: mem_type, Arg3: page_count
    AllocPage = 0x10004,
    /// Free Pages - Arg1: address, Arg2: page_count
    FreePage = 0x10005,
    /// Start AP Procedure - Arg1: procedure, Arg2: CPU index, Arg3: argument
    StartApProc = 0x10006,
    /// Save state read with extended support - Arg1: width, Arg2: buffer pointer
    SaveStateRead2 = 0x10021,
    /// MM memory unblocked - Arg1: address, Arg2: size
    MmMemoryUnblocked = 0x10022,
    /// MM is communication buffer - Arg1: address, Arg2: size
    MmIsCommBuffer = 0x10023,
}

impl SyscallIndex {
    /// Creates a `SyscallIndex` from a raw `u64` value.
    pub fn from_u64(value: u64) -> Option<Self> {
        match value {
            0x0000 => Some(Self::RdMsr),
            0x0001 => Some(Self::WrMsr),
            0x0002 => Some(Self::Cli),
            0x0003 => Some(Self::IoRead),
            0x0004 => Some(Self::IoWrite),
            0x0005 => Some(Self::Wbinvd),
            0x0006 => Some(Self::Hlt),
            0x0007 => Some(Self::SaveStateRead),
            0xFFFF => Some(Self::LegacyMax),
            0x10004 => Some(Self::AllocPage),
            0x10005 => Some(Self::FreePage),
            0x10006 => Some(Self::StartApProc),
            0x10021 => Some(Self::SaveStateRead2),
            0x10022 => Some(Self::MmMemoryUnblocked),
            0x10023 => Some(Self::MmIsCommBuffer),
            _ => None,
        }
    }

    /// Returns the raw `u64` value of this syscall index.
    pub fn as_u64(self) -> u64 {
        self as u64
    }
}
