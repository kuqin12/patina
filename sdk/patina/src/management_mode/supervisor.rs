//! Shared type definitions for MM supervisor and user cores.
//!
//! This crate provides the communication structures and enumerations that define
//! the ABI between the supervisor (ring 0) and user (ring 3) MM modules.

// GUID for gMmSupervisorHobMemoryAllocModuleGuid
// { 0x3efafe72, 0x3dbf, 0x4341, { 0xad, 0x04, 0x1c, 0xb6, 0xe8, 0xb6, 0x8e, 0x5e }}
/// GUID used in MemoryAllocationModule HOBs to identify MM Supervisor module allocations.
pub const MM_SUPERVISOR_HOB_MEMORY_ALLOC_MODULE_GUID: crate::BinaryGuid =
    crate::BinaryGuid::from_string("3efafe72-3dbf-4341-ad04-1cb6e8b68e5e");

// GUID for gMmSupervisorUserGuid
// { 0x30d1cc3f, 0xc1db, 0x41ed, { 0xb1, 0x13, 0xab, 0xce, 0x21, 0xb0, 0x2b, 0xce }}
/// GUID identifying the MM Supervisor User module.
pub const MM_SUPERVISOR_USER_GUID: crate::BinaryGuid =
    crate::BinaryGuid::from_string("30d1cc3f-c1db-41ed-b113-abce21b02bce");

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
    /// Read a CPU save-state field. Arg1: CPU index, Arg2: [`SaveStateType`],
    /// Arg3: pointer to an 8-byte user output buffer. Returns `EFI_STATUS` in
    /// RAX and writes the field value to the buffer on success.
    SaveStateRead = 0x0007,
    /// Maximum value for legacy syscall indices
    LegacyMax = 0xFFFF,
    /// Allocate Pages - Arg1: alloc_type, Arg2: mem_type, Arg3: page_count
    AllocPage = 0x10004,
    /// Free Pages - Arg1: address, Arg2: page_count
    FreePage = 0x10005,
    /// Start AP Procedure - Arg1: procedure, Arg2: CPU index, Arg3: argument
    StartApProc = 0x10006,
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

/// The save-state field a [`SyscallIndex::SaveStateRead`] request selects.
///
/// This is the `arg2` of the save-state read syscall. The supervisor reads only
/// these individual fields; assembling the composite
/// `EFI_MM_SAVE_STATE_IO_INFO` (for `EFI_MM_SAVE_STATE_REGISTER_IO`) is the
/// caller's responsibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum SaveStateType {
    /// The trapping CPU's ID (APIC ID). Informational; not policy-gated.
    ProcessorId = 0,
    /// The `RAX` register. Policy-gated; readable only when the trap was an I/O write.
    Rax = 1,
    /// The I/O trap descriptor (port, width, direction) of the trapping
    /// instruction, packed as described by [`IO_TRAP_PORT_SHIFT`] and friends.
    /// Does **not** include the I/O data (read `Rax` separately for that).
    IoTrap = 2,
}

/// Bit position of the I/O port (`u16`) in a packed [`SaveStateType::IoTrap`] value.
pub const IO_TRAP_PORT_SHIFT: u32 = 0;
/// Bit position of the I/O width (EFI `EFI_MM_SAVE_STATE_IO_WIDTH`, one byte) in
/// a packed [`SaveStateType::IoTrap`] value.
pub const IO_TRAP_WIDTH_SHIFT: u32 = 16;
/// Bit position of the I/O type (EFI `EFI_MM_SAVE_STATE_IO_TYPE`, one byte) in a
/// packed [`SaveStateType::IoTrap`] value.
pub const IO_TRAP_TYPE_SHIFT: u32 = 24;

impl SaveStateType {
    /// Creates a `SaveStateType` from a raw `u64` value.
    pub fn from_u64(value: u64) -> Option<Self> {
        match value {
            0 => Some(Self::ProcessorId),
            1 => Some(Self::Rax),
            2 => Some(Self::IoTrap),
            _ => None,
        }
    }

    /// Returns the raw `u64` value of this save-state type.
    pub fn as_u64(self) -> u64 {
        self as u64
    }
}
