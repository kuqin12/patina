//! Syscall Dispatcher
//!
//! This module handles syscall requests from Ring 3 code. When Ring 3 code
//! executes the `syscall` instruction, the CPU jumps to the address in
//! MSR_IA32_LSTAR (our SyscallCenter assembly stub), which then calls into
//! this dispatcher.
//!
//! ## Syscall Interface
//!
//! The syscall uses a custom calling convention:
//! - RAX: Call index (SyscallIndex)
//! - RDX: Argument 1
//! - R8:  Argument 2
//! - R9:  Argument 3
//! - RCX: Caller return address (set by syscall instruction)
//! - R11: RFLAGS (set by syscall instruction)
//!
//! The dispatcher validates the request and dispatches to the appropriate handler.
//!
//! ## Security
//!
//! All syscall handlers must validate their arguments and check that any
//! memory pointers are within valid user-accessible regions.
//!

use core::sync::atomic::{AtomicBool, Ordering};
use core::arch::{global_asm, asm};
use r_efi::efi::{AllocateType, ALLOCATE_ANY_PAGES, MemoryType, RUNTIME_SERVICES_DATA};

use patina_mm_policy::{AccessType, IoWidth, Instruction};

use super::{PrivilegeError, PrivilegeResult};
use crate::{COMM_BUFFER_CONFIG, POLICY_GATE, UNBLOCKED_MEMORY_TRACKER, PageOwnership, query_address_ownership};

global_asm!(include_str!("syscall_entry.asm"));

// ============================================================================
// EFI_MM_IO_WIDTH values (from EFI spec)
// ============================================================================

/// MM_IO_UINT8 - 8-bit I/O access width.
const MM_IO_UINT8: u64 = 0;
/// MM_IO_UINT16 - 16-bit I/O access width.
const MM_IO_UINT16: u64 = 1;
/// MM_IO_UINT32 - 32-bit I/O access width.
const MM_IO_UINT32: u64 = 2;

/// Converts an EFI_MM_IO_WIDTH enum value to our [`IoWidth`] type.
///
/// The EFI spec defines: MM_IO_UINT8=0, MM_IO_UINT16=1, MM_IO_UINT32=2.
fn efi_io_width_to_io_width(width: u64) -> Option<IoWidth> {
    match width {
        MM_IO_UINT8 => Some(IoWidth::Byte),
        MM_IO_UINT16 => Some(IoWidth::Word),
        MM_IO_UINT32 => Some(IoWidth::Dword),
        _ => None,
    }
}

// ============================================================================
// Hardware Operation Helpers
// ============================================================================

/// Reads an 8-bit value from an I/O port.
///
/// # Safety
///
/// The caller must ensure the port address is valid and access is allowed by policy.
#[inline]
unsafe fn io_read_u8(port: u16) -> u8 {
    let value: u8;
    unsafe {
        asm!("in al, dx", out("al") value, in("dx") port, options(nomem, nostack));
    }
    value
}

/// Reads a 16-bit value from an I/O port.
///
/// # Safety
///
/// The caller must ensure the port address is valid and access is allowed by policy.
#[inline]
unsafe fn io_read_u16(port: u16) -> u16 {
    let value: u16;
    unsafe {
        asm!("in ax, dx", out("ax") value, in("dx") port, options(nomem, nostack));
    }
    value
}

/// Reads a 32-bit value from an I/O port.
///
/// # Safety
///
/// The caller must ensure the port address is valid and access is allowed by policy.
#[inline]
unsafe fn io_read_u32(port: u16) -> u32 {
    let value: u32;
    unsafe {
        asm!("in eax, dx", out("eax") value, in("dx") port, options(nomem, nostack));
    }
    value
}

/// Writes an 8-bit value to an I/O port.
///
/// # Safety
///
/// The caller must ensure the port address is valid and access is allowed by policy.
#[inline]
unsafe fn io_write_u8(port: u16, value: u8) {
    unsafe {
        asm!("out dx, al", in("dx") port, in("al") value, options(nomem, nostack));
    }
}

/// Writes a 16-bit value to an I/O port.
///
/// # Safety
///
/// The caller must ensure the port address is valid and access is allowed by policy.
#[inline]
unsafe fn io_write_u16(port: u16, value: u16) {
    unsafe {
        asm!("out dx, ax", in("dx") port, in("ax") value, options(nomem, nostack));
    }
}

/// Writes a 32-bit value to an I/O port.
///
/// # Safety
///
/// The caller must ensure the port address is valid and access is allowed by policy.
#[inline]
unsafe fn io_write_u32(port: u16, value: u32) {
    unsafe {
        asm!("out dx, eax", in("dx") port, in("eax") value, options(nomem, nostack));
    }
}

// ============================================================================
// Syscall Indices
// ============================================================================

/// Syscall indices for the MM Supervisor syscall interface.
///
/// These match the definitions in SysCallLib.h.
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
    /// Register Handler Jump Pointer - Unsupported
    // RegHandlerJump = 0x10000,
    /// Install configuration table - Unsupported
    // InstallConfigTable = 0x10001,
    /// Allocate pool: unsupported
    // AllocPool = 0x10002,
    /// Free pool: unsupported
    // FreePool = 0x10003,
    /// Allocate Pages - Arg1: memory type, Arg2: page count, Arg3: address ptr
    AllocPage = 0x10004,
    /// Free Pages - Arg1: address, Arg2: page count
    FreePage = 0x10005,
    /// Start AP Procedure - Arg1: procedure, Arg2: CPU index, Arg3: argument
    StartApProc = 0x10006,
    /// Register MMI handler jump pointer - Unsupported
    // RegMmiHandlerJump = 0x10007,
    /// Unregister MMI handler - Unsupported
    // UnregMmiHandlerJump = 0x10018,
    /// Set CPL3 Page Table - Unsupported
    // SetCpl3Table = 0x10019,
    /// Install protocol - Unsupported
    // InstallProtocol = 0x1001A,
    /// Query hobs - Unsupported
    // QueryHobs = 0x1001B,
    /// Error Report Jump - Unsupported, this is moved to be handled by the user core
    // ErrReportJump = 0x1001C,
    /// MMI handler profile register - Unsupported
    // RegMmiProfile1 = 0x1001D,
    /// MMI handler profile register - Unsupported
    // RegMmiProfile2 = 0x1001E,
    /// MMI handler profile unregister - Unsupported
    // UnregMmiProfile1 = 0x1001F,
    /// MMI handler profile unregister - Unsupported
    // UnregMmiProfile2 = 0x10020,
    /// Save state read with extended support - Arg1: width, Arg2: buffer pointer
    SaveStateRead2 = 0x10021,
    /// MM memory unblocked - Arg1: address, Arg2: size
    MmMemoryUnblocked = 0x10022,
    /// MM memory is communication buffer - Arg1: address, Arg2: size
    MmIsCommBuffer = 0x10023,
}

impl SyscallIndex {
    /// Creates a SyscallIndex from a raw u64 value.
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

    /// Returns the raw u64 value of this syscall index.
    pub fn as_u64(self) -> u64 {
        self as u64
    }
}

// ============================================================================
// Syscall Result
// ============================================================================

/// Result of a syscall operation.
#[derive(Debug, Clone, Copy)]
pub struct SyscallResult {
    /// Return value (in RAX on return to Ring 3).
    pub value: u64,
    /// Status code (EFI_STATUS compatible).
    pub status: u64,
}

impl SyscallResult {
    /// Creates a successful result with a value.
    pub const fn success(value: u64) -> Self {
        Self { value, status: 0 }
    }

    /// Creates an error result.
    pub const fn error(status: u64) -> Self {
        Self { value: 0, status }
    }

    /// EFI_SUCCESS
    pub const EFI_SUCCESS: u64 = 0;
    /// EFI_INVALID_PARAMETER
    pub const EFI_INVALID_PARAMETER: u64 = 0x8000_0000_0000_0002;
    /// EFI_UNSUPPORTED
    pub const EFI_UNSUPPORTED: u64 = 0x8000_0000_0000_0003;
    /// EFI_ACCESS_DENIED
    pub const EFI_ACCESS_DENIED: u64 = 0x8000_0000_0000_000F;
    /// EFI_NOT_READY
    pub const EFI_NOT_READY: u64 = 0x8000_0000_0000_0006;
    /// EFI_OUT_OF_RESOURCES
    pub const EFI_OUT_OF_RESOURCES: u64 = 0x8000_0000_0000_0009;
    /// EFI_SECURITY_VIOLATION
    pub const EFI_SECURITY_VIOLATION: u64 = 0x8000_0000_0000_001A;
}

// ============================================================================
// Syscall Context
// ============================================================================

/// Context for a syscall invocation.
#[derive(Debug, Clone, Copy)]
pub struct SyscallContext {
    /// The syscall index (from RAX).
    pub call_index: u64,
    /// First argument (from RDX).
    pub arg1: u64,
    /// Second argument (from R8).
    pub arg2: u64,
    /// Third argument (from R9).
    pub arg3: u64,
    /// Caller return address (from RCX, set by syscall instruction).
    pub caller_addr: u64,
    /// Ring 3 stack pointer at syscall entry.
    pub ring3_stack_ptr: u64,
}

// ============================================================================
// Syscall Dispatcher
// ============================================================================

/// The syscall dispatcher handles incoming syscalls from Ring 3.
pub struct SyscallDispatcher {
    /// Whether the dispatcher has been initialized.
    initialized: AtomicBool,
}

impl SyscallDispatcher {
    /// Creates a new syscall dispatcher.
    pub const fn new() -> Self {
        Self {
            initialized: AtomicBool::new(false),
        }
    }

    /// Initializes the syscall dispatcher.
    pub fn init(&self) -> PrivilegeResult<()> {
        if self.initialized.swap(true, Ordering::SeqCst) {
            return Err(PrivilegeError::AlreadyInitialized);
        }

        log::info!("SyscallDispatcher initialized");
        Ok(())
    }

    /// Checks if the dispatcher is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    /// Dispatches a syscall.
    ///
    /// This is the main entry point called from the assembly syscall handler.
    /// It validates the syscall index and dispatches to the appropriate handler.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The syscall context containing all arguments
    ///
    /// # Returns
    ///
    /// The result to be returned to Ring 3 in RAX.
    pub fn dispatch(&self, ctx: &SyscallContext) -> SyscallResult {
        // Parse the syscall index
        let index = match SyscallIndex::from_u64(ctx.call_index) {
            Some(idx) => idx,
            None => {
                log::warn!("Unknown syscall index: 0x{:x}", ctx.call_index);
                return SyscallResult::error(SyscallResult::EFI_UNSUPPORTED);
            }
        };

        log::trace!(
            "Syscall: {:?} (0x{:x}), args: 0x{:x}, 0x{:x}, 0x{:x}",
            index,
            ctx.call_index,
            ctx.arg1,
            ctx.arg2,
            ctx.arg3
        );

        // Dispatch to the appropriate handler
        match index {
            SyscallIndex::RdMsr => self.handle_rdmsr(ctx),
            SyscallIndex::WrMsr => self.handle_wrmsr(ctx),
            SyscallIndex::Cli => self.handle_cli(ctx),
            SyscallIndex::IoRead => self.handle_io_read(ctx),
            SyscallIndex::IoWrite => self.handle_io_write(ctx),
            SyscallIndex::Wbinvd => self.handle_wbinvd(ctx),
            SyscallIndex::Hlt => self.handle_hlt(ctx),
            SyscallIndex::SaveStateRead => self.handle_save_state_read(ctx),
            SyscallIndex::LegacyMax => panic!("Invalid syscall index: LegacyMax is not a real syscall"),
            SyscallIndex::AllocPage => self.handle_alloc_page(ctx),
            SyscallIndex::FreePage => self.handle_free_page(ctx),
            SyscallIndex::StartApProc => self.handle_start_ap_proc(ctx),
            SyscallIndex::SaveStateRead2 => self.handle_save_state_read2(ctx),
            SyscallIndex::MmMemoryUnblocked => self.handle_mm_memory_unblocked(ctx),
            SyscallIndex::MmIsCommBuffer => self.handle_mm_is_comm_buffer(ctx),
        }
    }

    // ========================================================================
    // Syscall Handlers
    // ========================================================================

    /// Handles MSR read syscall.
    ///
    /// Validates the MSR read against firmware policy, then executes `rdmsr`.
    /// - Arg1: MSR index
    /// - Returns: MSR value in result.value
    fn handle_rdmsr(&self, ctx: &SyscallContext) -> SyscallResult {
        let msr_index = ctx.arg1 as u32;
        log::trace!("RDMSR: msr=0x{:x}", msr_index);

        // Validate against policy
        let gate = match POLICY_GATE.get() {
            Some(g) => g,
            None => {
                log::error!("RDMSR: Policy gate not initialized");
                return SyscallResult::error(SyscallResult::EFI_NOT_READY);
            }
        };

        if let Err(e) = gate.is_msr_allowed(msr_index, AccessType::Read) {
            log::error!("RDMSR: MSR 0x{:x} blocked by policy: {:?}", msr_index, e);
            return SyscallResult::error(SyscallResult::EFI_ACCESS_DENIED);
        }

        // Policy allows - execute the MSR read
        let value = unsafe { crate::cpu::read_msr(msr_index) }.unwrap_or_else(|e| {
            log::error!("RDMSR: rdmsr failed: {}", e);
            0
        });
        log::debug!("RDMSR: MSR 0x{:x} = 0x{:x}", msr_index, value);
        SyscallResult::success(value)
    }

    /// Handles MSR write syscall.
    ///
    /// Validates the MSR write against firmware policy, then executes `wrmsr`.
    /// - Arg1: MSR index
    /// - Arg2: Value to write
    fn handle_wrmsr(&self, ctx: &SyscallContext) -> SyscallResult {
        let msr_index = ctx.arg1 as u32;
        let value = ctx.arg2;
        log::trace!("WRMSR: msr=0x{:x}, value=0x{:x}", msr_index, value);

        // Validate against policy
        let gate = match POLICY_GATE.get() {
            Some(g) => g,
            None => {
                log::error!("WRMSR: Policy gate not initialized");
                return SyscallResult::error(SyscallResult::EFI_NOT_READY);
            }
        };

        if let Err(e) = gate.is_msr_allowed(msr_index, AccessType::Write) {
            log::error!("WRMSR: MSR 0x{:x} blocked by policy: {:?}", msr_index, e);
            return SyscallResult::error(SyscallResult::EFI_ACCESS_DENIED);
        }

        // Policy allows - execute the MSR write
        if let Err(e) = unsafe { crate::cpu::write_msr(msr_index, value) } {
            log::error!("WRMSR: wrmsr failed: {}", e);
            return SyscallResult::error(SyscallResult::EFI_UNSUPPORTED);
        }
        log::debug!("WRMSR: MSR 0x{:x} written with 0x{:x}", msr_index, value);
        SyscallResult::success(0)
    }

    /// Handles CLI (clear interrupt flag) syscall.
    ///
    /// Validates the CLI instruction against firmware policy, then executes `cli`.
    fn handle_cli(&self, _ctx: &SyscallContext) -> SyscallResult {
        log::trace!("CLI");

        // Validate against policy
        let gate = match POLICY_GATE.get() {
            Some(g) => g,
            None => {
                log::error!("CLI: Policy gate not initialized");
                return SyscallResult::error(SyscallResult::EFI_NOT_READY);
            }
        };

        if let Err(e) = gate.is_instruction_allowed(Instruction::Cli) {
            log::error!("CLI: Instruction blocked by policy: {:?}", e);
            return SyscallResult::error(SyscallResult::EFI_ACCESS_DENIED);
        }

        // Policy allows - disable interrupts
        unsafe { asm!("cli", options(nomem, nostack)) };
        log::debug!("CLI: Interrupts disabled");
        SyscallResult::success(0)
    }

    /// Handles I/O port read syscall.
    ///
    /// Validates the I/O read against firmware policy, then executes the `in` instruction.
    /// - Arg1: I/O port address
    /// - Arg2: EFI_MM_IO_WIDTH (0=UINT8, 1=UINT16, 2=UINT32)
    /// - Returns: Value read from the port in result.value
    fn handle_io_read(&self, ctx: &SyscallContext) -> SyscallResult {
        let port = ctx.arg1;
        let efi_width = ctx.arg2;
        log::trace!("IO_READ: port=0x{:x}, width={}", port, efi_width);

        // Convert EFI_MM_IO_WIDTH to IoWidth
        let io_width = match efi_io_width_to_io_width(efi_width) {
            Some(w) => w,
            None => {
                log::error!("IO_READ: Invalid IO width: {}", efi_width);
                return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
            }
        };

        // Validate against policy
        let gate = match POLICY_GATE.get() {
            Some(g) => g,
            None => {
                log::error!("IO_READ: Policy gate not initialized");
                return SyscallResult::error(SyscallResult::EFI_NOT_READY);
            }
        };

        if let Err(e) = gate.is_io_allowed(port as u32, io_width, AccessType::Read) {
            log::error!("IO_READ: Port 0x{:x} width {:?} blocked by policy: {:?}", port, io_width, e);
            return SyscallResult::error(SyscallResult::EFI_ACCESS_DENIED);
        }

        // Policy allows - execute the I/O read
        let port_addr = port as u16;
        let value: u64 = unsafe {
            match efi_width {
                MM_IO_UINT8 => io_read_u8(port_addr) as u64,
                MM_IO_UINT16 => io_read_u16(port_addr) as u64,
                MM_IO_UINT32 => io_read_u32(port_addr) as u64,
                _ => unreachable!(), // Already validated above
            }
        };

        log::debug!("IO_READ: port=0x{:x} => 0x{:x}", port, value);
        SyscallResult::success(value)
    }

    /// Handles I/O port write syscall.
    ///
    /// Validates the I/O write against firmware policy, then executes the `out` instruction.
    /// - Arg1: I/O port address
    /// - Arg2: EFI_MM_IO_WIDTH (0=UINT8, 1=UINT16, 2=UINT32)
    /// - Arg3: Value to write
    fn handle_io_write(&self, ctx: &SyscallContext) -> SyscallResult {
        let port = ctx.arg1;
        let efi_width = ctx.arg2;
        let value = ctx.arg3;
        log::trace!("IO_WRITE: port=0x{:x}, width={}, value=0x{:x}", port, efi_width, value);

        // Convert EFI_MM_IO_WIDTH to IoWidth
        let io_width = match efi_io_width_to_io_width(efi_width) {
            Some(w) => w,
            None => {
                log::error!("IO_WRITE: Invalid IO width: {}", efi_width);
                return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
            }
        };

        // Validate against policy
        let gate = match POLICY_GATE.get() {
            Some(g) => g,
            None => {
                log::error!("IO_WRITE: Policy gate not initialized");
                return SyscallResult::error(SyscallResult::EFI_NOT_READY);
            }
        };

        if let Err(e) = gate.is_io_allowed(port as u32, io_width, AccessType::Write) {
            log::error!("IO_WRITE: Port 0x{:x} width {:?} blocked by policy: {:?}", port, io_width, e);
            return SyscallResult::error(SyscallResult::EFI_ACCESS_DENIED);
        }

        // Policy allows - execute the I/O write
        let port_addr = port as u16;
        unsafe {
            match efi_width {
                MM_IO_UINT8 => io_write_u8(port_addr, value as u8),
                MM_IO_UINT16 => io_write_u16(port_addr, value as u16),
                MM_IO_UINT32 => io_write_u32(port_addr, value as u32),
                _ => unreachable!(), // Already validated above
            }
        }

        log::debug!("IO_WRITE: port=0x{:x} <= 0x{:x}", port, value);
        SyscallResult::success(0)
    }

    /// Handles WBINVD (write-back and invalidate cache) syscall.
    ///
    /// Validates the WBINVD instruction against firmware policy, then executes `wbinvd`.
    fn handle_wbinvd(&self, _ctx: &SyscallContext) -> SyscallResult {
        log::trace!("WBINVD");

        // Validate against policy
        let gate = match POLICY_GATE.get() {
            Some(g) => g,
            None => {
                log::error!("WBINVD: Policy gate not initialized");
                return SyscallResult::error(SyscallResult::EFI_NOT_READY);
            }
        };

        if let Err(e) = gate.is_instruction_allowed(Instruction::Wbinvd) {
            log::error!("WBINVD: Instruction blocked by policy: {:?}", e);
            return SyscallResult::error(SyscallResult::EFI_ACCESS_DENIED);
        }

        // Policy allows - write back and invalidate cache
        unsafe { asm!("wbinvd", options(nomem, nostack)) };
        log::debug!("WBINVD: Cache written back and invalidated");
        SyscallResult::success(0)
    }

    /// Handles HLT (halt processor) syscall.
    ///
    /// Validates the HLT instruction against firmware policy, then executes `hlt`.
    fn handle_hlt(&self, _ctx: &SyscallContext) -> SyscallResult {
        log::trace!("HLT");

        // Validate against policy
        let gate = match POLICY_GATE.get() {
            Some(g) => g,
            None => {
                log::error!("HLT: Policy gate not initialized");
                return SyscallResult::error(SyscallResult::EFI_NOT_READY);
            }
        };

        if let Err(e) = gate.is_instruction_allowed(Instruction::Hlt) {
            log::error!("HLT: Instruction blocked by policy: {:?}", e);
            return SyscallResult::error(SyscallResult::EFI_ACCESS_DENIED);
        }

        // Policy allows - halt processor (sleep until next interrupt)
        unsafe { asm!("hlt", options(nomem, nostack)) };
        log::debug!("HLT: Processor halted and resumed");
        SyscallResult::success(0)
    }

    /// Handles save state read syscall (legacy).
    ///
    /// - Arg1: User MM CPU protocol pointer
    /// - Arg2: Register to be read (`EFI_MM_SAVE_STATE_REGISTER`)
    /// - Arg3: CPU index to read from
    fn handle_save_state_read(&self, ctx: &SyscallContext) -> SyscallResult {
        log::trace!(
            "SAVE_STATE_READ: protocol=0x{:x}, register={}, cpu={}",
            ctx.arg1,
            ctx.arg2,
            ctx.arg3
        );

        // Validate parameters
        if ctx.arg1 == 0 {
            log::error!("SAVE_STATE_READ: Null protocol pointer");
            return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
        }

        // Delegate to save state module Phase 1
        crate::save_state::save_state_read_phase1(ctx.arg1, ctx.arg2, ctx.arg3)
    }

    /// Handles page allocation syscall.
    ///
    /// - Arg1: Allocate type (EFI_ALLOCATE_TYPE)
    /// - Arg2: Memory type (must be EfiRuntimeServicesData)
    /// - Arg3: Page count
    /// - Returns: Allocated physical address in result.value
    fn handle_alloc_page(&self, ctx: &SyscallContext) -> SyscallResult {
        let alloc_type = ctx.arg1 as AllocateType;
        let mem_type = ctx.arg2 as MemoryType;
        let page_count = ctx.arg3;
        log::trace!("ALLOC_PAGE: alloc_type={}, mem_type={}, count={}", alloc_type, mem_type, page_count);

        // Only BSP can allocate pages (AP allocating involves page table updates)
        if !crate::is_bsp() {
            log::error!("ALLOC_PAGE: AP cannot allocate pages");
            return SyscallResult::error(SyscallResult::EFI_ACCESS_DENIED);
        }

        if mem_type != RUNTIME_SERVICES_DATA {
            log::error!("ALLOC_PAGE: Invalid memory type: {}", mem_type);
            return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
        }

        // Currently only AllocateAnyPages is supported by our page allocator
        if alloc_type != ALLOCATE_ANY_PAGES {
            log::error!("ALLOC_PAGE: Only AllocateAnyPages (0) is supported, got {}", alloc_type);
            return SyscallResult::error(SyscallResult::EFI_UNSUPPORTED);
        }

        if page_count == 0 {
            log::error!("ALLOC_PAGE: Zero page count");
            return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
        }

        // Allocate pages as User type (Ring 3 driver request)
        match crate::PAGE_ALLOCATOR.allocate_pages_with_type(
            page_count as usize,
            crate::mm_mem::AllocationType::User,
        ) {
            Ok(addr) => {
                log::info!("ALLOC_PAGE: Allocated {} page(s) at 0x{:x}", page_count, addr);
                SyscallResult::success(addr)
            }
            Err(e) => {
                log::error!("ALLOC_PAGE: Allocation failed: {:?}", e);
                SyscallResult::error(SyscallResult::EFI_OUT_OF_RESOURCES)
            }
        }
    }

    /// Handles page free syscall.
    ///
    /// Mirrors the C implementation's `SMM_FREE_PAGE` case.
    /// - Arg1: Physical address to free
    /// - Arg2: Number of pages
    fn handle_free_page(&self, ctx: &SyscallContext) -> SyscallResult {
        let addr = ctx.arg1;
        let page_count = ctx.arg2;
        log::trace!("FREE_PAGE: addr=0x{:x}, count={}", addr, page_count);

        if page_count == 0 {
            log::error!("FREE_PAGE: Zero page count");
            return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
        }

        // Validate the address is page-aligned
        if addr % crate::PAGE_SIZE as u64 != 0 {
            log::error!("FREE_PAGE: Address 0x{:x} is not page-aligned", addr);
            return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
        }

        // Verify the range was allocated as User type (Ring 3 code should only free its own memory)
        // This prevents user code from freeing supervisor-internal allocations.
        match crate::PAGE_ALLOCATOR.get_allocation_type(addr) {
            Some(crate::mm_mem::AllocationType::User) => {
                // Good - this is user-owned memory
            }
            Some(crate::mm_mem::AllocationType::Supervisor) => {
                log::error!("FREE_PAGE: Address 0x{:x} is a supervisor allocation - access denied", addr);
                return SyscallResult::error(SyscallResult::EFI_SECURITY_VIOLATION);
            }
            None => {
                log::error!("FREE_PAGE: Address 0x{:x} is not allocated", addr);
                return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
            }
        }

        // Free the pages, verifying they are all User allocations
        match crate::PAGE_ALLOCATOR.free_pages_checked(
            addr,
            page_count as usize,
            crate::mm_mem::AllocationType::User,
        ) {
            Ok(()) => {
                log::debug!("FREE_PAGE: Freed {} page(s) at 0x{:x}", page_count, addr);
                SyscallResult::success(0)
            }
            Err(e) => {
                log::error!("FREE_PAGE: Free failed: {:?}", e);
                SyscallResult::error(SyscallResult::EFI_SECURITY_VIOLATION)
            }
        }
    }

    /// Handles start AP procedure syscall.
    ///
    /// Validates the request and delegates to the platform-specific AP startup
    /// function registered during [`MmSupervisorCore`] initialization.
    ///
    /// Checks performed before dispatch:
    /// - Procedure pointer is non-null
    /// - Procedure pointer is within user-accessible memory (unblocked region)
    /// - Argument pointer (if non-null) is within user-accessible memory
    ///
    /// The remaining validation (CPU index range, BSP check, AP busy check) and
    /// the actual dispatch are handled by the registered AP startup function,
    /// which has access to the CPU manager and mailbox manager.
    ///
    /// - Arg1: Procedure function pointer
    /// - Arg2: CPU index
    /// - Arg3: Argument pointer
    fn handle_start_ap_proc(&self, ctx: &SyscallContext) -> SyscallResult {
        let procedure = ctx.arg1;
        let cpu_index = ctx.arg2;
        let argument = ctx.arg3;

        log::info!(
            "START_AP_PROC: proc=0x{:x}, cpu={}, arg=0x{:x}",
            procedure, cpu_index, argument
        );

        // 1. Validate procedure pointer is non-null
        if procedure == 0 {
            log::error!("START_AP_PROC: Null procedure pointer");
            return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
        }

        // 2. Validate procedure pointer is within mapped memory via page table query
        if crate::query_address_ownership(procedure, core::mem::size_of::<usize>() as u64).is_none() {
            log::error!(
                "START_AP_PROC: Procedure 0x{:x} not in mapped memory",
                procedure
            );
            return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
        }

        // 3. Validate argument pointer (if non-null) is within mapped memory
        if argument != 0 {
            if crate::query_address_ownership(argument, core::mem::size_of::<usize>() as u64).is_none() {
                log::error!(
                    "START_AP_PROC: Argument 0x{:x} not in mapped memory",
                    argument
                );
                return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
            }
        }

        // 4. Delegate to the registered AP startup function
        match crate::AP_STARTUP_FN.get() {
            Some(start_fn) => {
                log::info!(
                    "START_AP_PROC: Dispatching to AP startup function at {:p} for CPU {}",
                    *start_fn as *const (), cpu_index
                );
                let status = start_fn(cpu_index, procedure, argument);
                if status == 0 {
                    SyscallResult::success(0)
                } else {
                    SyscallResult::error(status)
                }
            }
            None => {
                log::error!("START_AP_PROC: AP startup not initialized");
                SyscallResult::error(SyscallResult::EFI_NOT_READY)
            }
        }
    }

    /// Handles extended save state read syscall.
    ///
    /// - Arg1: User MM CPU protocol pointer
    /// - Arg2: Width of buffer to read in bytes
    /// - Arg3: User buffer to hold return data
    fn handle_save_state_read2(&self, ctx: &SyscallContext) -> SyscallResult {
        log::trace!(
            "SAVE_STATE_READ2: protocol=0x{:x}, width={}, buffer=0x{:x}",
            ctx.arg1,
            ctx.arg2,
            ctx.arg3
        );

        // Validate parameters
        if ctx.arg1 == 0 {
            log::error!("SAVE_STATE_READ2: Null protocol pointer");
            return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
        }

        // Delegate to save state module Phase 2
        crate::save_state::save_state_read_phase2(ctx.arg1, ctx.arg2, ctx.arg3)
    }

    /// Handles MM memory unblocked check syscall.
    ///
    /// Checks if a memory range is outside MMRAM and valid (unblocked), AND
    /// is within user-owned space.
    /// - Arg1: Physical address
    /// - Arg2: Size in bytes
    /// - Returns: 1 (TRUE) if valid, 0 (FALSE) otherwise
    fn handle_mm_memory_unblocked(&self, ctx: &SyscallContext) -> SyscallResult {
        let addr = ctx.arg1;
        let size = ctx.arg2;
        log::trace!("MM_MEMORY_UNBLOCKED: addr=0x{:x}, size=0x{:x}", addr, size);

        // Check if the buffer is within an unblocked memory region
        let is_valid = UNBLOCKED_MEMORY_TRACKER.is_within_unblocked_region(addr, size);

        if !is_valid {
            log::trace!("MM_MEMORY_UNBLOCKED: addr=0x{:x} size=0x{:x} not in unblocked region", addr, size);
            return SyscallResult::success(0); // FALSE
        }

        // Additional check - verify buffer is in user-owned space
        match query_address_ownership(addr, size) {
            Some(owner) => {
                if owner != PageOwnership::User {
                    log::trace!("MM_MEMORY_UNBLOCKED: addr=0x{:x} size=0x{:x} owned by {:?} - not valid", addr, size, owner);
                    return SyscallResult::success(0); // FALSE
                }
            }
            None => {
                log::trace!("MM_MEMORY_UNBLOCKED: addr=0x{:x} size=0x{:x} not in mapped memory", addr, size);
                return SyscallResult::success(0); // FALSE
            }
        }

        log::trace!("MM_MEMORY_UNBLOCKED: addr=0x{:x} size=0x{:x} is valid", addr, size);
        SyscallResult::success(1) // TRUE
    }

    /// Handles MM is communication buffer check syscall.
    ///
    /// Verifies that a given memory range is a valid communication buffer.
    /// - Arg1: Buffer address
    /// - Arg2: Buffer size
    /// - Returns: 1 (TRUE) if valid comm buffer, 0 (FALSE) otherwise
    fn handle_mm_is_comm_buffer(&self, ctx: &SyscallContext) -> SyscallResult {
        let address = ctx.arg1;
        let size = ctx.arg2;
        log::trace!("MM_IS_COMM_BUFFER: addr=0x{:x}, size=0x{:x}", address, size);

        let config = match COMM_BUFFER_CONFIG.get() {
            Some(c) => c,
            None => {
                log::error!("MM_IS_COMM_BUFFER: Comm buffer config not initialized");
                return SyscallResult::success(0); // FALSE
            }
        };

        let buf_start = config.user_comm_buffer_internal;
        let buf_end = buf_start.saturating_add(config.user_comm_buffer_size);
        let range_end = address.saturating_add(size);

        // Check that the range is non-empty and falls entirely within the user comm buffer.
        let is_valid = size > 0 && address >= buf_start && range_end <= buf_end;

        log::debug!("MM_IS_COMM_BUFFER: addr=0x{:x} size=0x{:x} => {}", address, size, is_valid);
        SyscallResult::success(if is_valid { 1 } else { 0 })
    }
}

// ============================================================================
// Global Instance
// ============================================================================

/// Global syscall dispatcher instance.
pub static SYSCALL_DISPATCHER: SyscallDispatcher = SyscallDispatcher::new();

// ============================================================================
// C-compatible Entry Point
// ============================================================================

/// C-compatible syscall dispatcher entry point.
///
/// This function is called from the assembly syscall entry stub (SyscallCenter).
///
/// # Arguments
///
/// * `call_index` - Syscall index (from RAX)
/// * `arg1` - First argument (from RDX)
/// * `arg2` - Second argument (from R8)
/// * `arg3` - Third argument (from R9)
/// * `caller_addr` - Caller return address (from RCX)
/// * `ring3_stack_ptr` - Ring 3 stack pointer
///
/// # Returns
///
/// The value to return in RAX.
#[unsafe(no_mangle)]
pub extern "efiapi" fn syscall_dispatcher(
    call_index: u64,
    arg1: u64,
    arg2: u64,
    arg3: u64,
    caller_addr: u64,
    ring3_stack_ptr: u64,
) -> u64 {
    let ctx = SyscallContext {
        call_index,
        arg1,
        arg2,
        arg3,
        caller_addr,
        ring3_stack_ptr,
    };

    let result = SYSCALL_DISPATCHER.dispatch(&ctx);

    // For now, just return the value. In the future, we may need to handle
    // error codes differently.
    if result.status != 0 {
        panic!("Syscall error: status=0x{:x}", result.status);
    } else {
        result.value
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syscall_index_roundtrip() {
        for idx in [
            SyscallIndex::RdMsr,
            SyscallIndex::WrMsr,
            SyscallIndex::Cli,
            SyscallIndex::IoRead,
            SyscallIndex::IoWrite,
        ] {
            assert_eq!(SyscallIndex::from_u64(idx.as_u64()), Some(idx));
        }
    }

    #[test]
    fn test_unknown_syscall_index() {
        assert_eq!(SyscallIndex::from_u64(0xFFFF), None);
        assert_eq!(SyscallIndex::from_u64(0), None);
    }

    #[test]
    fn test_syscall_result() {
        let success = SyscallResult::success(42);
        assert_eq!(success.value, 42);
        assert_eq!(success.status, 0);

        let error = SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
        assert_eq!(error.value, 0);
        assert_eq!(error.status, SyscallResult::EFI_INVALID_PARAMETER);
    }
}
