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

use super::{PrivilegeError, PrivilegeResult};

global_asm!(include_str!("syscall_entry.asm"));

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
    RdMsr = 0x0001,
    /// Write MSR - Arg1: MSR index, Arg2: value
    WrMsr = 0x0002,
    /// CLI - Clear interrupts
    Cli = 0x0003,
    /// IO Read - Arg1: port, Arg2: width
    IoRead = 0x0004,
    /// IO Write - Arg1: port, Arg2: width, Arg3: value
    IoWrite = 0x0005,
    /// WBINVD - Write back and invalidate cache
    Wbinvd = 0x0006,
    /// HLT - Halt processor
    Hlt = 0x0007,
    /// Save State Read - Arg1: register, Arg2: CPU index
    SaveStateRead = 0x0008,
    /// Save State Read 2 - Extended save state read
    SaveStateRead2 = 0x0009,
    /// Register Handler Jump Pointer
    RegHandlerJump = 0x000A,
    /// Allocate Pages - Arg1: memory type, Arg2: page count, Arg3: address ptr
    AllocPage = 0x0010,
    /// Free Pages - Arg1: address, Arg2: page count
    FreePage = 0x0011,
    /// Start AP Procedure - Arg1: procedure, Arg2: CPU index, Arg3: argument
    StartApProc = 0x0012,
    /// Set CPL3 Table - Register user MMST
    SetCpl3Table = 0x0020,
    /// Error Report Jump - Register error reporting function
    ErrReportJump = 0x0030,
}

impl SyscallIndex {
    /// Creates a SyscallIndex from a raw u64 value.
    pub fn from_u64(value: u64) -> Option<Self> {
        match value {
            0x0001 => Some(Self::RdMsr),
            0x0002 => Some(Self::WrMsr),
            0x0003 => Some(Self::Cli),
            0x0004 => Some(Self::IoRead),
            0x0005 => Some(Self::IoWrite),
            0x0006 => Some(Self::Wbinvd),
            0x0007 => Some(Self::Hlt),
            0x0008 => Some(Self::SaveStateRead),
            0x0009 => Some(Self::SaveStateRead2),
            0x000A => Some(Self::RegHandlerJump),
            0x0010 => Some(Self::AllocPage),
            0x0011 => Some(Self::FreePage),
            0x0012 => Some(Self::StartApProc),
            0x0020 => Some(Self::SetCpl3Table),
            0x0030 => Some(Self::ErrReportJump),
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
    /// Registered Ring 3 handler jump pointer.
    registered_ring3_jump_pointer: core::sync::atomic::AtomicU64,
    /// Registered AP Ring 3 jump pointer.
    reg_ap_ring3_jump_pointer: core::sync::atomic::AtomicU64,
    /// Registered error report jump pointer.
    reg_error_report_jump_pointer: core::sync::atomic::AtomicU64,
    /// User MM System Table pointer.
    user_mmst: core::sync::atomic::AtomicU64,
}

impl SyscallDispatcher {
    /// Creates a new syscall dispatcher.
    pub const fn new() -> Self {
        Self {
            initialized: AtomicBool::new(false),
            registered_ring3_jump_pointer: core::sync::atomic::AtomicU64::new(0),
            reg_ap_ring3_jump_pointer: core::sync::atomic::AtomicU64::new(0),
            reg_error_report_jump_pointer: core::sync::atomic::AtomicU64::new(0),
            user_mmst: core::sync::atomic::AtomicU64::new(0),
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
            SyscallIndex::SaveStateRead2 => self.handle_save_state_read2(ctx),
            SyscallIndex::RegHandlerJump => self.handle_reg_handler_jump(ctx),
            SyscallIndex::AllocPage => self.handle_alloc_page(ctx),
            SyscallIndex::FreePage => self.handle_free_page(ctx),
            SyscallIndex::StartApProc => self.handle_start_ap_proc(ctx),
            SyscallIndex::SetCpl3Table => self.handle_set_cpl3_table(ctx),
            SyscallIndex::ErrReportJump => self.handle_err_report_jump(ctx),
        }
    }

    // ========================================================================
    // Syscall Handlers (stubs for now)
    // ========================================================================

    fn handle_rdmsr(&self, ctx: &SyscallContext) -> SyscallResult {
        // TODO: Validate MSR access against policy
        // TODO: Read MSR and return value
        log::trace!("RDMSR: msr=0x{:x}", ctx.arg1);
        SyscallResult::error(SyscallResult::EFI_UNSUPPORTED)
    }

    fn handle_wrmsr(&self, ctx: &SyscallContext) -> SyscallResult {
        // TODO: Validate MSR access against policy
        // TODO: Write MSR
        log::trace!("WRMSR: msr=0x{:x}, value=0x{:x}", ctx.arg1, ctx.arg2);
        SyscallResult::error(SyscallResult::EFI_UNSUPPORTED)
    }

    fn handle_cli(&self, _ctx: &SyscallContext) -> SyscallResult {
        // TODO: Validate CLI is allowed by policy
        log::trace!("CLI");
        SyscallResult::error(SyscallResult::EFI_UNSUPPORTED)
    }

    fn handle_io_read(&self, ctx: &SyscallContext) -> SyscallResult {
        // TODO: Validate IO port access against policy
        // TODO: Read IO port
        log::trace!("IO_READ: port=0x{:x}, width={}", ctx.arg1, ctx.arg2);
        SyscallResult::error(SyscallResult::EFI_UNSUPPORTED)
    }

    fn handle_io_write(&self, ctx: &SyscallContext) -> SyscallResult {
        // TODO: Validate IO port access against policy
        // TODO: Write IO port
        log::trace!("IO_WRITE: port=0x{:x}, width={}, value=0x{:x}", ctx.arg1, ctx.arg2, ctx.arg3);
        SyscallResult::error(SyscallResult::EFI_UNSUPPORTED)
    }

    fn handle_wbinvd(&self, _ctx: &SyscallContext) -> SyscallResult {
        // TODO: Validate WBINVD is allowed by policy
        log::trace!("WBINVD");
        SyscallResult::error(SyscallResult::EFI_UNSUPPORTED)
    }

    fn handle_hlt(&self, _ctx: &SyscallContext) -> SyscallResult {
        // TODO: Validate HLT is allowed by policy
        log::trace!("HLT");
        SyscallResult::error(SyscallResult::EFI_UNSUPPORTED)
    }

    fn handle_save_state_read(&self, ctx: &SyscallContext) -> SyscallResult {
        // TODO: Validate save state access against policy
        log::trace!("SAVE_STATE_READ: register={}, cpu={}", ctx.arg1, ctx.arg2);
        SyscallResult::error(SyscallResult::EFI_UNSUPPORTED)
    }

    fn handle_save_state_read2(&self, ctx: &SyscallContext) -> SyscallResult {
        // TODO: Extended save state read
        log::trace!("SAVE_STATE_READ2: width={}, buffer=0x{:x}", ctx.arg1, ctx.arg2);
        SyscallResult::error(SyscallResult::EFI_UNSUPPORTED)
    }

    fn handle_reg_handler_jump(&self, ctx: &SyscallContext) -> SyscallResult {
        // Register the Ring 3 handler jump pointer
        // TODO: Validate the pointer is in user-accessible memory
        self.registered_ring3_jump_pointer.store(ctx.arg1, Ordering::Release);
        log::info!("Registered Ring 3 handler jump pointer: 0x{:x}", ctx.arg1);
        SyscallResult::success(0)
    }

    fn handle_alloc_page(&self, ctx: &SyscallContext) -> SyscallResult {
        // TODO: Allocate pages for user
        log::trace!("ALLOC_PAGE: type={}, count={}", ctx.arg1, ctx.arg2);
        SyscallResult::error(SyscallResult::EFI_UNSUPPORTED)
    }

    fn handle_free_page(&self, ctx: &SyscallContext) -> SyscallResult {
        // TODO: Free user pages
        log::trace!("FREE_PAGE: addr=0x{:x}, count={}", ctx.arg1, ctx.arg2);
        SyscallResult::error(SyscallResult::EFI_UNSUPPORTED)
    }

    fn handle_start_ap_proc(&self, ctx: &SyscallContext) -> SyscallResult {
        // TODO: Start AP procedure
        log::trace!("START_AP_PROC: proc=0x{:x}, cpu={}, arg=0x{:x}", ctx.arg1, ctx.arg2, ctx.arg3);
        SyscallResult::error(SyscallResult::EFI_UNSUPPORTED)
    }

    fn handle_set_cpl3_table(&self, ctx: &SyscallContext) -> SyscallResult {
        // Set the user MM System Table pointer
        // TODO: Validate the pointer is in user-accessible memory
        self.user_mmst.store(ctx.arg1, Ordering::Release);
        log::info!("Registered User MMST: 0x{:x}", ctx.arg1);
        SyscallResult::success(0)
    }

    fn handle_err_report_jump(&self, ctx: &SyscallContext) -> SyscallResult {
        // Register the error report jump pointer
        // TODO: Validate the pointer is in user-accessible memory
        self.reg_error_report_jump_pointer.store(ctx.arg1, Ordering::Release);
        log::info!("Registered error report jump pointer: 0x{:x}", ctx.arg1);
        SyscallResult::success(0)
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Gets the registered Ring 3 handler jump pointer.
    pub fn get_ring3_handler_jump(&self) -> u64 {
        self.registered_ring3_jump_pointer.load(Ordering::Acquire)
    }

    /// Gets the registered AP Ring 3 jump pointer.
    pub fn get_ap_ring3_jump(&self) -> u64 {
        self.reg_ap_ring3_jump_pointer.load(Ordering::Acquire)
    }

    /// Gets the registered error report jump pointer.
    pub fn get_error_report_jump(&self) -> u64 {
        self.reg_error_report_jump_pointer.load(Ordering::Acquire)
    }

    /// Gets the user MM System Table pointer.
    pub fn get_user_mmst(&self) -> u64 {
        self.user_mmst.load(Ordering::Acquire)
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
        result.status
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
