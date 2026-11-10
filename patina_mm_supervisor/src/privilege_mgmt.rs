//! Privilege management for the MM Supervisor.
//!
//! Moves execution between Ring 0 (supervisor) and Ring 3 (user) code in MM:
//!
//! 1. Demotion: [`invoke_demoted_routine`] saves and programs the syscall MSRs (see
//!    [`syscall_setup`]), installs the return call gate (see [`call_gate`]) and far returns into
//!    the Ring 3 routine on its CPL3 stack.
//! 2. Syscalls: `syscall` enters `syscall_center`, which calls the syscall dispatcher and returns
//!    to Ring 3 with `sysret`.
//! 3. Return: the demoted routine returns to Ring 0 through the call gate, and the saved MSRs
//!    are restored.
//!
//! ## SMI Entry GDT Selectors
//!
//! ```text
//! LONG_CS_R0        = 0x38  (Ring 0 code segment)
//! LONG_DS_R0        = 0x40  (Ring 0 data segment)
//! LONG_CS_R3_PH     = 0x4B  (Ring 3 code segment placeholder)
//! LONG_DS_R3        = 0x53  (Ring 3 data segment)
//! LONG_CS_R3        = 0x5B  (Ring 3 code segment)
//! CALL_GATE_OFFSET  = 0x60  (Call gate descriptor offset; the asm uses selector 0x63, RPL 3)
//! TSS_SEL_OFFSET    = 0x70  (TSS descriptor offset in the GDT)
//! TSS_DESC_OFFSET   = 0x80  (Offset of the TSS that follows the GDT)
//! ```
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use patina::standard::efi::Status;

pub(crate) mod call_gate;
pub(crate) mod save_state;
mod syscall_dispatcher;
mod syscall_ops;
pub(crate) mod syscall_setup;

/// Result of a syscall: the value to hand back to Ring 3, or the status to report.
pub type SyscallResult = Result<u64, Status>;

// The assembly exists only on the UEFI target; host builds use the wrapper's host paths below.
#[cfg(target_os = "uefi")]
unsafe extern "efiapi" {
    /// The Ring 0 to Ring 3 transition in `call_gate_transfer.asm`. Only called through
    /// [`invoke_demoted_routine`].
    #[link_name = "invoke_demoted_routine"]
    fn invoke_demoted_routine_asm(cpu_index: usize, cpl3_routine: u64, cpl3_stack: u64, arg_count: usize, ...)
    -> usize;
}

/// Number of arguments [`invoke_demoted_routine`] passes to the demoted routine.
pub(crate) const DEMOTED_ROUTINE_ARG_COUNT: usize = 3;

/// Runs a routine at CPL 3 (Ring 3) and returns to Ring 0 through the call gate.
///
/// The routine receives the first `arg_count` of `arg1`, `arg2` and `arg3` as EFIAPI arguments.
/// Returns the routine's EFI status, or an error from the transition.
///
/// Host test builds call the `mock` module instead; other host builds return 0.
///
/// ## Safety
///
/// Must run in Ring 0. `cpl3_routine` must be a valid Ring 3 entry point, `cpl3_stack` the top of
/// this CPU's CPL3 stack, and `arg_count` at most [`DEMOTED_ROUTINE_ARG_COUNT`].
pub(crate) unsafe fn invoke_demoted_routine(
    cpu_index: usize,
    cpl3_routine: u64,
    cpl3_stack: u64,
    arg_count: usize,
    arg1: u64,
    arg2: u64,
    arg3: u64,
) -> usize {
    #[cfg(target_os = "uefi")]
    // SAFETY: the caller upholds this function's contract.
    let ret = unsafe { invoke_demoted_routine_asm(cpu_index, cpl3_routine, cpl3_stack, arg_count, arg1, arg2, arg3) };

    #[cfg(all(not(target_os = "uefi"), test))]
    let ret = mock::invoke_demoted_routine(cpu_index, cpl3_routine, cpl3_stack, arg_count, arg1, arg2, arg3);

    // Other host builds (e.g. doctests) never run the transition.
    #[cfg(all(not(target_os = "uefi"), not(test)))]
    let ret = {
        let _ = (cpu_index, cpl3_routine, cpl3_stack, arg_count, arg1, arg2, arg3);
        0
    };

    ret
}

/// Test-only mock backing [`invoke_demoted_routine`].
///
/// The assembly privilege transition cannot run on the host, so in test builds the wrapper
/// delegates here. Install a handler with [`set_handler`] to observe calls and control the
/// return value; with no handler installed, calls return `0` (`EFI_SUCCESS`).
#[cfg(test)]
pub(crate) mod mock {
    use core::cell::RefCell;

    /// Handler signature: receives the same arguments as the real routine and returns the value
    /// the supervisor should observe.
    type Handler = Box<dyn FnMut(usize, u64, u64, usize, u64, u64, u64) -> usize>;

    thread_local! {
        static HANDLER: RefCell<Option<Handler>> = const { RefCell::new(None) };
    }

    /// Installs a handler invoked in place of the real Ring 0 -> Ring 3 transition.
    pub(crate) fn set_handler<F>(handler: F)
    where
        F: FnMut(usize, u64, u64, usize, u64, u64, u64) -> usize + 'static,
    {
        HANDLER.with(|handler_cell| *handler_cell.borrow_mut() = Some(Box::new(handler)));
    }

    /// Removes any installed handler; subsequent calls return `0`.
    pub(crate) fn clear() {
        HANDLER.with(|handler_cell| *handler_cell.borrow_mut() = None);
    }

    pub(super) fn invoke_demoted_routine(
        cpu_index: usize,
        cpl3_routine: u64,
        cpl3_stack: u64,
        arg_count: usize,
        arg1: u64,
        arg2: u64,
        arg3: u64,
    ) -> usize {
        HANDLER.with(|handler_cell| match handler_cell.borrow_mut().as_mut() {
            Some(handler) => handler(cpu_index, cpl3_routine, cpl3_stack, arg_count, arg1, arg2, arg3),
            None => 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invoke_demoted_routine_delegates_to_mock_handler() {
        // With no handler installed, the wrapper returns 0 (EFI_SUCCESS).
        mock::clear();
        // SAFETY: under `test` the wrapper delegates to the mock; no real privilege transition occurs.
        let ret = unsafe { invoke_demoted_routine(0, 0x1000, 0x2000, 3, 1, 2, 3) };
        assert_eq!(ret, 0);

        // An installed handler observes the arguments and controls the return value, so callers of
        // `invoke_demoted_routine` can be exercised on the host.
        mock::set_handler(|cpu_index, cpl3_routine, cpl3_stack, arg_count, arg1, arg2, arg3| {
            assert_eq!(cpu_index, 7);
            assert_eq!(cpl3_routine, 0xdead_beef);
            assert_eq!(cpl3_stack, 0xcafe);
            assert_eq!(arg_count, 3);
            assert_eq!((arg1, arg2, arg3), (10, 20, 30));
            0x1234
        });
        // SAFETY: see above.
        let ret = unsafe { invoke_demoted_routine(7, 0xdead_beef, 0xcafe, 3, 10, 20, 30) };
        assert_eq!(ret, 0x1234);

        mock::clear();
    }
}
