//! MM save-state read syscalls (Ring 3 user MM → MM Supervisor).
//!
//! The MM save state lives in supervisor-only SMRAM, so the CPU protocol's
//! `ReadSaveState` cannot read it directly; it issues these syscalls to have the
//! MM Supervisor perform the read (and its security-policy check) in Ring 0.
//!
//! ## ABI
//!
//! The read is a two-phase syscall (mirroring the historical `SysCallLib.h`
//! interface consumed by the C `MmSupervisedCpu` driver):
//!
//! 1. [`SyscallIndex::SaveStateRead`] — arg1 = `this` token, arg2 = register,
//!    arg3 = CPU index. Validates the request and records it.
//! 2. [`SyscallIndex::SaveStateRead2`] — arg1 = `this` token, arg2 = width,
//!    arg3 = output buffer. Applies the MM security policy, reads the register
//!    (assembling the composite `EFI_MM_SAVE_STATE_IO_INFO` for the `IO`
//!    pseudo-register), and copies the result into the caller's buffer.
//!
//! The supervisor returns the `EFI_STATUS` for each phase in `RAX`, and the
//! `this` token is echoed across both phases for a consistency check.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use patina::management_mode::supervisor::SyscallIndex;
use r_efi::efi;

/// Issue a raw `syscall` to the MM Supervisor from Ring 3 user MM and return the
/// value the supervisor placed in `RAX`.
///
/// ## Safety
///
/// Transfers control to the supervisor; the arguments must be valid for the
/// given syscall index. Only meaningful in Ring 3 user MM on x86-64.
#[cfg(all(target_os = "uefi", target_arch = "x86_64"))]
unsafe fn raw_syscall(call_index: u64, arg1: u64, arg2: u64, arg3: u64) -> u64 {
    let value: u64;

    // ABI: RAX = call index, RDX = arg1, R8 = arg2, R9 = arg3. The supervisor
    // returns its result in RAX. RCX and R11 are clobbered by `syscall`.
    // SAFETY: A `syscall` into the MM Supervisor with the documented register ABI.
    // The listed clobbers (RCX, R11) match the `syscall` instruction, and no memory
    // operands are used here, so the operation cannot violate Rust's memory model.
    unsafe {
        core::arch::asm!(
            "syscall",
            inlateout("rax") call_index => value,
            inlateout("rdx") arg1 => _,
            in("r8") arg2,
            in("r9") arg3,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }

    value
}

/// Host/non-UEFI stub so the crate links for tests and non-x86 UEFI targets.
///
/// Save-state reads are only meaningful in Ring 3 user MM on x86-64; anywhere
/// else the operation is unsupported.
#[cfg(not(all(target_os = "uefi", target_arch = "x86_64")))]
unsafe fn raw_syscall(_call_index: u64, _arg1: u64, _arg2: u64, _arg3: u64) -> u64 {
    efi::Status::UNSUPPORTED.as_usize() as u64
}

/// Read `width` bytes of the given save-state `register` from the specified
/// CPU's MM save state into `buffer`, via the MM Supervisor.
///
/// `register` is a raw `EFI_MM_SAVE_STATE_REGISTER` value. `this` is an opaque
/// token echoed across the two syscall phases for a consistency check (the CPU
/// protocol producer passes its protocol interface pointer).
///
/// Returns `EFI_SUCCESS` on success, or the supervisor's error status
/// (`EFI_NOT_FOUND`, `EFI_ACCESS_DENIED`, `EFI_INVALID_PARAMETER`, …).
///
/// ## Safety
///
/// `buffer` must point to at least `width` bytes of user-owned, writable memory.
/// The supervisor validates that the buffer is user-owned before writing, but
/// the caller must still guarantee the pointer and length are valid.
pub(crate) unsafe fn read_save_state_register(
    this: usize,
    width: usize,
    register: u32,
    cpu_index: usize,
    buffer: *mut u8,
) -> efi::Status {
    // Phase 1: record the register and CPU index.
    // SAFETY: `SaveStateRead` takes the `this` token, register, and CPU index as
    // scalar arguments; no memory is dereferenced by the supervisor in this phase.
    let phase1 =
        unsafe { raw_syscall(SyscallIndex::SaveStateRead.as_u64(), this as u64, register as u64, cpu_index as u64) };
    let phase1 = efi::Status::from_usize(phase1 as usize);
    if phase1 != efi::Status::SUCCESS {
        return phase1;
    }

    // Phase 2: apply policy, read, and copy into the caller's buffer.
    // SAFETY: `SaveStateRead2` writes up to `width` bytes into `buffer`; the caller
    // guarantees (per this function's contract) that `buffer` is valid for `width`
    // bytes, and the supervisor additionally validates user ownership before writing.
    let phase2 =
        unsafe { raw_syscall(SyscallIndex::SaveStateRead2.as_u64(), this as u64, width as u64, buffer as u64) };
    efi::Status::from_usize(phase2 as usize)
}
