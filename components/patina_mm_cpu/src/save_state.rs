//! MM save-state read syscall (Ring 3 user MM → MM Supervisor).
//!
//! The MM save state lives in supervisor-only SMRAM, so the CPU protocol's
//! `ReadSaveState` cannot read it directly; it issues this syscall to have the
//! MM Supervisor perform the read (and its security-policy check) in Ring 0.
//!
//! ## ABI
//!
//! A single [`SyscallIndex::SaveStateRead`] syscall reads one raw field:
//!
//! - arg1 = CPU index
//! - arg2 = [`SaveStateType`] (`ProcessorId` / `Rax` / `IoTrap`)
//! - arg3 = pointer to an 8-byte output buffer
//!
//! The supervisor returns the `EFI_STATUS` in RAX and, on success, writes the
//! field value (little-endian `u64`) into the buffer. The supervisor reads only
//! one field — this crate assembles the composite `EFI_MM_SAVE_STATE_IO_INFO`
//! itself (see [`crate::component`]).
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use patina::management_mode::supervisor::{SaveStateType, SyscallIndex};
use r_efi::efi;

/// Issue a raw `syscall` to the MM Supervisor from Ring 3 user MM and return the
/// value the supervisor placed in `RAX` (the `EFI_STATUS`).
///
/// ## Safety
///
/// Transfers control to the supervisor; the arguments must be valid for the
/// given syscall index. Only meaningful in Ring 3 user MM on x86-64.
#[cfg(all(target_os = "uefi", target_arch = "x86_64"))]
unsafe fn raw_syscall(call_index: u64, arg1: u64, arg2: u64, arg3: u64) -> u64 {
    let value: u64;

    // ABI: RAX = call index, RDX = arg1, R8 = arg2, R9 = arg3. The supervisor
    // returns its status in RAX. RCX and R11 are clobbered by `syscall`.
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

/// Reads a single raw save-state `field` for `cpu_index` from the MM Supervisor.
///
/// Returns the raw field value on success, or the supervisor's error status
/// (`EFI_NOT_FOUND`, `EFI_ACCESS_DENIED`, `EFI_INVALID_PARAMETER`, …).
pub(crate) fn read_field(cpu_index: usize, field: SaveStateType) -> Result<u64, efi::Status> {
    let mut value: u64 = 0;

    // SAFETY: `value` is a valid, writable 8-byte local. The supervisor validates
    // that the pointer is user-owned before writing exactly 8 bytes into it and
    // returns the status in RAX.
    let status = unsafe {
        raw_syscall(
            SyscallIndex::SaveStateRead.as_u64(),
            cpu_index as u64,
            field.as_u64(),
            core::ptr::addr_of_mut!(value) as u64,
        )
    };

    let status = efi::Status::from_usize(status as usize);
    if status == efi::Status::SUCCESS { Ok(value) } else { Err(status) }
}
