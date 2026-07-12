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

use patina::management_mode::supervisor::{SaveStateType, SyscallIndex, raw_syscall};
use r_efi::efi;

/// Reads a single raw save-state `field` for `cpu_index` from the MM Supervisor.
///
/// Returns the raw field value on success, or the supervisor's error status
/// (`EFI_NOT_FOUND`, `EFI_ACCESS_DENIED`, `EFI_INVALID_PARAMETER`, …).
pub(crate) fn read_field(cpu_index: usize, field: SaveStateType) -> Result<u64, efi::Status> {
    let mut value: u64 = 0;

    // SAFETY: `value` is a valid, writable 8-byte local. The supervisor validates
    // that the pointer is user-owned before writing exactly 8 bytes into it and
    // returns the status in RAX.
    let result = unsafe {
        raw_syscall(
            SyscallIndex::SaveStateRead.as_u64(),
            cpu_index as u64,
            field.as_u64(),
            core::ptr::addr_of_mut!(value) as u64,
        )
    };

    // The save-state read returns its `EFI_STATUS` in RAX (`result.value`).
    let status = efi::Status::from_usize(result.value as usize);
    if status == efi::Status::SUCCESS { Ok(value) } else { Err(status) }
}
