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
//!
//! The supervisor returns the field value directly in RAX. It reads only one
//! field — this crate assembles the composite `EFI_MM_SAVE_STATE_IO_INFO` itself
//! (see [`crate::component`]). An `IoTrap` read returns `0` when the CPU did not
//! trap an I/O instruction; any other failure faults in the supervisor.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use patina::management_mode::supervisor::{SaveStateType, SyscallIndex, raw_syscall};

/// Reads a single raw save-state `field` for `cpu_index` from the MM Supervisor,
/// returning the value the supervisor placed in RAX.
///
/// An `IoTrap` read returns `0` when the CPU did not trap an I/O instruction. Any
/// other failure (invalid CPU, policy denial, …) faults in the supervisor rather
/// than returning, so a return here always carries a valid field value.
pub(crate) fn read_field(cpu_index: usize, field: SaveStateType) -> u64 {
    // SAFETY: the arguments are plain scalars (no memory is shared with the
    // supervisor). The supervisor returns the field value in RAX, or faults on
    // error rather than returning.
    unsafe { raw_syscall(SyscallIndex::SaveStateRead.as_u64(), cpu_index as u64, field.as_u64(), 0) }
}
