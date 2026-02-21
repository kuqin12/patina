//! Save State Read Operations for the MM Supervisor Syscall Dispatcher
//!
//! Implements the two-phase save state read protocol used by the
//! `EFI_MM_CPU_PROTOCOL.ReadSaveState()` user-space API.
//!
//! **Phase 1** (`SyscallIndex::SaveStateRead`): stores the requested register
//! and CPU index in a per-BSP holder.
//!
//! **Phase 2** (`SyscallIndex::SaveStateRead2`): validates the request against
//! the MM security policy, reads the register value from the CPU's SMRAM save
//! state area, and copies the result into the caller-supplied buffer.
//!
//! ## Security Model
//!
//! - User buffer addresses are validated via page-table ownership queries.
//! - Policy-gated registers (RAX, IO) are checked through
//!   [`PolicyGate::is_save_state_read_allowed`](patina_mm_policy::PolicyGate::is_save_state_read_allowed).
//! - `PROCESSOR_ID` is always allowed (informational, not security-sensitive).
//! - Other architectural registers pass through without policy gating, matching
//!   the C reference implementation's allow-list semantics.
//!
//! ## Vendor Selection
//!
//! The SMRAM save state layout (Intel vs AMD) is selected **at build time**
//! via Cargo features on the `patina` crate (`save_state_intel` or
//! `save_state_amd`).  All vendor-specific register offsets and I/O field
//! parsing live in the SDK; this module is vendor-agnostic.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0

use spin::Mutex;

use patina::component::service::save_state::{
    self, MmSaveStateIoInfo, MmSaveStateRegister, IO_INFO_SIZE, IO_TYPE_INPUT, IO_TYPE_OUTPUT,
    LMA_32BIT, LMA_64BIT, PROCESSOR_INFO_ENTRY_SIZE, IA32_EFER_LMA,
};
use patina_mm_policy::{SaveStateCondition, SaveStateField};

use crate::privilege_mgmt::SyscallResult;
use crate::{PageOwnership, POLICY_GATE, SMM_CPU_PRIVATE, SmmCpuPrivateData, query_address_ownership};

// ============================================================================
// Policy Field Mapping (supervisor-specific, not in SDK)
// ============================================================================

/// Maps a save state register to a policy-gated [`SaveStateField`], if any.
///
/// Only RAX and IO are subject to policy gating.  All other registers are
/// either always allowed or have special handling (PROCESSOR_ID).
fn to_policy_field(reg: MmSaveStateRegister) -> Option<SaveStateField> {
    match reg {
        MmSaveStateRegister::Rax => Some(SaveStateField::Rax),
        MmSaveStateRegister::Io => Some(SaveStateField::IoTrap),
        _ => None,
    }
}

// ============================================================================
// Two-Phase State Holder
// ============================================================================

/// Holds the parameters from Phase 1 until Phase 2 completes the read.
struct SaveStateAccessHolder {
    /// User protocol pointer (must match across both phases).
    user_protocol: u64,
    /// Register to read.
    register: MmSaveStateRegister,
    /// CPU index to read from.
    cpu_index: u64,
}

/// Global holder for the in-flight two-phase save state read.
///
/// Only one read can be in flight at a time (enforced by the single-threaded
/// BSP syscall dispatch model).
static SAVE_STATE_ACCESS: Mutex<Option<SaveStateAccessHolder>> = Mutex::new(None);

// ============================================================================
// Phase 1: Store Register + CPU Index
// ============================================================================

/// Processes Phase 1 of the save state read syscall.
///
/// Validates and stores the register and CPU index for the subsequent Phase 2
/// call.
///
/// # Arguments
///
/// * `protocol` - User MM CPU protocol pointer (for consistency check in Phase 2)
/// * `register_raw` - Raw `EFI_MM_SAVE_STATE_REGISTER` value
/// * `cpu_index` - CPU index to read the save state from
pub fn save_state_read_phase1(
    protocol: u64,
    register_raw: u64,
    cpu_index: u64,
) -> SyscallResult {
    // Validate register
    let register = match MmSaveStateRegister::from_u64(register_raw) {
        Some(r) => r,
        None => {
            log::error!(
                "SAVE_STATE_READ: Unknown register value: {}",
                register_raw
            );
            return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
        }
    };

    // Validate CPU index against NumberOfCpus
    let num_cpus = match get_number_of_cpus() {
        Ok(n) => n,
        Err(status) => return SyscallResult::error(status),
    };

    if cpu_index >= num_cpus {
        log::error!(
            "SAVE_STATE_READ: CPU index {} >= NumberOfCpus {}",
            cpu_index,
            num_cpus
        );
        return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
    }

    // Store for Phase 2
    let mut access = SAVE_STATE_ACCESS.lock();
    *access = Some(SaveStateAccessHolder {
        user_protocol: protocol,
        register,
        cpu_index,
    });

    log::debug!(
        "SAVE_STATE_READ: Stored register={:?}, cpu_index={} for Phase 2",
        register,
        cpu_index
    );
    SyscallResult::success(0)
}

// ============================================================================
// Phase 2: Policy Check + Read + Copy
// ============================================================================

/// Processes Phase 2 of the save state read syscall.
///
/// Validates the request against the MM security policy, reads the register
/// from the CPU's SMRAM save state area, and copies the result into the user
/// buffer.
///
/// # Arguments
///
/// * `protocol` - User MM CPU protocol pointer (must match Phase 1)
/// * `width` - Width of the read in bytes
/// * `buffer` - User buffer to receive the register value
pub fn save_state_read_phase2(protocol: u64, width: u64, buffer: u64) -> SyscallResult {
    // Retrieve and consume the Phase 1 state
    let holder = {
        let mut access = SAVE_STATE_ACCESS.lock();
        match access.take() {
            Some(h) => h,
            None => {
                log::error!("SAVE_STATE_READ2: Phase 1 not completed");
                return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
            }
        }
    };

    // Verify protocol matches Phase 1
    if holder.user_protocol != protocol {
        log::error!(
            "SAVE_STATE_READ2: Protocol mismatch: expected 0x{:x}, got 0x{:x}",
            holder.user_protocol,
            protocol
        );
        return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
    }

    // Validate width and buffer
    if width == 0 || buffer == 0 {
        log::error!(
            "SAVE_STATE_READ2: Invalid width ({}) or null buffer",
            width
        );
        return SyscallResult::error(SyscallResult::EFI_INVALID_PARAMETER);
    }

    let register = holder.register;
    let cpu_index = holder.cpu_index;

    // Determine the actual number of bytes we'll write
    let write_size = actual_write_size(register, width);
    if write_size == 0 {
        log::error!(
            "SAVE_STATE_READ2: Unsupported width {} for register {:?}",
            width,
            register
        );
        return SyscallResult::error(SyscallResult::EFI_UNSUPPORTED);
    }

    // Validate buffer is in user-owned memory
    match query_address_ownership(buffer, write_size as u64) {
        Some(PageOwnership::User) => {}
        Some(owner) => {
            log::error!(
                "SAVE_STATE_READ2: Buffer 0x{:x} owned by {:?}, expected User",
                buffer,
                owner
            );
            return SyscallResult::error(SyscallResult::EFI_ACCESS_DENIED);
        }
        None => {
            log::error!(
                "SAVE_STATE_READ2: Buffer 0x{:x} not in mapped memory",
                buffer
            );
            return SyscallResult::error(SyscallResult::EFI_ACCESS_DENIED);
        }
    }

    // Special case: PROCESSOR_ID — always allowed, no policy check
    if register == MmSaveStateRegister::ProcessorId {
        return read_processor_id(cpu_index, buffer);
    }

    // Policy check for gated registers (RAX, IO)
    if let Some(policy_field) = to_policy_field(register) {
        let condition = inspect_io_condition(cpu_index);
        let gate = match POLICY_GATE.get() {
            Some(g) => g,
            None => {
                log::error!("SAVE_STATE_READ2: Policy gate not initialized");
                return SyscallResult::error(SyscallResult::EFI_NOT_READY);
            }
        };

        if let Err(e) = gate.is_save_state_read_allowed(policy_field, width as usize, condition) {
            log::error!(
                "SAVE_STATE_READ2: Policy denied read of {:?}: {:?}",
                register,
                e
            );
            return SyscallResult::error(SyscallResult::EFI_ACCESS_DENIED);
        }
    }

    // Get the save state base pointer for this CPU
    let save_state_base = match get_save_state_base(cpu_index) {
        Ok(base) => base,
        Err(status) => return SyscallResult::error(status),
    };

    // Dispatch to the appropriate read handler.
    //
    // SAFETY: `save_state_base` points to a valid SMRAM save state region
    // (obtained from SMM_CPU_PRIVATE which is set up by PiSmmCpuDxeSmm).
    // `buffer` has been validated as a user-owned region of sufficient size.
    let status = match register {
        MmSaveStateRegister::Io => unsafe { read_io_register(save_state_base, buffer as *mut u8) },
        MmSaveStateRegister::Lma => unsafe {
            read_lma_register(save_state_base, width, buffer as *mut u8)
        },
        _ => unsafe {
            read_architectural_register(save_state_base, register, width, buffer as *mut u8)
        },
    };

    if status == SyscallResult::EFI_SUCCESS {
        log::debug!(
            "SAVE_STATE_READ2: Read {:?} (cpu={}, width={}) successfully",
            register,
            cpu_index,
            width
        );
        SyscallResult::success(0)
    } else {
        SyscallResult::error(status)
    }
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// Returns the number of CPUs from the SMM CPU private data.
fn get_number_of_cpus() -> Result<u64, u64> {
    let cpu_private_addr = match SMM_CPU_PRIVATE.get() {
        Some(&addr) if addr != 0 => addr,
        _ => {
            log::error!("SMM CPU Private data not initialized");
            return Err(SyscallResult::EFI_NOT_READY);
        }
    };

    // SAFETY: cpu_private_addr is provided by MM IPL via PassDown HOB and validated
    // during initialization to point to a valid SmmCpuPrivateData in SMRAM.
    let cpu_private = unsafe { &*(cpu_private_addr as *const SmmCpuPrivateData) };
    Ok(cpu_private.smm_core_entry_context.number_of_cpus)
}

/// Returns the save state base pointer for a given CPU index.
///
/// The pointer comes from `SmmCpuPrivateData.cpu_save_state[cpu_index]`,
/// which points to SMBASE + 0x7C00 for that CPU's save state area.
fn get_save_state_base(cpu_index: u64) -> Result<*const u8, u64> {
    let cpu_private_addr = match SMM_CPU_PRIVATE.get() {
        Some(&addr) if addr != 0 => addr,
        _ => {
            log::error!("SMM CPU Private data not initialized for save state read");
            return Err(SyscallResult::EFI_NOT_READY);
        }
    };

    // SAFETY: cpu_private_addr points to valid SmmCpuPrivateData in SMRAM.
    let cpu_private = unsafe { &*(cpu_private_addr as *const SmmCpuPrivateData) };
    let save_state_array = cpu_private.cpu_save_state;
    if save_state_array == 0 {
        log::error!("CpuSaveState array pointer is null");
        return Err(SyscallResult::EFI_NOT_READY);
    }

    // Read the per-CPU save state pointer from the array.
    // SAFETY: cpu_save_state points to a valid array of VOID* pointers in SMRAM,
    // and cpu_index has been validated against NumberOfCpus.
    let save_state_ptr = unsafe {
        let array_ptr = save_state_array as *const u64;
        *array_ptr.add(cpu_index as usize)
    };

    if save_state_ptr == 0 {
        log::error!("CpuSaveState[{}] is null", cpu_index);
        return Err(SyscallResult::EFI_INVALID_PARAMETER);
    }

    Ok(save_state_ptr as *const u8)
}

/// Determines the actual number of bytes that will be written to the user buffer.
///
/// Returns 0 if the width is not supported for the given register.
fn actual_write_size(register: MmSaveStateRegister, width: u64) -> usize {
    match register {
        MmSaveStateRegister::Io => IO_INFO_SIZE,
        MmSaveStateRegister::ProcessorId => 8,
        MmSaveStateRegister::Lma => {
            if width == 4 || width == 8 {
                width as usize
            } else {
                0
            }
        }
        _ => {
            if let Some(info) = save_state::register_info(register) {
                if width == 2 && info.native_width >= 2 {
                    2
                } else if width == 4 && info.native_width >= 4 {
                    4
                } else if width == 8 && info.native_width == 8 {
                    8
                } else {
                    0
                }
            } else {
                0
            }
        }
    }
}

/// Reads the PROCESSOR_ID for a given CPU and writes it to the user buffer.
///
/// The ProcessorId (APIC ID) is read from the `EFI_PROCESSOR_INFORMATION` array
/// that was set up by PiSmmCpuDxeSmm and passed through the PassDown HOB.
fn read_processor_id(cpu_index: u64, buffer: u64) -> SyscallResult {
    let cpu_private_addr = match SMM_CPU_PRIVATE.get() {
        Some(&addr) if addr != 0 => addr,
        _ => return SyscallResult::error(SyscallResult::EFI_NOT_READY),
    };

    // SAFETY: validated during initialization.
    let cpu_private = unsafe { &*(cpu_private_addr as *const SmmCpuPrivateData) };

    if cpu_private.processor_info == 0 {
        log::error!("PROCESSOR_ID: ProcessorInfo array is null");
        return SyscallResult::error(SyscallResult::EFI_NOT_READY);
    }

    // Read ProcessorId (first field, offset 0) from the EFI_PROCESSOR_INFORMATION
    // entry at the given CPU index.
    //
    // SAFETY: processor_info points to a valid array of EFI_PROCESSOR_INFORMATION
    // entries in firmware memory, and cpu_index has been validated.
    let processor_id: u64 = unsafe {
        let base = cpu_private.processor_info as *const u8;
        let entry_ptr = base.add(cpu_index as usize * PROCESSOR_INFO_ENTRY_SIZE);
        core::ptr::read_unaligned(entry_ptr as *const u64)
    };

    // Write the 8-byte ProcessorId to the user buffer.
    //
    // SAFETY: buffer is validated as user-owned with sufficient size (8 bytes).
    unsafe {
        core::ptr::write_unaligned(buffer as *mut u64, processor_id);
    }

    log::debug!("PROCESSOR_ID: CPU {} = 0x{:x}", cpu_index, processor_id);
    SyscallResult::success(0)
}

/// Inspects the I/O condition (IN vs OUT) from the save state for policy checking.
///
/// Reads the vendor-specific IO field from the CPU's save state and uses the
/// SDK's [`save_state::parse_io_field`] to determine whether the I/O trap was
/// caused by an IN or OUT instruction.
fn inspect_io_condition(cpu_index: u64) -> Option<SaveStateCondition> {
    log::info!(
        "Inspecting I/O condition for CPU {}: retrieving save state base",
        cpu_index
    );
    let save_state_base = match get_save_state_base(cpu_index) {
        Ok(base) => base,
        Err(e) => {
            log::error!("Failed to get save state base for CPU {} due to {}", cpu_index, e);
            return None
        }
    };

    let vc = save_state::vendor_constants();

    // Verify the save state revision supports IO info before reading the field.
    //
    // SAFETY: save_state_base is valid and smmrevid_offset is within the
    // save state map region.
    let smm_rev_id: u32 = unsafe {
        core::ptr::read_volatile(save_state_base.add(vc.smmrevid_offset as usize) as *const u32)
    };

    if smm_rev_id < vc.min_rev_id_io {
        log::warn!(
            "inspect_io_condition: SMMRevId 0x{:x} < 0x{:x}, IO info not available for CPU {}",
            smm_rev_id,
            vc.min_rev_id_io,
            cpu_index
        );
        return None;
    }

    // Read the vendor-specific IO information field (u32).
    //
    // SAFETY: save_state_base is valid and io_info_offset is within the
    // save state map region.
    let io_field: u32 = unsafe {
        core::ptr::read_volatile(save_state_base.add(vc.io_info_offset as usize) as *const u32)
    };

    log::info!(
        "Inspecting IO condition for CPU {}: IO field = 0x{:x}",
        cpu_index,
        io_field
    );

    // Use the SDK's vendor-specific parser.
    match save_state::parse_io_field(io_field) {
        Some(parsed) => match parsed.io_type {
            IO_TYPE_INPUT => Some(SaveStateCondition::IoRead),
            IO_TYPE_OUTPUT => Some(SaveStateCondition::IoWrite),
            _ => Some(SaveStateCondition::IoWrite),
        },
        None => {
            // No valid IO info (Intel: SmiFlag not set, AMD: reserved size) —
            // default to write condition (matching C behaviour).
            Some(SaveStateCondition::IoWrite)
        }
    }
}

/// Reads an architectural register from the Intel x64 save state map.
///
/// # Safety
///
/// - `save_state_base` must point to a valid `SMRAM_SAVE_STATE_MAP64` region.
/// - `buffer` must point to a user-owned region with at least `width` bytes.
unsafe fn read_architectural_register(
    save_state_base: *const u8,
    register: MmSaveStateRegister,
    width: u64,
    buffer: *mut u8,
) -> u64 {
    let info = match save_state::register_info(register) {
        Some(i) => i,
        None => {
            log::error!(
                "Register {:?} not found in save state map",
                register
            );
            return SyscallResult::EFI_UNSUPPORTED;
        }
    };

    if width == 2 {
        if info.native_width < 2 {
            return SyscallResult::EFI_UNSUPPORTED;
        }
        // Read the low 2 bytes (AMD segment selectors, DT limits).
        let val = unsafe {
            core::ptr::read_volatile(save_state_base.add(info.lo_offset as usize) as *const u16)
        };
        unsafe {
            core::ptr::write_unaligned(buffer as *mut u16, val);
        }
    } else if width == 4 {
        if info.native_width < 4 {
            return SyscallResult::EFI_UNSUPPORTED;
        }
        // Read the low 4 bytes.
        let lo = unsafe {
            core::ptr::read_volatile(save_state_base.add(info.lo_offset as usize) as *const u32)
        };
        unsafe {
            core::ptr::write_unaligned(buffer as *mut u32, lo);
        }
    } else if width == 8 {
        if info.native_width != 8 {
            return SyscallResult::EFI_UNSUPPORTED;
        }
        // Read lo dword then hi dword (handles both contiguous and split layouts).
        let lo = unsafe {
            core::ptr::read_volatile(save_state_base.add(info.lo_offset as usize) as *const u32)
        };
        let hi = unsafe {
            core::ptr::read_volatile(save_state_base.add(info.hi_offset as usize) as *const u32)
        };
        // Write as two adjacent dwords (matching C split-register behaviour).
        unsafe {
            core::ptr::write_unaligned(buffer as *mut u32, lo);
            core::ptr::write_unaligned((buffer as *mut u32).add(1), hi);
        }
    } else {
        return SyscallResult::EFI_INVALID_PARAMETER;
    }

    SyscallResult::EFI_SUCCESS
}

/// Reads the IO pseudo-register and writes an `EFI_MM_SAVE_STATE_IO_INFO`
/// structure to the user buffer.
///
/// The IO pseudo-register provides information about the I/O instruction that
/// triggered the SMI, including the port, width, direction, and data value.
///
/// # Safety
///
/// - `save_state_base` must point to a valid `SMRAM_SAVE_STATE_MAP64` region.
/// - `buffer` must point to a user-owned region with at least [`IO_INFO_SIZE`] bytes.
unsafe fn read_io_register(save_state_base: *const u8, buffer: *mut u8) -> u64 {
    let vc = save_state::vendor_constants();

    // 1. Read SMMRevId to verify IO info is available.
    let smm_rev_id = unsafe {
        core::ptr::read_volatile(
            save_state_base.add(vc.smmrevid_offset as usize) as *const u32,
        )
    };

    if smm_rev_id < vc.min_rev_id_io {
        log::error!(
            "IO_READ: SMMRevId 0x{:x} < 0x{:x}, IO info not supported",
            smm_rev_id,
            vc.min_rev_id_io
        );
        return SyscallResult::EFI_UNSUPPORTED;
    }

    // 2. Read the vendor-specific IO information field and parse it.
    let io_field = unsafe {
        core::ptr::read_volatile(
            save_state_base.add(vc.io_info_offset as usize) as *const u32,
        )
    };

    let parsed = match save_state::parse_io_field(io_field) {
        Some(p) => p,
        None => {
            log::debug!("IO_READ: IO field 0x{:x} did not indicate a valid I/O trap", io_field);
            return SyscallResult::EFI_UNSUPPORTED;
        }
    };

    // 3. Read I/O data from RAX (only the significant bytes).
    let io_data: u64 = unsafe {
        let rax_ptr = save_state_base.add(vc.rax_offset as usize);
        match parsed.byte_count {
            1 => core::ptr::read_volatile(rax_ptr as *const u8) as u64,
            2 => core::ptr::read_volatile(rax_ptr as *const u16) as u64,
            4 => core::ptr::read_volatile(rax_ptr as *const u32) as u64,
            _ => 0,
        }
    };

    // 4. Compose the EFI_MM_SAVE_STATE_IO_INFO structure and copy to user buffer.
    let io_info = MmSaveStateIoInfo {
        io_data,
        io_port: parsed.io_port as u64,
        io_width: parsed.io_width,
        io_type: parsed.io_type,
    };

    unsafe {
        core::ptr::copy_nonoverlapping(
            &io_info as *const MmSaveStateIoInfo as *const u8,
            buffer,
            IO_INFO_SIZE,
        );
    }

    SyscallResult::EFI_SUCCESS
}

/// Reads the LMA pseudo-register (processor Long Mode Active state).
///
/// Returns `LMA_32BIT` (32) or `LMA_64BIT` (64) depending on the IA32_EFER.LMA
/// bit in the save state.
///
/// # Safety
///
/// - `save_state_base` must point to a valid `SMRAM_SAVE_STATE_MAP64` region.
/// - `buffer` must point to a user-owned region with at least `width` bytes.
unsafe fn read_lma_register(
    save_state_base: *const u8,
    width: u64,
    buffer: *mut u8,
) -> u64 {
    let vc = save_state::vendor_constants();

    // AMD64 always operates in 64-bit mode during SMM.
    let lma_value = if vc.lma_always_64 {
        LMA_64BIT
    } else {
        // Read IA32_EFER from the save state.
        let efer = unsafe {
            core::ptr::read_volatile(
                save_state_base.add(vc.efer_offset as usize) as *const u64,
            )
        };
        if (efer & IA32_EFER_LMA) != 0 {
            LMA_64BIT
        } else {
            LMA_32BIT
        }
    };

    if width == 4 {
        unsafe {
            core::ptr::write_unaligned(buffer as *mut u32, lma_value as u32);
        }
    } else if width == 8 {
        unsafe {
            core::ptr::write_unaligned(buffer as *mut u64, lma_value);
        }
    } else {
        return SyscallResult::EFI_INVALID_PARAMETER;
    }

    SyscallResult::EFI_SUCCESS
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_from_u64() {
        assert_eq!(
            MmSaveStateRegister::from_u64(38),
            Some(MmSaveStateRegister::Rax)
        );
        assert_eq!(
            MmSaveStateRegister::from_u64(512),
            Some(MmSaveStateRegister::Io)
        );
        assert_eq!(
            MmSaveStateRegister::from_u64(514),
            Some(MmSaveStateRegister::ProcessorId)
        );
        assert_eq!(MmSaveStateRegister::from_u64(999), None);
        assert_eq!(MmSaveStateRegister::from_u64(0), None);
    }

    #[test]
    fn test_to_policy_field() {
        assert_eq!(
            to_policy_field(MmSaveStateRegister::Rax),
            Some(SaveStateField::Rax)
        );
        assert_eq!(
            to_policy_field(MmSaveStateRegister::Io),
            Some(SaveStateField::IoTrap)
        );
        assert_eq!(to_policy_field(MmSaveStateRegister::Rbx), None);
        assert_eq!(to_policy_field(MmSaveStateRegister::ProcessorId), None);
    }

    #[test]
    fn test_actual_write_size() {
        // IO always writes IO_INFO_SIZE
        assert_eq!(actual_write_size(MmSaveStateRegister::Io, 4), IO_INFO_SIZE);
        assert_eq!(actual_write_size(MmSaveStateRegister::Io, 24), IO_INFO_SIZE);

        // PROCESSOR_ID always writes 8
        assert_eq!(actual_write_size(MmSaveStateRegister::ProcessorId, 8), 8);

        // LMA supports 4 and 8
        assert_eq!(actual_write_size(MmSaveStateRegister::Lma, 4), 4);
        assert_eq!(actual_write_size(MmSaveStateRegister::Lma, 8), 8);
        assert_eq!(actual_write_size(MmSaveStateRegister::Lma, 3), 0);

        // RAX (native 8): supports Width=2, 4, and 8
        assert_eq!(actual_write_size(MmSaveStateRegister::Rax, 2), 2);
        assert_eq!(actual_write_size(MmSaveStateRegister::Rax, 4), 4);
        assert_eq!(actual_write_size(MmSaveStateRegister::Rax, 8), 8);
        assert_eq!(actual_write_size(MmSaveStateRegister::Rax, 16), 0);
    }

    #[test]
    fn test_save_state_access_holder() {
        // Test that the mutex works correctly for Phase 1/Phase 2.
        {
            let mut access = SAVE_STATE_ACCESS.lock();
            assert!(access.is_none());
            *access = Some(SaveStateAccessHolder {
                user_protocol: 0xDEAD,
                register: MmSaveStateRegister::Rax,
                cpu_index: 0,
            });
        }

        {
            let mut access = SAVE_STATE_ACCESS.lock();
            let holder = access.take().unwrap();
            assert_eq!(holder.user_protocol, 0xDEAD);
            assert_eq!(holder.register, MmSaveStateRegister::Rax);
            assert_eq!(holder.cpu_index, 0);
        }

        {
            let access = SAVE_STATE_ACCESS.lock();
            assert!(access.is_none());
        }
    }
}
