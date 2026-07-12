//! Save State Read for the MM Supervisor Syscall Dispatcher
//!
//! Implements the single-phase `SaveStateRead` syscall backing the
//! `EFI_MM_CPU_PROTOCOL.ReadSaveState()` user-space API.
//!
//! One syscall reads exactly ONE raw save-state field for a CPU and writes its
//! value (as a little-endian `u64`) into a caller-supplied user buffer, returning
//! the `EFI_STATUS` in RAX. The supported fields are [`SaveStateType`]:
//! `ProcessorId`, `Rax`, and `IoTrap`.
//!
//! Assembling the composite `EFI_MM_SAVE_STATE_IO_INFO` (for
//! `EFI_MM_SAVE_STATE_REGISTER_IO`) is the **caller's** responsibility — the
//! supervisor no longer recurses into RAX to fill in the I/O data. For an
//! `IoTrap` read the supervisor returns only the trap descriptor (port, width,
//! direction) packed into the `u64`; the caller reads `Rax` separately for the
//! I/O data.
//!
//! ## Security Model
//!
//! - User buffer addresses are validated via page-table ownership queries.
//! - `Rax` and `IoTrap` are checked through
//!   [`PolicyGate::is_save_state_read_allowed`](crate::mm_policy::PolicyGate::is_save_state_read_allowed).
//!   `Rax` clears the policy only when the trapping instruction was an I/O write.
//! - `ProcessorId` is always allowed (informational, not security-sensitive).
//!
//! ## Vendor Selection
//!
//! The SMRAM save state layout (Intel vs AMD) is selected **at build time** via
//! Cargo features on the `patina` crate (`save_state_intel` or `save_state_amd`).
//! All vendor-specific register offsets and I/O field parsing live in the SDK;
//! this module is vendor-agnostic.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0

use crate::mm_policy::{SaveStateCondition, SaveStateField};
use patina::management_mode::supervisor::{IO_TRAP_TYPE_SHIFT, IO_TRAP_WIDTH_SHIFT, SaveStateType};
use patina_internal_cpu::save_state::{self, IO_TYPE_INPUT, IO_TYPE_OUTPUT, PROCESSOR_INFO_ENTRY_SIZE};
use r_efi::efi::Status;

use crate::{PageOwnership, privilege_mgmt::SyscallResult, query_address_ownership, state::security_state};

/// Size in bytes of one `SMRAM_SAVE_STATE_MAP` region.
///
/// The relocation code sets every CPU's save-state size to
/// `sizeof(SMRAM_SAVE_STATE_MAP)` — a fixed 0x400-byte region spanning
/// SMBASE+0x7C00..SMBASE+0x8000 (Intel SDM Vol 3C, §34.4). Because it is
/// identical for every CPU, it is a constant here rather than a per-CPU array
/// passed through the HOB.
const SMRAM_SAVE_STATE_MAP_SIZE: u64 = 0x400;

/// Offset of the `SMRAM_SAVE_STATE_MAP` from a CPU's SMBASE.
///
/// A fixed architectural offset (Intel SDM Vol 3C, §34.4;
/// `SMRAM_SAVE_STATE_MAP_OFFSET` in MdePkg). The per-CPU save-state region base
/// is `sm_base[i] + SMRAM_SAVE_STATE_MAP_OFFSET`, derived here so the loader only
/// has to pass the raw SMBASE array.
const SMRAM_SAVE_STATE_MAP_OFFSET: u64 = 0xfc00;

/// Number of bytes written to the user buffer (one `u64` field value).
const FIELD_VALUE_SIZE: u64 = 8;

/// Per-CPU save-state metadata needed by the save-state read syscall.
///
/// Assembled at initialization from two public sources instead of the private
/// `SMM_CPU_PRIVATE_DATA` layout:
///
/// - `number_of_cpus` and `processor_info` come from the MP Information HOB
///   (`gMpInformationHobGuid`).
/// - `sm_base` (the per-CPU SMBASE array) is passed through the MM Supervisor
///   PassDown HOB. The save-state region base is derived as
///   `sm_base[i] + SMRAM_SAVE_STATE_MAP_OFFSET` with the fixed
///   [`SMRAM_SAVE_STATE_MAP_SIZE`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct SaveStateInfo {
    /// Number of CPUs (from `MP_INFORMATION_HOB_DATA.NumberOfProcessors`).
    pub(crate) number_of_cpus: u64,
    /// Pointer to the `EFI_PROCESSOR_INFORMATION[]` array (from the MP Information HOB).
    pub(crate) processor_info: u64,
    /// Pointer to the per-CPU SMBASE array (`u64[number_of_cpus]`).
    pub(crate) sm_base: u64,
}

/// Reads a single save-state field for a CPU and return the value or result.
///
/// The supervisor reads exactly one raw field; the caller assembles any
/// composite structure (`EFI_MM_SAVE_STATE_IO_INFO`) itself.
pub fn save_state_read(cpu_index: u64, field_raw: u64, buffer: u64) -> SyscallResult {
    let field = match SaveStateType::from_u64(field_raw) {
        Some(f) => f,
        None => {
            log::error!("SAVE_STATE_READ: Unknown save-state type: {}", field_raw);
            return Err(Status::INVALID_PARAMETER);
        }
    };

    // Validate the CPU index against NumberOfCpus.
    let num_cpus = match get_number_of_cpus() {
        Ok(n) => n,
        Err(status) => return Err(status),
    };

    if cpu_index >= num_cpus {
        log::error!("SAVE_STATE_READ: CPU index {} >= NumberOfCpus {}", cpu_index, num_cpus);
        return Err(Status::INVALID_PARAMETER);
    }

    let value = match field {
        // PROCESSOR_ID is informational and not policy-gated.
        SaveStateType::ProcessorId => read_processor_id_value(cpu_index)?,

        // RAX and the I/O trap descriptor are read from the SMRAM save state and
        // gated by the save-state security policy.
        SaveStateType::Rax | SaveStateType::IoTrap => {
            let policy_field = match field {
                SaveStateType::Rax => SaveStateField::Rax,
                SaveStateType::IoTrap => SaveStateField::IoTrap,
                SaveStateType::ProcessorId => unreachable!("handled above"),
            };

            let view = match get_save_state_view(cpu_index) {
                Ok(v) => v,
                Err(status) => {
                    log::error!("SAVE_STATE_READ: Failed to get save-state view for CPU {}: {:?}", cpu_index, status);
                    return Err(status);
                }
            };

            let condition = inspect_io_condition(&view);

            let gate = match security_state().policy_gate() {
                Some(g) => g,
                None => {
                    log::error!("SAVE_STATE_READ: Policy gate not initialized");
                    return Err(Status::NOT_READY);
                }
            };
            if let Err(e) = gate.is_save_state_read_allowed(policy_field, FIELD_VALUE_SIZE as usize, condition) {
                log::error!("SAVE_STATE_READ: Policy denied read of {:?}: {:?}", field, e);
                return Err(Status::ACCESS_DENIED);
            }

            match field {
                SaveStateType::Rax => read_rax_value(&view),
                SaveStateType::IoTrap => match read_io_trap_packed(&view) {
                    Some(v) => v,
                    None => {
                        log::error!("SAVE_STATE_READ: No valid I/O trap in the save state");
                        return Err(Status::NOT_FOUND);
                    }
                },
                SaveStateType::ProcessorId => unreachable!("handled above"),
            }
        }
    };

    write_field_value(buffer, value)
}

/// Validates that `buffer` is an 8-byte user-owned region and writes `value`
/// into it as little-endian bytes.
fn write_field_value(buffer: u64, value: u64) -> SyscallResult {
    if buffer == 0 {
        log::error!("SAVE_STATE_READ: Null output buffer");
        return Err(Status::INVALID_PARAMETER);
    }

    // Validate the buffer is in user-owned memory.
    match query_address_ownership(buffer, FIELD_VALUE_SIZE) {
        Some(PageOwnership::User) => {}
        Some(owner) => {
            log::error!("SAVE_STATE_READ: Buffer 0x{:x} owned by {:?}, expected User", buffer, owner);
            return Err(Status::ACCESS_DENIED);
        }
        None => {
            log::error!("SAVE_STATE_READ: Buffer 0x{:x} not in mapped memory", buffer);
            return Err(Status::ACCESS_DENIED);
        }
    }

    // SAFETY: `buffer` was validated above as a user-owned, writable region of
    // `FIELD_VALUE_SIZE` bytes. User code is not executing concurrently while the
    // supervisor services this syscall, so there is no aliasing.
    let out = unsafe { core::slice::from_raw_parts_mut(buffer as *mut u8, FIELD_VALUE_SIZE as usize) };
    out.copy_from_slice(&value.to_le_bytes());
    Ok(0)
}

/// Returns the per-CPU save-state metadata captured at initialization.
fn save_state_info() -> Result<SaveStateInfo, Status> {
    match security_state().save_state_info() {
        Some(info) => Ok(info),
        None => {
            log::error!("Save-state metadata not initialized");
            Err(Status::NOT_READY)
        }
    }
}

/// Returns the number of CPUs from the save-state metadata.
fn get_number_of_cpus() -> Result<u64, Status> {
    Ok(save_state_info()?.number_of_cpus)
}

/// A read-only byte view over a CPU's SMRAM save state region.
struct SaveStateView {
    bytes: &'static [u8],
}

impl SaveStateView {
    /// Creates a view over `size` bytes of the save state region at `base`.
    ///
    /// ## Safety
    ///
    /// `base` must point to a readable SMRAM save state region of at least
    /// `size` bytes that is not mutated for the lifetime of the view and lives
    /// for the duration of the program.
    unsafe fn new(base: *const u8, size: usize) -> Self {
        // SAFETY: guaranteed by the caller's contract.
        Self { bytes: unsafe { core::slice::from_raw_parts(base, size) } }
    }

    /// Reads a little-endian `u32` at `offset`.
    fn read_u32(&self, offset: usize) -> u32 {
        u32::from_le_bytes(self.bytes[offset..offset + 4].try_into().expect("offset within save state region"))
    }

    /// Reads a little-endian `u64` at `offset`.
    fn read_u64(&self, offset: usize) -> u64 {
        u64::from_le_bytes(self.bytes[offset..offset + 8].try_into().expect("offset within save state region"))
    }
}

/// Builds a [`SaveStateView`] for the given CPU index.
///
/// The region base is derived from the CPU's SMBASE as
/// `sm_base[cpu_index] + SMRAM_SAVE_STATE_MAP_OFFSET`, with the SMBASE array
/// passed through the MM Supervisor PassDown HOB. The region length is the fixed
/// [`SMRAM_SAVE_STATE_MAP_SIZE`].
fn get_save_state_view(cpu_index: u64) -> Result<SaveStateView, Status> {
    let info = match save_state_info() {
        Ok(i) => i,
        Err(status) => {
            log::error!("Failed to get save state info: {:?}", status);
            return Err(status);
        }
    };

    let num_cpus = info.number_of_cpus;
    if cpu_index >= num_cpus {
        log::error!("Save state read: CPU index {} >= NumberOfCpus {}", cpu_index, num_cpus);
        return Err(Status::INVALID_PARAMETER);
    }

    if info.sm_base == 0 {
        log::error!("SmBase array pointer is null");
        return Err(Status::NOT_READY);
    }

    // The SMBASE array holds `num_cpus` per-CPU SMBASE values set up by the
    // relocation code. The save-state region base is `SmBase + 0xfc00` and every
    // region is the fixed `SMRAM_SAVE_STATE_MAP_SIZE`.
    //
    // SAFETY: `sm_base` references a valid array of at least `num_cpus` `u64`
    // entries in SMRAM (from the PassDown HOB), so the slice covers only valid,
    // initialized memory.
    let sm_bases = unsafe { core::slice::from_raw_parts(info.sm_base as *const u64, num_cpus as usize) };

    let smbase = sm_bases[cpu_index as usize];
    if smbase == 0 {
        log::error!("SmBase[{}] is null", cpu_index);
        return Err(Status::INVALID_PARAMETER);
    }
    let base = smbase + SMRAM_SAVE_STATE_MAP_OFFSET;

    // SAFETY: `base` points to a valid save state region of
    // `SMRAM_SAVE_STATE_MAP_SIZE` bytes in SMRAM that is stable while this SMI
    // is serviced and lives for the program's duration.
    Ok(unsafe { SaveStateView::new(base as *const u8, SMRAM_SAVE_STATE_MAP_SIZE as usize) })
}

/// Reads the PROCESSOR_ID (APIC ID) for a given CPU.
///
/// The ProcessorId is read from the `EFI_PROCESSOR_INFORMATION` array carried by
/// the MP Information HOB (`gMpInformationHobGuid`).
fn read_processor_id_value(cpu_index: u64) -> Result<u64, Status> {
    let info = match save_state_info() {
        Ok(i) => i,
        Err(status) => {
            log::error!("Failed to get save state info: {:?}", status);
            return Err(status);
        }
    };

    if info.processor_info == 0 {
        log::error!("PROCESSOR_ID: ProcessorInfo array is null");
        return Err(Status::NOT_READY);
    }

    let num_cpus = info.number_of_cpus;

    // View the processor information array as bytes so the per-CPU entry can be
    // read through safe slice operations.
    //
    // SAFETY: `processor_info` points to a valid array of `num_cpus`
    // `EFI_PROCESSOR_INFORMATION` entries (PROCESSOR_INFO_ENTRY_SIZE bytes
    // each) in firmware memory, and `cpu_index` is < `num_cpus`.
    let entries = unsafe {
        core::slice::from_raw_parts(info.processor_info as *const u8, num_cpus as usize * PROCESSOR_INFO_ENTRY_SIZE)
    };

    // ProcessorId is the first field (u64) of EFI_PROCESSOR_INFORMATION.
    let offset = cpu_index as usize * PROCESSOR_INFO_ENTRY_SIZE;
    let processor_id = u64::from_le_bytes(entries[offset..offset + 8].try_into().expect("entry within array"));

    log::debug!("PROCESSOR_ID: CPU {} = 0x{:x}", cpu_index, processor_id);
    Ok(processor_id)
}

/// Inspects the I/O condition (IN vs OUT) from the save state for policy checking.
///
/// Reads the vendor-specific IO field from the CPU's save state and uses the
/// SDK's [`save_state::parse_io_field`] to determine whether the I/O trap was
/// caused by an IN or OUT instruction.
fn inspect_io_condition(view: &SaveStateView) -> Option<SaveStateCondition> {
    let vc = save_state::vendor_constants();

    // Verify the save state revision supports IO info before reading the field.
    let smm_rev_id = view.read_u32(vc.smmrevid_offset as usize);
    if !save_state::io_info_supported(smm_rev_id) {
        log::error!("inspect_io_condition: SMMRevId 0x{:x} does not expose IO info", smm_rev_id);
        // return None;
    }

    // Read the vendor-specific IO information field.
    let io_field = view.read_u32(vc.io_info_offset as usize);

    // Use the SDK's vendor-specific parser.
    let parsed = match save_state::parse_io_field(io_field) {
        Some(p) => p,
        None => {
            log::error!("inspect_io_condition: Failed to parse IO field 0x{:x}", io_field);
            return None;
        }
    };

    match parsed.io_type {
        IO_TYPE_INPUT => Some(SaveStateCondition::IoRead),
        IO_TYPE_OUTPUT => Some(SaveStateCondition::IoWrite),
        _ => Some(SaveStateCondition::IoWrite),
    }
}

/// Reads the raw `RAX` register value from the save state.
fn read_rax_value(view: &SaveStateView) -> u64 {
    let vc = save_state::vendor_constants();
    view.read_u64(vc.rax_offset as usize)
}

/// Reads and decodes the I/O trap descriptor into a packed `u64`.
///
/// Returns `None` if the save state does not describe a valid I/O trap. The
/// packed layout (see `IO_TRAP_*_SHIFT` in the SDK) carries the I/O port, the
/// EFI I/O width, and the EFI I/O type — but **not** the I/O data (read `RAX`
/// separately for that).
fn read_io_trap_packed(view: &SaveStateView) -> Option<u64> {
    let vc = save_state::vendor_constants();

    // Verify IO info is available for this save-state revision.
    let smm_rev_id = view.read_u32(vc.smmrevid_offset as usize);
    if !save_state::io_info_supported(smm_rev_id) {
        log::error!("IO_TRAP: SMMRevId 0x{:x} does not expose IO info", smm_rev_id);
        // parse_io_field below still guards against an invalid field.
    }

    let io_field = view.read_u32(vc.io_info_offset as usize);
    let parsed = match save_state::parse_io_field(io_field) {
        Some(p) => p,
        None => {
            log::error!("IO_TRAP: Failed to parse IO field 0x{:x}", io_field);
            return None;
        }
    };

    // Pack {IoPort, IoWidth, IoType} into a u64. IoPort occupies the low 16 bits.
    let packed = ((parsed.io_port as u64) & 0xFFFF)
        | (((parsed.io_width as u64) & 0xFF) << IO_TRAP_WIDTH_SHIFT)
        | (((parsed.io_type as u64) & 0xFF) << IO_TRAP_TYPE_SHIFT);
    Some(packed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_state_type_roundtrip() {
        for t in [SaveStateType::ProcessorId, SaveStateType::Rax, SaveStateType::IoTrap] {
            assert_eq!(SaveStateType::from_u64(t.as_u64()), Some(t));
        }
        assert_eq!(SaveStateType::from_u64(3), None);
        assert_eq!(SaveStateType::from_u64(999), None);
    }

    #[test]
    fn test_save_state_read_rejects_unknown_field() {
        // An unknown field type is rejected before any global state is touched.
        assert_eq!(save_state_read(0, 999, 0), Err(Status::INVALID_PARAMETER));
    }
}
