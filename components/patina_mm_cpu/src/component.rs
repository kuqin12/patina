//! The MM CPU component.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use core::ffi::c_void;

use patina::{
    component::component,
    error::{EfiError, Result},
    management_mode::supervisor::{IO_TRAP_TYPE_SHIFT, IO_TRAP_WIDTH_SHIFT, SaveStateType},
    mm_services::MmServiceProvider,
};
use r_efi::efi;

use crate::{
    protocol::{self, MM_CPU_PROTOCOL_GUID, MmCpuProtocol},
    save_state::read_field,
};

/// The static `EFI_MM_CPU_PROTOCOL` interface installed by [`MmCpuComponent`].
///
/// Function pointers are `Send`/`Sync`, so this is safe to place in a `static`
/// and hand to consumers as a stable interface pointer.
static MM_CPU_PROTOCOL: MmCpuProtocol =
    MmCpuProtocol { read_save_state: mm_cpu_read_save_state, write_save_state: mm_cpu_write_save_state };

/// Component that produces the `EFI_MM_CPU_PROTOCOL` in the MM User Core.
///
/// Replaces the C `MmSupervisorPkg/Drivers/MmSupervisedCpu` driver. It installs
/// `gEfiMmCpuProtocolGuid`; its `ReadSaveState` forwards save-state reads to the
/// MM Supervisor, which owns SMRAM and enforces the save-state security policy.
#[derive(Default)]
pub struct MmCpuComponent;

impl MmCpuComponent {
    /// Creates a new [`MmCpuComponent`].
    pub fn new() -> Self {
        Self
    }
}

#[component]
impl MmCpuComponent {
    fn entry_point(self, mm: MmServiceProvider) -> Result<()> {
        let mut handle: efi::Handle = core::ptr::null_mut();

        // SAFETY: `MM_CPU_PROTOCOL` is a valid, `'static` interface; `handle` is a
        // valid in/out local initialized to null so a new handle is allocated.
        let result = unsafe {
            mm.install_protocol_interface(
                &mut handle,
                &MM_CPU_PROTOCOL_GUID,
                efi::NATIVE_INTERFACE,
                &MM_CPU_PROTOCOL as *const MmCpuProtocol as *mut c_void,
            )
        };

        match result {
            Ok(()) => {
                log::info!("Installed EFI_MM_CPU_PROTOCOL on handle {handle:p}");
                Ok(())
            }
            Err(status) => {
                log::error!("Failed to install EFI_MM_CPU_PROTOCOL: {status:?}");
                EfiError::status_to_result(status)
            }
        }
    }
}

/// `EFI_MM_CPU_PROTOCOL.ReadSaveState` implementation.
///
/// Only `PROCESSOR_ID`, `RAX`, and `IO` are supported; any other register is not
/// defined for the save state and returns `EFI_NOT_FOUND` (per the PI spec).
///
/// Each supported field is read from the MM Supervisor (which owns SMRAM and
/// enforces the save-state policy). The composite `EFI_MM_SAVE_STATE_IO_INFO`
/// for the `IO` pseudo-register is assembled here from separate `IoTrap` and
/// `Rax` reads.
extern "efiapi" fn mm_cpu_read_save_state(
    this: *const MmCpuProtocol,
    width: usize,
    register: u32,
    cpu_index: usize,
    buffer: *mut c_void,
) -> efi::Status {
    if this.is_null() || buffer.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }

    match register {
        protocol::REGISTER_PROCESSOR_ID => write_scalar(cpu_index, SaveStateType::ProcessorId, width, buffer),
        protocol::REGISTER_RAX => write_scalar(cpu_index, SaveStateType::Rax, width, buffer),
        protocol::REGISTER_IO => write_io_info(cpu_index, width, buffer),
        // Any other register is not defined for this save state.
        _ => efi::Status::NOT_FOUND,
    }
}

/// Reads a scalar field and writes its low `width` bytes (capped at 8) to `buffer`.
fn write_scalar(cpu_index: usize, field: SaveStateType, width: usize, buffer: *mut c_void) -> efi::Status {
    let value = match read_field(cpu_index, field) {
        Ok(v) => v,
        Err(status) => return status,
    };

    let n = width.min(core::mem::size_of::<u64>());
    // SAFETY: `buffer` is a caller-provided output of at least `width` bytes and
    // `n <= width`; the source is an 8-byte little-endian value.
    unsafe {
        core::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), buffer.cast::<u8>(), n);
    }
    efi::Status::SUCCESS
}

/// Reads the I/O trap descriptor (and, for a write, the I/O data) and assembles
/// an `EFI_MM_SAVE_STATE_IO_INFO` into `buffer`.
fn write_io_info(cpu_index: usize, width: usize, buffer: *mut c_void) -> efi::Status {
    if width < protocol::IO_INFO_SIZE {
        return efi::Status::INVALID_PARAMETER;
    }

    // The trap descriptor is packed as {IoPort:16, IoWidth:8, IoType:8}.
    let packed = match read_field(cpu_index, SaveStateType::IoTrap) {
        Ok(v) => v,
        Err(status) => return status,
    };
    let io_port = (packed & 0xFFFF) as u16;
    let io_width = ((packed >> IO_TRAP_WIDTH_SHIFT) & 0xFF) as u32;
    let io_type = ((packed >> IO_TRAP_TYPE_SHIFT) & 0xFF) as u32;

    // The I/O data is only meaningful for an OUT (write); for an IN it is not yet
    // present, so report zero rather than reading (and being denied) RAX.
    let io_data = if io_type == protocol::IO_TYPE_OUTPUT {
        let rax = match read_field(cpu_index, SaveStateType::Rax) {
            Ok(v) => v,
            Err(status) => return status,
        };
        let byte_count = 1usize << io_width;
        let mask =
            if byte_count >= core::mem::size_of::<u64>() { u64::MAX } else { (1u64 << (byte_count * 8)) - 1 };
        rax & mask
    } else {
        0
    };

    // Serialize EFI_MM_SAVE_STATE_IO_INFO { IoData@0, IoPort@8, IoWidth@12, IoType@16 }.
    let base = buffer.cast::<u8>();
    // SAFETY: `width >= IO_INFO_SIZE` (24), so every field offset is within the
    // caller's buffer. Unaligned writes avoid any alignment assumption on `buffer`.
    unsafe {
        core::ptr::write_unaligned(base as *mut u64, io_data);
        core::ptr::write_unaligned(base.add(8) as *mut u16, io_port);
        core::ptr::write_unaligned(base.add(12) as *mut u32, io_width);
        core::ptr::write_unaligned(base.add(16) as *mut u32, io_type);
    }
    efi::Status::SUCCESS
}

/// `EFI_MM_CPU_PROTOCOL.WriteSaveState` implementation.
///
/// Writing the MM save state is not supported, mirroring the C `MmSupervisedCpu`
/// driver, which installs a NULL `WriteSaveState`.
extern "efiapi" fn mm_cpu_write_save_state(
    _this: *const MmCpuProtocol,
    _width: usize,
    _register: u32,
    _cpu_index: usize,
    _buffer: *const c_void,
) -> efi::Status {
    efi::Status::UNSUPPORTED
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-null protocol pointer for tests. The read path never dereferences
    /// `this`; it only forwards its address to the (stubbed) syscall.
    fn dummy_this() -> *const MmCpuProtocol {
        &MM_CPU_PROTOCOL as *const MmCpuProtocol
    }

    #[test]
    fn test_mm_cpu_read_save_state_rejects_null_this() {
        let mut buf = [0u8; 8];
        let status =
            mm_cpu_read_save_state(core::ptr::null(), 8, protocol::REGISTER_RAX, 0, buf.as_mut_ptr().cast());
        assert_eq!(status, efi::Status::INVALID_PARAMETER);
    }

    #[test]
    fn test_mm_cpu_read_save_state_rejects_null_buffer() {
        let status =
            mm_cpu_read_save_state(dummy_this(), 8, protocol::REGISTER_RAX, 0, core::ptr::null_mut());
        assert_eq!(status, efi::Status::INVALID_PARAMETER);
    }

    #[test]
    fn test_mm_cpu_read_save_state_unsupported_register_is_not_found() {
        let mut buf = [0u8; 8];
        // 39 = RBX — a valid PI register, but not one this component exposes.
        let status = mm_cpu_read_save_state(dummy_this(), 8, 39, 0, buf.as_mut_ptr().cast());
        assert_eq!(status, efi::Status::NOT_FOUND);
    }

    #[test]
    fn test_mm_cpu_read_save_state_supported_register_forwards_to_syscall() {
        let mut buf = [0u8; protocol::IO_INFO_SIZE];
        // A supported scalar register passes the whitelist and forwards to the
        // syscall wrapper, which on the host (non-UEFI) target is a stub reporting
        // the operation as unsupported. Reaching UNSUPPORTED proves the forward path.
        for register in [protocol::REGISTER_PROCESSOR_ID, protocol::REGISTER_RAX] {
            let status = mm_cpu_read_save_state(dummy_this(), 8, register, 0, buf.as_mut_ptr().cast());
            assert_eq!(status, efi::Status::UNSUPPORTED);
        }

        // IO needs a full IO_INFO buffer; it forwards on the initial IoTrap read.
        let status =
            mm_cpu_read_save_state(dummy_this(), protocol::IO_INFO_SIZE, protocol::REGISTER_IO, 0, buf.as_mut_ptr().cast());
        assert_eq!(status, efi::Status::UNSUPPORTED);
    }

    #[test]
    fn test_mm_cpu_read_save_state_io_rejects_small_buffer() {
        let mut buf = [0u8; 8];
        // A buffer smaller than EFI_MM_SAVE_STATE_IO_INFO is rejected before any syscall.
        let status = mm_cpu_read_save_state(dummy_this(), 8, protocol::REGISTER_IO, 0, buf.as_mut_ptr().cast());
        assert_eq!(status, efi::Status::INVALID_PARAMETER);
    }

    #[test]
    fn test_mm_cpu_write_save_state_is_unsupported() {
        let buf = [0u8; 8];
        let status = mm_cpu_write_save_state(dummy_this(), 8, protocol::REGISTER_RAX, 0, buf.as_ptr().cast());
        assert_eq!(status, efi::Status::UNSUPPORTED);
    }
}
