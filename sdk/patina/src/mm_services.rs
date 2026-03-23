//! MM (Management Mode) Services type definitions and trait.
//!
//! This module provides the Rust definitions for the PI `EFI_MM_SYSTEM_TABLE`
//! and an `MmServices` trait that wraps the raw C function-pointer table with
//! safe Rust method signatures, following the same pattern as
//! [`boot_services::BootServices`](crate::boot_services::BootServices).
//!
//! ## Layout
//!
//! * [`EfiMmSystemTable`] — `#[repr(C)]` struct matching the C
//!   `_EFI_MM_SYSTEM_TABLE` layout from `PiMmCis.h`.
//! * [`MmServices`] — Safe Rust trait exposing the system-table services.
//! * [`StandardMmServices`] — Concrete wrapper around `*mut EfiMmSystemTable`
//!   that implements `MmServices` by calling through the function pointers.
//!
//! Cores (e.g., `patina_mm_user_core`) allocate an `EfiMmSystemTable`, populate
//! its function pointers with their own `extern "efiapi"` thunks, and hand the
//! raw pointer to dispatched MM drivers. Drivers that want safe access can wrap
//! it in a `StandardMmServices`.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use core::ffi::c_void;

use crate::pi::mm_cis::{EfiMmSystemTable, MmiHandlerEntryPoint};
use r_efi::efi;
use spin::Once;

/// Wrapper around a raw `*mut EfiMmSystemTable` pointer that implements
/// [`MmServices`] by calling through the C function-pointer table.
///
/// This is the MM equivalent of
/// [`StandardBootServices`](crate::boot_services::StandardBootServices).
pub struct StandardMmServices {
    efi_mm_system_table: Once<*mut EfiMmSystemTable>,
}

// SAFETY: With efi_mm_system_table being handed off to the c MM drivers, this
// is intrinsically unsafe. The only guarantee is that the rust environment will
// only setup the content once and then never modify it again.
unsafe impl Sync for StandardMmServices {}
// SAFETY: Same as above...
unsafe impl Send for StandardMmServices {}

impl StandardMmServices {
    /// Create a new `StandardMmServices` from an existing system table pointer.
    pub fn new(mm_system_table: *mut EfiMmSystemTable) -> Self {
        let this = Self::new_uninit();
        this.init(mm_system_table);
        this
    }

    /// Create an uninitialized instance.
    pub const fn new_uninit() -> Self {
        Self { efi_mm_system_table: Once::new() }
    }

    /// Initialize with the given system table pointer.
    pub fn init(&self, mm_system_table: *mut EfiMmSystemTable) {
        self.efi_mm_system_table.call_once(|| mm_system_table);
    }

    /// Returns `true` if the instance has been initialized.
    pub fn is_init(&self) -> bool {
        self.efi_mm_system_table.is_completed()
    }

    /// Returns the raw system table pointer (panics if uninitialized).
    pub fn as_mut_ptr(&self) -> *mut EfiMmSystemTable {
        *self.efi_mm_system_table.get().expect("StandardMmServices is not initialized!")
    }

    /// Returns a shared reference to the underlying MM System Table.
    ///
    /// This is the single point where the validated table pointer is
    /// dereferenced; every service method then accesses the function-pointer
    /// fields through this normal Rust reference.
    ///
    /// # Panics
    ///
    /// Panics if the instance has not been initialized.
    fn table(&self) -> &EfiMmSystemTable {
        // SAFETY: `as_mut_ptr()` returns the once-initialized table pointer. The `EfiMmSystemTable`
        // is allocated once for the lifetime of the core and shared read-only with drivers, so a
        // shared reference to it is valid and properly aligned.
        unsafe { &*self.as_mut_ptr() }
    }
}

impl Clone for StandardMmServices {
    fn clone(&self) -> Self {
        if let Some(ptr) = self.efi_mm_system_table.get() {
            StandardMmServices::new(*ptr)
        } else {
            StandardMmServices::new_uninit()
        }
    }
}

impl core::fmt::Debug for StandardMmServices {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if !self.is_init() {
            return f.debug_struct("StandardMmServices").field("table", &"Not Initialized").finish();
        }
        f.debug_struct("StandardMmServices").field("table", &self.as_mut_ptr()).finish()
    }
}

/// Safe Rust interface to the MM System Table services.
///
/// This is the MM analogue of
/// [`BootServices`](crate::boot_services::BootServices).
/// Each method maps 1:1 to a function pointer in [`EfiMmSystemTable`].
pub trait MmServices {
    // ---- Memory services ------------------------------------------------

    /// Allocate pool memory.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmAllocatePool`
    fn allocate_pool(&self, pool_type: efi::MemoryType, size: usize) -> Result<*mut u8, efi::Status>;

    /// Free pool memory.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmFreePool`
    fn free_pool(&self, buffer: *mut u8) -> Result<(), efi::Status>;

    /// Allocate pages.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmAllocatePages`
    fn allocate_pages(
        &self,
        alloc_type: efi::AllocateType,
        memory_type: efi::MemoryType,
        pages: usize,
    ) -> Result<u64, efi::Status>;

    /// Free pages.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmFreePages`
    fn free_pages(&self, memory: u64, pages: usize) -> Result<(), efi::Status>;

    // ---- Protocol services ----------------------------------------------

    /// Install a protocol interface on a handle.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmInstallProtocolInterface`
    ///
    /// # Safety
    ///
    /// `interface` must be a valid pointer to the protocol structure or null.
    unsafe fn install_protocol_interface(
        &self,
        handle: *mut efi::Handle,
        protocol: &efi::Guid,
        interface_type: efi::InterfaceType,
        interface: *mut c_void,
    ) -> Result<(), efi::Status>;

    /// Uninstall a protocol interface from a handle.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmUninstallProtocolInterface`
    ///
    /// # Safety
    ///
    /// `interface` must match the pointer that was installed.
    unsafe fn uninstall_protocol_interface(
        &self,
        handle: efi::Handle,
        protocol: &efi::Guid,
        interface: *mut c_void,
    ) -> Result<(), efi::Status>;

    /// Query a handle for a protocol.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmHandleProtocol`
    ///
    /// # Safety
    ///
    /// The returned pointer must be used carefully to avoid aliasing violations.
    unsafe fn handle_protocol(&self, handle: efi::Handle, protocol: &efi::Guid) -> Result<*mut c_void, efi::Status>;

    /// Locate the first device that supports a protocol.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmLocateProtocol`
    ///
    /// # Safety
    ///
    /// The returned pointer must be used carefully to avoid aliasing violations.
    unsafe fn locate_protocol(&self, protocol: &efi::Guid) -> Result<*mut c_void, efi::Status>;

    // ---- MMI management -------------------------------------------------

    /// Manage (dispatch) an MMI.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmiManage`
    ///
    /// # Safety
    ///
    /// `context`, `comm_buffer`, and `comm_buffer_size` are all optional pointers.
    /// But they must be valid if provided.
    unsafe fn mmi_manage(
        &self,
        handler_type: Option<&efi::Guid>,
        context: *const c_void,
        comm_buffer: *mut c_void,
        comm_buffer_size: *mut usize,
    ) -> efi::Status;

    /// Register an MMI handler.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmiHandlerRegister`
    fn mmi_handler_register(
        &self,
        handler: MmiHandlerEntryPoint,
        handler_type: Option<&efi::Guid>,
    ) -> Result<efi::Handle, efi::Status>;

    /// Unregister an MMI handler.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmiHandlerUnRegister`
    ///
    /// # Safety
    ///
    /// `dispatch_handle` should be a valid handle returned by a previous call to `mmi_handler_register`.
    /// Otherwise, this function will do nothing and return `EFI_NOT_FOUND`.
    /// So this operation is safe to call with an invalid handle, but it will not have any effect.
    unsafe fn mmi_handler_unregister(&self, dispatch_handle: efi::Handle) -> Result<(), efi::Status>;
}

impl MmServices for StandardMmServices {
    fn allocate_pool(&self, pool_type: efi::MemoryType, size: usize) -> Result<*mut u8, efi::Status> {
        let mmst = self.table();
        let mut buffer: *mut c_void = core::ptr::null_mut();
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_allocate_pool` service.
        // `buffer` is a valid out-pointer for the duration of the call.
        let status = unsafe { (mmst.mm_allocate_pool)(pool_type, size, &mut buffer) };
        if status == efi::Status::SUCCESS { Ok(buffer as *mut u8) } else { Err(status) }
    }

    fn free_pool(&self, buffer: *mut u8) -> Result<(), efi::Status> {
        let mmst = self.table();
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_free_pool` service.
        let status = unsafe { (mmst.mm_free_pool)(buffer as *mut c_void) };
        if status == efi::Status::SUCCESS { Ok(()) } else { Err(status) }
    }

    fn allocate_pages(
        &self,
        alloc_type: efi::AllocateType,
        memory_type: efi::MemoryType,
        pages: usize,
    ) -> Result<u64, efi::Status> {
        let mmst = self.table();
        let mut memory: efi::PhysicalAddress = 0;
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_allocate_pages` service.
        // `memory` is a valid out-pointer for the duration of the call.
        let status = unsafe { (mmst.mm_allocate_pages)(alloc_type, memory_type, pages, &mut memory) };
        if status == efi::Status::SUCCESS { Ok(memory) } else { Err(status) }
    }

    fn free_pages(&self, memory: u64, pages: usize) -> Result<(), efi::Status> {
        let mmst = self.table();
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_free_pages` service.
        let status = unsafe { (mmst.mm_free_pages)(memory, pages) };
        if status == efi::Status::SUCCESS { Ok(()) } else { Err(status) }
    }

    unsafe fn install_protocol_interface(
        &self,
        handle: *mut efi::Handle,
        protocol: &efi::Guid,
        interface_type: efi::InterfaceType,
        interface: *mut c_void,
    ) -> Result<(), efi::Status> {
        let mmst = self.table();
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_install_protocol_interface`
        // service. The caller of this `unsafe fn` guarantees `interface` is a valid pointer or null.
        let status = unsafe {
            (mmst.mm_install_protocol_interface)(
                handle,
                protocol as *const efi::Guid as *mut efi::Guid,
                interface_type,
                interface,
            )
        };
        if status == efi::Status::SUCCESS { Ok(()) } else { Err(status) }
    }

    unsafe fn uninstall_protocol_interface(
        &self,
        handle: efi::Handle,
        protocol: &efi::Guid,
        interface: *mut c_void,
    ) -> Result<(), efi::Status> {
        let mmst = self.table();
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_uninstall_protocol_interface`
        // service. The caller of this `unsafe fn` guarantees `interface` matches the installed pointer.
        let status = unsafe {
            (mmst.mm_uninstall_protocol_interface)(handle, protocol as *const efi::Guid as *mut efi::Guid, interface)
        };
        if status == efi::Status::SUCCESS { Ok(()) } else { Err(status) }
    }

    unsafe fn handle_protocol(&self, handle: efi::Handle, protocol: &efi::Guid) -> Result<*mut c_void, efi::Status> {
        let mmst = self.table();
        let mut interface: *mut c_void = core::ptr::null_mut();
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_handle_protocol` service.
        // `interface` is a valid out-pointer for the duration of the call.
        let status = unsafe {
            (mmst.mm_handle_protocol)(handle, protocol as *const efi::Guid as *mut efi::Guid, &mut interface)
        };
        if status == efi::Status::SUCCESS { Ok(interface) } else { Err(status) }
    }

    unsafe fn locate_protocol(&self, protocol: &efi::Guid) -> Result<*mut c_void, efi::Status> {
        let mmst = self.table();
        let mut interface: *mut c_void = core::ptr::null_mut();
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_locate_protocol` service.
        // `interface` is a valid out-pointer for the duration of the call.
        let status = unsafe {
            (mmst.mm_locate_protocol)(
                protocol as *const efi::Guid as *mut efi::Guid,
                core::ptr::null_mut(),
                &mut interface,
            )
        };
        if status == efi::Status::SUCCESS { Ok(interface) } else { Err(status) }
    }

    /// Manage (dispatch) an MMI.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmiManage`
    ///
    /// # Safety
    ///
    /// `context`, `comm_buffer`, and `comm_buffer_size` are all optional pointers.
    /// But they must be valid if provided.
    unsafe fn mmi_manage(
        &self,
        handler_type: Option<&efi::Guid>,
        context: *const c_void,
        comm_buffer: *mut c_void,
        comm_buffer_size: *mut usize,
    ) -> efi::Status {
        let mmst = self.table();
        let guid_ptr = handler_type.map_or(core::ptr::null(), |g| g as *const efi::Guid);
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mmi_manage` service. The GUID,
        // context, and comm-buffer pointers are passed through unchanged to the dispatcher.
        unsafe { (mmst.mmi_manage)(guid_ptr, context, comm_buffer, comm_buffer_size) }
    }

    fn mmi_handler_register(
        &self,
        handler: MmiHandlerEntryPoint,
        handler_type: Option<&efi::Guid>,
    ) -> Result<efi::Handle, efi::Status> {
        let mmst = self.table();
        let guid_ptr = handler_type.map_or(core::ptr::null(), |g| g as *const efi::Guid);
        let mut dispatch_handle: efi::Handle = core::ptr::null_mut();
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mmi_handler_register` service.
        // `dispatch_handle` is a valid out-pointer for the duration of the call.
        let status = unsafe { (mmst.mmi_handler_register)(handler, guid_ptr, &mut dispatch_handle) };
        if status == efi::Status::SUCCESS { Ok(dispatch_handle) } else { Err(status) }
    }

    /// Unregister an MMI handler.
    ///
    /// PI Spec: `EFI_MM_SYSTEM_TABLE.MmiHandlerUnRegister`
    ///
    /// # Safety
    ///
    /// `dispatch_handle` should be a valid handle returned by a previous call to `mmi_handler_register`.
    /// Otherwise, this function will do nothing and return `EFI_NOT_FOUND`.
    /// So this operation is safe to call with an invalid handle, but it will not have any effect.
    unsafe fn mmi_handler_unregister(&self, dispatch_handle: efi::Handle) -> Result<(), efi::Status> {
        let mmst = self.table();
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mmi_handler_unregister` service.
        let status = unsafe { (mmst.mmi_handler_unregister)(dispatch_handle) };
        if status == efi::Status::SUCCESS { Ok(()) } else { Err(status) }
    }
}
