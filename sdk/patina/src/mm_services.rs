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

use crate::{
    boot_services::allocation::AllocType,
    efi_types::EfiMemoryType,
    pi::mm_cis::{EfiMmSystemTable, MmiHandlerEntryPoint},
};
use r_efi::efi;
use spin::Once;

#[doc(inline)]
pub use crate::service_table::{MemoryServices, ProtocolPrimitives, ProtocolServices};

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
    //
    // Pool / page allocation (`allocate_pool`, `free_pool`, `allocate_pages`, `free_pages`) is
    // shared with `BootServices` through the [`MemoryServices`] trait — the underlying PI/UEFI
    // allocators are the same, so the signatures and (`unsafe`) safety contracts are defined once
    // there. `StandardMmServices` implements them directly (see the `impl MemoryServices` block
    // below); callers get them by bringing [`MemoryServices`] into scope.

    // ---- Protocol services ----------------------------------------------
    //
    // Typed protocol services (`install_protocol_interface`, `handle_protocol`, `locate_protocol`,
    // ...) are shared with `BootServices` through the [`ProtocolServices`] trait, which is
    // blanket-implemented over the [`ProtocolPrimitives`] raw accessors. `StandardMmServices`
    // implements those primitives directly (see the `impl ProtocolPrimitives` block below); callers
    // get the full typed surface by bringing [`ProtocolServices`] into scope.

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

macro_rules! efi_mm_system_table_fn {
    ($efi_mm_system_table:expr, $fn_name:ident) => {
        $crate::service_table::service_table_fn!($efi_mm_system_table, $fn_name, "MM system table")
    };
}

impl MmServices for StandardMmServices {
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
        let guid_ptr = handler_type.map_or(core::ptr::null(), |g| g as *const efi::Guid);
        // SAFETY: See safety comment in allocate_pool for details on corner cases around external modifications.
        let mmi_manage = unsafe { efi_mm_system_table_fn!(*self.as_mut_ptr(), mmi_manage) };
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mmi_manage` service. The GUID,
        // context, and comm-buffer pointers are passed through unchanged to the dispatcher.
        unsafe { mmi_manage(guid_ptr, context, comm_buffer, comm_buffer_size) }
    }

    fn mmi_handler_register(
        &self,
        handler: MmiHandlerEntryPoint,
        handler_type: Option<&efi::Guid>,
    ) -> Result<efi::Handle, efi::Status> {
        let guid_ptr = handler_type.map_or(core::ptr::null(), |g| g as *const efi::Guid);
        let mut dispatch_handle: efi::Handle = core::ptr::null_mut();
        // SAFETY: See safety comment in allocate_pool for details on corner cases around external modifications.
        let mmi_handler_register = unsafe { efi_mm_system_table_fn!(*self.as_mut_ptr(), mmi_handler_register) };
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mmi_handler_register` service.
        // `dispatch_handle` is a valid out-pointer for the duration of the call.
        let status = unsafe { mmi_handler_register(handler, guid_ptr, &mut dispatch_handle) };
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
        // SAFETY: See safety comment in allocate_pool for details on corner cases around external modifications.
        let mmi_handler_unregister = unsafe { efi_mm_system_table_fn!(*self.as_mut_ptr(), mmi_handler_unregister) };
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mmi_handler_unregister` service.
        let status = unsafe { mmi_handler_unregister(dispatch_handle) };
        if status == efi::Status::SUCCESS { Ok(()) } else { Err(status) }
    }
}

// Pool / page allocation for the MM system table. The signatures and `unsafe` contracts are shared
// with `BootServices` via [`MemoryServices`]; the underlying allocators are the same.
impl MemoryServices for StandardMmServices {
    fn allocate_pool(&self, memory_type: EfiMemoryType, size: usize) -> Result<*mut u8, efi::Status> {
        // SAFETY: The function pointer is read directly from the table through the raw pointer
        // rather than through a long-lived `&EfiMmSystemTable`. If the table were mutated in an
        // interrupt/callback while this read is in progress, a torn pointer value could result
        // (the read is not guaranteed atomic). This relies on the MM system table being published
        // once and then shared read-only with drivers, which holds for the standalone MM environment.
        let mm_allocate_pool = unsafe { efi_mm_system_table_fn!(*self.as_mut_ptr(), mm_allocate_pool) };
        let mut buffer: *mut c_void = core::ptr::null_mut();
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_allocate_pool` service.
        // `buffer` is a valid out-pointer for the duration of the call.
        let status = unsafe { mm_allocate_pool(memory_type.into(), size, &mut buffer) };
        if status == efi::Status::SUCCESS { Ok(buffer as *mut u8) } else { Err(status) }
    }

    unsafe fn free_pool(&self, buffer: *mut u8) -> Result<(), efi::Status> {
        // SAFETY: See safety comment in allocate_pool for details on corner cases around external modifications.
        let mm_free_pool = unsafe { efi_mm_system_table_fn!(*self.as_mut_ptr(), mm_free_pool) };
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_free_pool` service. The caller
        // of this `unsafe fn` guarantees `buffer` is a live, exclusively-owned pool allocation.
        let status = unsafe { mm_free_pool(buffer as *mut c_void) };
        if status == efi::Status::SUCCESS { Ok(()) } else { Err(status) }
    }

    fn allocate_pages(
        &self,
        alloc_type: AllocType,
        memory_type: EfiMemoryType,
        nb_pages: usize,
    ) -> Result<usize, efi::Status> {
        let mut memory_address = match alloc_type {
            AllocType::Address(address) => address,
            AllocType::MaxAddress(address) => address,
            _ => 0,
        };
        // SAFETY: See safety comment in allocate_pool for details on corner cases around external modifications.
        let mm_allocate_pages = unsafe { efi_mm_system_table_fn!(*self.as_mut_ptr(), mm_allocate_pages) };
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_allocate_pages` service.
        // `memory_address` is a valid out-pointer for the duration of the call.
        let status = unsafe {
            mm_allocate_pages(
                alloc_type.into(),
                memory_type.into(),
                nb_pages,
                core::ptr::addr_of_mut!(memory_address) as *mut u64,
            )
        };
        if status == efi::Status::SUCCESS { Ok(memory_address) } else { Err(status) }
    }

    unsafe fn free_pages(&self, address: usize, nb_pages: usize) -> Result<(), efi::Status> {
        // SAFETY: See safety comment in allocate_pool for details on corner cases around external modifications.
        let mm_free_pages = unsafe { efi_mm_system_table_fn!(*self.as_mut_ptr(), mm_free_pages) };
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_free_pages` service. The caller
        // of this `unsafe fn` guarantees `address`/`nb_pages` match a live, exclusively-owned range.
        let status = unsafe { mm_free_pages(address as u64, nb_pages) };
        if status == efi::Status::SUCCESS { Ok(()) } else { Err(status) }
    }
}

// Raw protocol primitives for the MM system table. The typed [`ProtocolServices`] surface — shared
// with `BootServices` — is blanket-implemented on top of these.
impl ProtocolPrimitives for StandardMmServices {
    /// # Safety
    ///
    /// If `interface` is non-null, it must remain valid for the lifetime of the installed interface.
    unsafe fn install_protocol_interface_unchecked(
        &self,
        handle: Option<efi::Handle>,
        protocol: &'static efi::Guid,
        interface: *mut c_void,
    ) -> Result<efi::Handle, efi::Status> {
        let mut handle = handle.unwrap_or_default();
        // SAFETY: See safety comment in allocate_pool for details on corner cases around external modifications.
        let mm_install_protocol_interface =
            unsafe { efi_mm_system_table_fn!(*self.as_mut_ptr(), mm_install_protocol_interface) };
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_install_protocol_interface`
        // service. `handle` is a local out-pointer; the caller guarantees `interface` validity.
        let status = unsafe {
            mm_install_protocol_interface(
                &raw mut handle,
                protocol as *const efi::Guid as *mut efi::Guid,
                efi::NATIVE_INTERFACE,
                interface,
            )
        };
        if status == efi::Status::SUCCESS { Ok(handle) } else { Err(status) }
    }

    unsafe fn uninstall_protocol_interface_unchecked(
        &self,
        handle: efi::Handle,
        protocol: &'static efi::Guid,
        interface: *mut c_void,
    ) -> Result<(), efi::Status> {
        // SAFETY: See safety comment in allocate_pool for details on corner cases around external modifications.
        let mm_uninstall_protocol_interface =
            unsafe { efi_mm_system_table_fn!(*self.as_mut_ptr(), mm_uninstall_protocol_interface) };
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_uninstall_protocol_interface`
        // service. The caller of this `unsafe fn` guarantees `interface` matches the installed pointer.
        let status = unsafe {
            mm_uninstall_protocol_interface(handle, protocol as *const efi::Guid as *mut efi::Guid, interface)
        };
        if status == efi::Status::SUCCESS { Ok(()) } else { Err(status) }
    }

    unsafe fn handle_protocol_unchecked(
        &self,
        handle: efi::Handle,
        protocol: &efi::Guid,
    ) -> Result<*mut c_void, efi::Status> {
        // SAFETY: See safety comment in allocate_pool for details on corner cases around external modifications.
        let mm_handle_protocol = unsafe { efi_mm_system_table_fn!(*self.as_mut_ptr(), mm_handle_protocol) };
        let mut interface: *mut c_void = core::ptr::null_mut();
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_handle_protocol` service.
        // `interface` is a valid out-pointer for the duration of the call.
        let status =
            unsafe { mm_handle_protocol(handle, protocol as *const efi::Guid as *mut efi::Guid, &mut interface) };
        if status == efi::Status::SUCCESS { Ok(interface) } else { Err(status) }
    }

    unsafe fn locate_protocol_unchecked(
        &self,
        protocol: &'static efi::Guid,
        registration: *mut c_void,
    ) -> Result<*mut c_void, efi::Status> {
        // SAFETY: See safety comment in allocate_pool for details on corner cases around external modifications.
        let mm_locate_protocol = unsafe { efi_mm_system_table_fn!(*self.as_mut_ptr(), mm_locate_protocol) };
        let mut interface: *mut c_void = core::ptr::null_mut();
        // SAFETY: Intrinsically unsafe FFI call into the C-provided `mm_locate_protocol` service.
        // `interface` is a valid out-pointer for the duration of the call. `registration` is passed
        // through unchanged (it may be null when no registration is in use).
        let status =
            unsafe { mm_locate_protocol(protocol as *const efi::Guid as *mut efi::Guid, registration, &mut interface) };
        if status == efi::Status::SUCCESS { Ok(interface) } else { Err(status) }
    }
}

#[cfg(test)]
#[coverage(off)]
mod tests {
    use super::*;
    use core::{mem::MaybeUninit, ptr};

    const GUID_BYTES: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    static GUID: efi::Guid = efi::Guid::from_bytes(&GUID_BYTES);

    /// Builds a `StandardMmServices` backed by a zeroed `EfiMmSystemTable` in which only the named
    /// function-pointer fields are populated with the given test thunks. Mirrors the `boot_services!`
    /// helper in `boot_services.rs`.
    macro_rules! mm_system_table {
        ($($field:ident = $fn:ident),*) => {{
            // SAFETY: Test-only. A zeroed MM system table is created and only the specified function
            // pointers are populated with valid test implementations. Unset fields remain null; the
            // `StandardMmServices` null-check guards against invoking them.
            let table = Box::leak(Box::new(unsafe {
                #[allow(unused_mut)]
                let mut t = MaybeUninit::<EfiMmSystemTable>::zeroed();
                $(
                t.assume_init_mut().$field = $fn;
                )*
                t.assume_init()
            }));
            StandardMmServices::new(table)
        }};
    }

    /// A no-op MMI handler used as a test value for `mmi_handler_register`.
    extern "efiapi" fn dummy_handler(
        _dispatch_handle: efi::Handle,
        _context: *const c_void,
        _comm_buffer: *mut c_void,
        _comm_buffer_size: *mut usize,
    ) -> efi::Status {
        efi::Status::SUCCESS
    }

    #[test]
    #[should_panic(expected = "StandardMmServices is not initialized!")]
    fn test_accessing_uninit_mm_services_should_panic() {
        let mm = StandardMmServices::new_uninit();
        mm.as_mut_ptr();
    }

    #[test]
    fn test_debug_print_works_before_init() {
        let mm = StandardMmServices::new_uninit();
        let output = format!("{mm:?}");
        assert!(output.contains("Not Initialized"));
    }

    #[test]
    fn test_debug_print_works_after_init() {
        let mm = mm_system_table!();
        let output = format!("{mm:?}");
        assert!(output.contains("StandardMmServices"));
    }

    #[test]
    fn test_clone_uninit() {
        let mm = StandardMmServices::new_uninit();
        let clone = mm.clone();
        assert!(!clone.is_init());
    }

    #[test]
    fn test_clone_initialized() {
        let mm = mm_system_table!();
        let clone = mm.clone();
        assert!(clone.is_init());
        assert_eq!(mm.as_mut_ptr(), clone.as_mut_ptr());
    }

    // ---- allocate_pool ----

    #[test]
    #[should_panic = "MM system table function mm_allocate_pool is not initialized."]
    fn test_allocate_pool_not_init() {
        let mm = mm_system_table!();
        let _ = mm.allocate_pool(EfiMemoryType::BootServicesData, 16);
    }

    #[test]
    fn test_allocate_pool() {
        let mm = mm_system_table!(mm_allocate_pool = efi_allocate_pool);

        extern "efiapi" fn efi_allocate_pool(
            pool_type: efi::MemoryType,
            size: usize,
            buffer: *mut *mut c_void,
        ) -> efi::Status {
            assert_eq!(4, pool_type);
            assert_eq!(16, size);
            // SAFETY: test mock - writing the output pointer parameter.
            unsafe { ptr::write(buffer, 0x1000 as *mut c_void) };
            efi::Status::SUCCESS
        }

        assert_eq!(mm.allocate_pool(EfiMemoryType::BootServicesData, 16), Ok(0x1000 as *mut u8));
    }

    #[test]
    fn test_allocate_pool_err() {
        let mm = mm_system_table!(mm_allocate_pool = efi_allocate_pool);

        extern "efiapi" fn efi_allocate_pool(
            _pool_type: efi::MemoryType,
            _size: usize,
            _buffer: *mut *mut c_void,
        ) -> efi::Status {
            efi::Status::OUT_OF_RESOURCES
        }

        assert_eq!(mm.allocate_pool(EfiMemoryType::BootServicesData, 16), Err(efi::Status::OUT_OF_RESOURCES));
    }

    // ---- free_pool ----

    #[test]
    #[should_panic = "MM system table function mm_free_pool is not initialized."]
    fn test_free_pool_not_init() {
        let mm = mm_system_table!();
        // SAFETY: test code - expected to panic before the FFI call is reached.
        let _ = unsafe { mm.free_pool(0x1000 as *mut u8) };
    }

    #[test]
    fn test_free_pool() {
        let mm = mm_system_table!(mm_free_pool = efi_free_pool);

        extern "efiapi" fn efi_free_pool(buffer: *mut c_void) -> efi::Status {
            assert_eq!(0x1000 as *mut c_void, buffer);
            efi::Status::SUCCESS
        }

        // SAFETY: test code - 0x1000 is the buffer the mock expects to free.
        assert_eq!(unsafe { mm.free_pool(0x1000 as *mut u8) }, Ok(()));
    }

    // ---- allocate_pages ----

    #[test]
    #[should_panic = "MM system table function mm_allocate_pages is not initialized."]
    fn test_allocate_pages_not_init() {
        let mm = mm_system_table!();
        let _ = mm.allocate_pages(AllocType::AnyPage, EfiMemoryType::BootServicesData, 2);
    }

    #[test]
    fn test_allocate_pages() {
        let mm = mm_system_table!(mm_allocate_pages = efi_allocate_pages);

        extern "efiapi" fn efi_allocate_pages(
            alloc_type: efi::AllocateType,
            memory_type: efi::MemoryType,
            pages: usize,
            memory: *mut efi::PhysicalAddress,
        ) -> efi::Status {
            assert_eq!(0, alloc_type);
            assert_eq!(4, memory_type);
            assert_eq!(2, pages);
            // SAFETY: test mock - writing the output address parameter.
            unsafe { ptr::write(memory, 0x2000) };
            efi::Status::SUCCESS
        }

        assert_eq!(mm.allocate_pages(AllocType::AnyPage, EfiMemoryType::BootServicesData, 2), Ok(0x2000));
    }

    // ---- free_pages ----

    #[test]
    #[should_panic = "MM system table function mm_free_pages is not initialized."]
    fn test_free_pages_not_init() {
        let mm = mm_system_table!();
        // SAFETY: test code - expected to panic before the FFI call is reached.
        let _ = unsafe { mm.free_pages(0x2000, 2) };
    }

    #[test]
    fn test_free_pages() {
        let mm = mm_system_table!(mm_free_pages = efi_free_pages);

        extern "efiapi" fn efi_free_pages(memory: efi::PhysicalAddress, pages: usize) -> efi::Status {
            assert_eq!(0x2000, memory);
            assert_eq!(2, pages);
            efi::Status::SUCCESS
        }

        // SAFETY: test code - 0x2000/2 is the range the mock expects to free.
        assert_eq!(unsafe { mm.free_pages(0x2000, 2) }, Ok(()));
    }

    // ---- install_protocol_interface ----

    #[test]
    #[should_panic = "MM system table function mm_install_protocol_interface is not initialized."]
    fn test_install_protocol_interface_not_init() {
        let mm = mm_system_table!();
        // SAFETY: test code - expected to panic before the FFI call is reached.
        let _ = unsafe { mm.install_protocol_interface_unchecked(None, &GUID, ptr::null_mut()) };
    }

    #[test]
    fn test_install_protocol_interface_unchecked() {
        let mm = mm_system_table!(mm_install_protocol_interface = efi_install);

        extern "efiapi" fn efi_install(
            handle: *mut efi::Handle,
            guid: *mut efi::Guid,
            interface_type: efi::InterfaceType,
            interface: *mut c_void,
        ) -> efi::Status {
            // SAFETY: test mock - a `None` handle becomes a null default.
            assert_eq!(ptr::null_mut(), unsafe { ptr::read(handle) });
            // SAFETY: test mock - reading the protocol GUID to verify it.
            assert_eq!(&GUID_BYTES, unsafe { ptr::read(guid) }.as_bytes());
            assert_eq!(efi::NATIVE_INTERFACE, interface_type);
            assert_eq!(0x55 as *mut c_void, interface);
            // SAFETY: test mock - writing the resulting handle out-parameter.
            unsafe { ptr::write(handle, 17_usize as efi::Handle) };
            efi::Status::SUCCESS
        }

        // SAFETY: test code - all pointers are valid test values for the duration of the call.
        let result = unsafe { mm.install_protocol_interface_unchecked(None, &GUID, 0x55 as *mut c_void) };
        assert_eq!(result.map(|h| h as usize), Ok(17));
    }

    // ---- uninstall_protocol_interface ----

    #[test]
    #[should_panic = "MM system table function mm_uninstall_protocol_interface is not initialized."]
    fn test_uninstall_protocol_interface_not_init() {
        let mm = mm_system_table!();
        // SAFETY: test code - expected to panic before the FFI call is reached.
        let _ = unsafe { mm.uninstall_protocol_interface_unchecked(ptr::null_mut(), &GUID, ptr::null_mut()) };
    }

    #[test]
    fn test_uninstall_protocol_interface_unchecked() {
        let mm = mm_system_table!(mm_uninstall_protocol_interface = efi_uninstall);

        extern "efiapi" fn efi_uninstall(
            handle: efi::Handle,
            guid: *mut efi::Guid,
            interface: *mut c_void,
        ) -> efi::Status {
            assert_eq!(1, handle as usize);
            // SAFETY: test mock - reading the protocol GUID to verify it.
            assert_eq!(&GUID_BYTES, unsafe { ptr::read(guid) }.as_bytes());
            assert_eq!(0x77 as *mut c_void, interface);
            efi::Status::SUCCESS
        }

        // SAFETY: test code - all pointers are valid test values for the duration of the call.
        let result =
            unsafe { mm.uninstall_protocol_interface_unchecked(1_usize as efi::Handle, &GUID, 0x77 as *mut c_void) };
        assert_eq!(result, Ok(()));
    }

    // ---- handle_protocol ----

    #[test]
    #[should_panic = "MM system table function mm_handle_protocol is not initialized."]
    fn test_handle_protocol_not_init() {
        let mm = mm_system_table!();
        // SAFETY: test code - expected to panic before the FFI call is reached.
        let _ = unsafe { mm.handle_protocol_unchecked(ptr::null_mut(), &GUID) };
    }

    #[test]
    fn test_handle_protocol() {
        let mm = mm_system_table!(mm_handle_protocol = efi_handle_protocol);

        extern "efiapi" fn efi_handle_protocol(
            handle: efi::Handle,
            guid: *mut efi::Guid,
            interface: *mut *mut c_void,
        ) -> efi::Status {
            assert_eq!(1, handle as usize);
            // SAFETY: test mock - reading the protocol GUID to verify it.
            assert_eq!(&GUID_BYTES, unsafe { ptr::read(guid) }.as_bytes());
            // SAFETY: test mock - writing the output interface pointer.
            unsafe { ptr::write(interface, 0x99 as *mut c_void) };
            efi::Status::SUCCESS
        }

        // SAFETY: test code - calling handle_protocol with valid test values.
        let result = unsafe { mm.handle_protocol_unchecked(1_usize as efi::Handle, &GUID) };
        assert_eq!(result, Ok(0x99 as *mut c_void));
    }

    // ---- locate_protocol ----

    #[test]
    #[should_panic = "MM system table function mm_locate_protocol is not initialized."]
    fn test_locate_protocol_not_init() {
        let mm = mm_system_table!();
        // SAFETY: test code - expected to panic before the FFI call is reached.
        let _ = unsafe { mm.locate_protocol_unchecked(&GUID, ptr::null_mut()) };
    }

    #[test]
    fn test_locate_protocol() {
        let mm = mm_system_table!(mm_locate_protocol = efi_locate_protocol);

        extern "efiapi" fn efi_locate_protocol(
            guid: *mut efi::Guid,
            registration: *mut c_void,
            interface: *mut *mut c_void,
        ) -> efi::Status {
            // SAFETY: test mock - reading the protocol GUID to verify it.
            assert_eq!(&GUID_BYTES, unsafe { ptr::read(guid) }.as_bytes());
            assert!(registration.is_null());
            // SAFETY: test mock - writing the output interface pointer.
            unsafe { ptr::write(interface, 0xAB as *mut c_void) };
            efi::Status::SUCCESS
        }

        // SAFETY: test code - calling locate_protocol with a valid test GUID.
        let result = unsafe { mm.locate_protocol_unchecked(&GUID, ptr::null_mut()) };
        assert_eq!(result, Ok(0xAB as *mut c_void));
    }

    // ---- mmi_manage ----

    #[test]
    #[should_panic = "MM system table function mmi_manage is not initialized."]
    fn test_mmi_manage_not_init() {
        let mm = mm_system_table!();
        // SAFETY: test code - expected to panic before the FFI call is reached.
        let _ = unsafe { mm.mmi_manage(None, ptr::null(), ptr::null_mut(), ptr::null_mut()) };
    }

    #[test]
    fn test_mmi_manage() {
        let mm = mm_system_table!(mmi_manage = efi_mmi_manage);

        extern "efiapi" fn efi_mmi_manage(
            handler_type: *const efi::Guid,
            context: *const c_void,
            comm_buffer: *mut c_void,
            comm_buffer_size: *mut usize,
        ) -> efi::Status {
            // SAFETY: test mock - reading the handler-type GUID to verify it.
            assert_eq!(&GUID_BYTES, unsafe { ptr::read(handler_type) }.as_bytes());
            assert_eq!(0x11 as *const c_void, context);
            assert_eq!(0x22 as *mut c_void, comm_buffer);
            assert_eq!(0x33 as *mut usize, comm_buffer_size);
            efi::Status::SUCCESS
        }

        // SAFETY: test code - pointers are opaque test values that the mock does not dereference.
        let status =
            unsafe { mm.mmi_manage(Some(&GUID), 0x11 as *const c_void, 0x22 as *mut c_void, 0x33 as *mut usize) };
        assert_eq!(status, efi::Status::SUCCESS);
    }

    // ---- mmi_handler_register ----

    #[test]
    #[should_panic = "MM system table function mmi_handler_register is not initialized."]
    fn test_mmi_handler_register_not_init() {
        let mm = mm_system_table!();
        let _ = mm.mmi_handler_register(dummy_handler, None);
    }

    #[test]
    fn test_mmi_handler_register() {
        let mm = mm_system_table!(mmi_handler_register = efi_register);

        extern "efiapi" fn efi_register(
            handler: MmiHandlerEntryPoint,
            handler_type: *const efi::Guid,
            dispatch_handle: *mut efi::Handle,
        ) -> efi::Status {
            assert_eq!(dummy_handler as *const () as usize, handler as usize);
            // SAFETY: test mock - reading the handler-type GUID to verify it.
            assert_eq!(&GUID_BYTES, unsafe { ptr::read(handler_type) }.as_bytes());
            // SAFETY: test mock - writing the output dispatch handle.
            unsafe { ptr::write(dispatch_handle, 7_usize as efi::Handle) };
            efi::Status::SUCCESS
        }

        let result = mm.mmi_handler_register(dummy_handler, Some(&GUID));
        assert_eq!(result.map(|h| h as usize), Ok(7));
    }

    // ---- mmi_handler_unregister ----

    #[test]
    #[should_panic = "MM system table function mmi_handler_unregister is not initialized."]
    fn test_mmi_handler_unregister_not_init() {
        let mm = mm_system_table!();
        // SAFETY: test code - expected to panic before the FFI call is reached.
        let _ = unsafe { mm.mmi_handler_unregister(ptr::null_mut()) };
    }

    #[test]
    fn test_mmi_handler_unregister() {
        let mm = mm_system_table!(mmi_handler_unregister = efi_unregister);

        extern "efiapi" fn efi_unregister(dispatch_handle: efi::Handle) -> efi::Status {
            assert_eq!(9, dispatch_handle as usize);
            efi::Status::SUCCESS
        }

        // SAFETY: test code - calling unregister with a valid test handle.
        let result = unsafe { mm.mmi_handler_unregister(9_usize as efi::Handle) };
        assert_eq!(result, Ok(()));
    }
}
