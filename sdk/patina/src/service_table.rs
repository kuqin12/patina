//! Shared helpers for safe access to UEFI / PI service tables of C function pointers.
//!
//! Both [`BootServices`](crate::boot_services) and [`MmServices`](crate::mm_services) wrap a
//! `#[repr(C)]` table whose fields are raw `extern "efiapi"` function pointers. Reading a slot that
//! was never populated (null) and calling it would be undefined behavior, so every accessor first
//! null-checks the slot. That null-check is identical across both tables; only the table type and
//! the label used in the panic message differ. This module centralizes the check.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use core::ffi::c_void;

use r_efi::efi;

use crate::{
    boot_services::{
        allocation::AllocType,
        c_ptr::{CMutRef, PtrMetadata},
        protocol_handler::Registration,
    },
    efi_types::EfiMemoryType,
    uefi_protocol::ProtocolInterface,
};

/// Reads a function-pointer field from a service table, panicking with a descriptive message if
/// the slot is null (i.e. the service was never initialized).
///
/// A macro rather than a function because it must (a) name the table field by token — a struct
/// field cannot be selected by name through a generic function — and (b) null-check a concrete
/// function-pointer type via `f as usize == 0`, which has no generic `where` bound to express.
///
/// `$kind` is a human-readable label for the table (e.g. `"Boot services"` or
/// `"MM system table"`) used verbatim in the panic message.
macro_rules! service_table_fn {
    ($table:expr, $field:ident, $kind:literal) => {{
        match $table.$field {
            f if f as usize == 0 => {
                panic!("{} function {} is not initialized.", $kind, stringify!($field))
            }
            f => f,
        }
    }};
}

pub(crate) use service_table_fn;

/// Shared implementation for the typed `install_protocol_interface` wrappers exposed by
/// [`BootServices`](crate::boot_services::BootServices) and [`MmServices`](crate::mm_services::MmServices).
///
/// The protocol GUID is derived from `T`, and `protocol_interface` (e.g. a `Box<T>`) is leaked to a
/// raw pointer — or null for a zero-sized interface — and handed to `install`. On success the
/// returned [`PtrMetadata`] key can later recover ownership via [`uninstall_protocol_interface`]; on
/// failure ownership of `protocol_interface` is reclaimed and dropped.
pub(crate) fn install_protocol_interface<T, R>(
    protocol_interface: R,
    install: impl FnOnce(&'static efi::Guid, *mut c_void) -> Result<efi::Handle, efi::Status>,
) -> Result<(efi::Handle, PtrMetadata<'static, R>), efi::Status>
where
    R: CMutRef<'static, Type = T> + 'static,
    T: ProtocolInterface + 'static,
{
    let protocol: &'static efi::Guid = &T::PROTOCOL_GUID;
    let key = protocol_interface.metadata();

    let interface = match core::mem::size_of::<T>() {
        0 => core::ptr::null_mut(),
        _ => protocol_interface.into_mut_ptr() as *mut c_void,
    };

    match install(protocol, interface) {
        Ok(handle) => Ok((handle, key)),
        Err(status) => {
            // SAFETY: `install` failed, so the firmware did not take ownership of the interface. The
            // pointer is unchanged from `into_mut_ptr` above, so reclaiming and dropping it is sound.
            _ = unsafe { key.try_into_original_ptr() };
            Err(status)
        }
    }
}

/// Shared implementation for the typed `uninstall_protocol_interface` wrappers exposed by
/// [`BootServices`](crate::boot_services::BootServices) and [`MmServices`](crate::mm_services::MmServices).
///
/// The protocol GUID is derived from `T`, and the pointer recorded in `key` (or null for a
/// zero-sized interface) is handed to `uninstall`. On success ownership of the original interface
/// `R` is recovered and returned.
pub(crate) fn uninstall_protocol_interface<T, R>(
    key: PtrMetadata<'static, R>,
    uninstall: impl FnOnce(&'static efi::Guid, *mut c_void) -> Result<(), efi::Status>,
) -> Result<R, efi::Status>
where
    R: CMutRef<'static, Type = T> + 'static,
    T: ProtocolInterface + 'static,
{
    let protocol: &'static efi::Guid = &T::PROTOCOL_GUID;

    let interface = match core::mem::size_of::<T>() {
        0 => core::ptr::null_mut(),
        _ => key.ptr_value as *mut c_void,
    };

    uninstall(protocol, interface)?;

    // SAFETY: The pointer was leaked on install and kept unchanged; reclaim ownership.
    unsafe { key.try_into_original_ptr() }.ok_or(efi::Status::INVALID_PARAMETER)
}

/// Shared implementation for the typed `handle_protocol` / `locate_protocol` wrappers exposed by
/// [`BootServices`](crate::boot_services::BootServices) and [`MmServices`](crate::mm_services::MmServices).
///
/// `query` performs the raw `*_unchecked` lookup for the GUID derived from `T` and yields the raw
/// interface pointer. A null pointer is mapped to `None`, which is the expected result for a marker
/// (zero-sized) protocol.
///
/// # Safety
///
/// The caller must not create more than one mutable reference to the returned interface.
pub(crate) unsafe fn protocol_ref_maybe_empty<T>(
    query: impl FnOnce(&'static efi::Guid) -> Result<*mut c_void, efi::Status>,
) -> Result<Option<&'static mut T>, efi::Status>
where
    T: ProtocolInterface + 'static,
{
    let protocol: &'static efi::Guid = &T::PROTOCOL_GUID;
    let interface = query(protocol)? as *mut T;
    // SAFETY: `ProtocolInterface` guarantees `interface` (when non-null) points to a valid `T`. The
    // caller upholds the no-aliasing contract documented on this function.
    Ok(unsafe { interface.as_mut() })
}

/// Shared implementation for the typed `handle_protocol` / `locate_protocol` wrappers exposed by
/// [`BootServices`](crate::boot_services::BootServices) and [`MmServices`](crate::mm_services::MmServices).
///
/// Like [`protocol_ref_maybe_empty`] but reconciles the interface pointer with the size of `T`: a
/// non-marker (`size_of::<T>() > 0`) protocol must return a non-null interface, while a marker
/// (zero-sized) protocol must return null — for which a dangling reference is yielded. A mismatch is
/// a programming error, reported via `debug_assert!` and `INVALID_PARAMETER`.
///
/// # Safety
///
/// The caller must not create more than one mutable reference to the returned interface.
pub(crate) unsafe fn protocol_ref<T>(
    query: impl FnOnce(&'static efi::Guid) -> Result<*mut c_void, efi::Status>,
) -> Result<&'static mut T, efi::Status>
where
    T: ProtocolInterface + 'static,
{
    // SAFETY: The caller upholds the no-aliasing contract documented on this function.
    match unsafe { protocol_ref_maybe_empty::<T>(query)? } {
        Some(_) if core::mem::size_of::<T>() == 0 => {
            debug_assert!(
                false,
                "Expect null interface. Type {} need to have a size of 0.",
                core::any::type_name::<T>()
            );
            Err(efi::Status::INVALID_PARAMETER)
        }
        None if core::mem::size_of::<T>() > 0 => {
            debug_assert!(
                false,
                "Expect non null interface. Type {} need to have a size greater than 0.",
                core::any::type_name::<T>()
            );
            Err(efi::Status::INVALID_PARAMETER)
        }
        Some(interface) => Ok(interface),
        // SAFETY: `T` is a ZST; `NonNull::dangling()` is well-aligned and non-null, and reads/writes
        // through it are zero-sized. Each call yields its own reference, avoiding aliasing.
        None => Ok(unsafe { core::ptr::NonNull::<T>::dangling().as_mut() }),
    }
}

/// The raw, table-specific protocol primitives that a UEFI / PI service table must provide.
///
/// This is the lower of two layers shared by
/// [`BootServices`](crate::boot_services::BootServices) and
/// [`MmServices`](crate::mm_services::MmServices): each concrete table (e.g.
/// [`StandardBootServices`](crate::boot_services::StandardBootServices),
/// [`StandardMmServices`](crate::mm_services::StandardMmServices)) implements these four
/// `*_unchecked` accessors by calling through its own C function-pointer table. The typed, checked
/// wrappers in [`ProtocolServices`] are then provided once — for free — on top of these primitives.
pub trait ProtocolPrimitives {
    /// Installs a protocol interface on a handle, returning the (possibly newly created) handle.
    ///
    /// Use [`ProtocolServices::install_protocol_interface`] when possible.
    ///
    /// # Safety
    ///
    /// If `interface` is non-null, it must adhere to the structure associated with `protocol` and
    /// remain valid for the lifetime of the installed interface.
    unsafe fn install_protocol_interface_unchecked(
        &self,
        handle: Option<efi::Handle>,
        protocol: &'static efi::Guid,
        interface: *mut c_void,
    ) -> Result<efi::Handle, efi::Status>;

    /// Removes a protocol interface from a handle.
    ///
    /// Use [`ProtocolServices::uninstall_protocol_interface`] when possible.
    ///
    /// # Safety
    ///
    /// `interface` must be a valid pointer of the type expected by `protocol`, and `handle` must
    /// refer to a handle on which `protocol` was previously installed with `interface`.
    unsafe fn uninstall_protocol_interface_unchecked(
        &self,
        handle: efi::Handle,
        protocol: &'static efi::Guid,
        interface: *mut c_void,
    ) -> Result<(), efi::Status>;

    /// Queries a handle for a protocol interface, returning the raw interface pointer.
    ///
    /// Use [`ProtocolServices::handle_protocol`] when possible.
    ///
    /// # Safety
    ///
    /// The caller must cast the returned pointer to the interface type associated with `protocol`
    /// and must not create multiple mutable references to it.
    unsafe fn handle_protocol_unchecked(
        &self,
        handle: efi::Handle,
        protocol: &efi::Guid,
    ) -> Result<*mut c_void, efi::Status>;

    /// Locates the first interface matching a protocol, returning the raw interface pointer.
    ///
    /// Use [`ProtocolServices::locate_protocol`] when possible.
    ///
    /// # Safety
    ///
    /// The caller must cast the returned pointer to the interface type associated with `protocol`
    /// and must not create multiple mutable references to it.
    unsafe fn locate_protocol_unchecked(
        &self,
        protocol: &'static efi::Guid,
        registration: *mut c_void,
    ) -> Result<*mut c_void, efi::Status>;
}

/// Typed, checked protocol services shared by
/// [`BootServices`](crate::boot_services::BootServices) and
/// [`MmServices`](crate::mm_services::MmServices).
///
/// This is the upper of two layers: a single shared "first layer" (GUID derivation from
/// [`ProtocolInterface`], pointer leak/reclaim, and null reconciliation) implemented once on top of
/// the raw [`ProtocolPrimitives`]. It is blanket-implemented for every `ProtocolPrimitives`
/// implementor, so a concrete table only has to provide the four primitives to gain the full typed
/// surface.
pub trait ProtocolServices: ProtocolPrimitives {
    /// Installs a protocol interface on a handle.
    ///
    /// The protocol GUID is derived from the [`ProtocolInterface`] type `T`, and `protocol_interface`
    /// (e.g. a `Box<T>`) is handed off to the firmware. The returned [`PtrMetadata`] key is required
    /// to later recover ownership via [`ProtocolServices::uninstall_protocol_interface`].
    fn install_protocol_interface<T, R>(
        &self,
        handle: Option<efi::Handle>,
        protocol_interface: R,
    ) -> Result<(efi::Handle, PtrMetadata<'static, R>), efi::Status>
    where
        R: CMutRef<'static, Type = T> + 'static,
        T: ProtocolInterface + 'static,
    {
        install_protocol_interface(protocol_interface, |protocol, interface| {
            // SAFETY: `ProtocolInterface` guarantees the GUID matches the interface type.
            unsafe { self.install_protocol_interface_unchecked(handle, protocol, interface) }
        })
    }

    /// Removes a protocol interface from a handle, recovering the original interface `R`.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // efi::Handle aliases *mut c_void but is an opaque key.
    fn uninstall_protocol_interface<T, R>(
        &self,
        handle: efi::Handle,
        key: PtrMetadata<'static, R>,
    ) -> Result<R, efi::Status>
    where
        R: CMutRef<'static, Type = T> + 'static,
        T: ProtocolInterface + 'static,
    {
        uninstall_protocol_interface(key, |protocol, interface| {
            // SAFETY: `ProtocolInterface` guarantees the GUID matches the interface type.
            unsafe { self.uninstall_protocol_interface_unchecked(handle, protocol, interface) }
        })
    }

    /// Queries a handle for a typed protocol interface.
    ///
    /// The protocol GUID is derived from the [`ProtocolInterface`] type `T`.
    ///
    /// # Safety
    ///
    /// Do not create more than one mutable reference to the returned interface.
    unsafe fn handle_protocol<T>(&self, handle: efi::Handle) -> Result<&'static mut T, efi::Status>
    where
        T: Sized + ProtocolInterface + 'static,
    {
        // SAFETY: The caller guarantees no aliasing of the returned reference.
        unsafe { protocol_ref(|protocol| self.handle_protocol_unchecked(handle, protocol)) }
    }

    /// Queries a handle for a typed protocol interface, mapping a null interface to `None`.
    ///
    /// The protocol GUID is derived from the [`ProtocolInterface`] type `T`.
    ///
    /// # Safety
    ///
    /// Do not create more than one mutable reference to the returned interface.
    unsafe fn handle_protocol_maybe_empty<T>(&self, handle: efi::Handle) -> Result<Option<&'static mut T>, efi::Status>
    where
        T: Sized + ProtocolInterface + 'static,
    {
        // SAFETY: The caller guarantees no aliasing of the returned reference.
        unsafe { protocol_ref_maybe_empty(|protocol| self.handle_protocol_unchecked(handle, protocol)) }
    }

    /// Locates the first interface that matches a typed protocol.
    ///
    /// The protocol GUID is derived from the [`ProtocolInterface`] type `T`.
    ///
    /// # Safety
    ///
    /// Do not create more than one mutable reference to the returned interface.
    unsafe fn locate_protocol<T>(&self, registration: Option<Registration>) -> Result<&'static mut T, efi::Status>
    where
        T: Sized + ProtocolInterface + 'static,
    {
        let registration = registration.map_or(core::ptr::null_mut(), |r| r.as_ptr());
        // SAFETY: The caller guarantees no aliasing of the returned reference.
        unsafe { protocol_ref(|protocol| self.locate_protocol_unchecked(protocol, registration)) }
    }

    /// Locates the first interface that matches a typed protocol, mapping a null interface to `None`.
    ///
    /// The protocol GUID is derived from the [`ProtocolInterface`] type `T`.
    ///
    /// # Safety
    ///
    /// Do not create more than one mutable reference to the returned interface.
    unsafe fn locate_protocol_maybe_empty<T>(
        &self,
        registration: Option<Registration>,
    ) -> Result<Option<&'static mut T>, efi::Status>
    where
        T: Sized + ProtocolInterface + 'static,
    {
        let registration = registration.map_or(core::ptr::null_mut(), |r| r.as_ptr());
        // SAFETY: The caller guarantees no aliasing of the returned reference.
        unsafe { protocol_ref_maybe_empty(|protocol| self.locate_protocol_unchecked(protocol, registration)) }
    }
}

impl<S: ProtocolPrimitives + ?Sized> ProtocolServices for S {}

/// Memory allocation services shared by
/// [`BootServices`](crate::boot_services::BootServices) and
/// [`MmServices`](crate::mm_services::MmServices).
///
/// Both tables wrap the *same* underlying PI / UEFI pool and page allocators
/// (`AllocatePool`/`FreePool`, `AllocatePages`/`FreePages`), so the safe Rust surface is identical
/// and defined once here — including the `unsafe` contract on the free operations. A `free` takes a
/// caller-supplied pointer/address that the function cannot validate; passing an invalid, dangling,
/// already-freed, or foreign value corrupts the firmware allocator, which is a memory-safety hazard
/// regardless of which table performs the call.
///
/// Each concrete table implements these by calling through its own C function-pointer table
/// ([`StandardMmServices`](crate::mm_services::StandardMmServices)) or by forwarding to its existing
/// inherent methods ([`StandardBootServices`](crate::boot_services::StandardBootServices)).
pub trait MemoryServices {
    /// Allocates `size` bytes of pool memory of the given [`EfiMemoryType`].
    fn allocate_pool(&self, memory_type: EfiMemoryType, size: usize) -> Result<*mut u8, efi::Status>;

    /// Frees pool memory previously allocated by [`MemoryServices::allocate_pool`].
    ///
    /// # Safety
    ///
    /// `buffer` must point to a live allocation returned by [`MemoryServices::allocate_pool`] that
    /// the caller exclusively owns and does not use after this call. Passing an invalid, dangling,
    /// already-freed, or foreign pointer corrupts the allocator.
    unsafe fn free_pool(&self, buffer: *mut u8) -> Result<(), efi::Status>;

    /// Allocates `nb_pages` pages of the given [`EfiMemoryType`] using the given [`AllocType`]
    /// strategy, returning the base address of the allocation.
    fn allocate_pages(
        &self,
        alloc_type: AllocType,
        memory_type: EfiMemoryType,
        nb_pages: usize,
    ) -> Result<usize, efi::Status>;

    /// Frees a page range previously allocated by [`MemoryServices::allocate_pages`].
    ///
    /// # Safety
    ///
    /// `address` and `nb_pages` must exactly match a live allocation returned by
    /// [`MemoryServices::allocate_pages`] that the caller exclusively owns and does not use after
    /// this call. A mismatched, foreign, or already-freed range corrupts the allocator.
    unsafe fn free_pages(&self, address: usize, nb_pages: usize) -> Result<(), efi::Status>;
}

#[cfg(test)]
#[coverage(off)]
mod tests {
    use super::*;
    use crate::{BinaryGuid, boot_services::c_ptr::CPtr};
    use core::ptr;

    #[derive(Debug)]
    struct TestProtocol(u32);

    // SAFETY: provides a unique GUID for unit tests.
    unsafe impl ProtocolInterface for TestProtocol {
        const PROTOCOL_GUID: BinaryGuid =
            BinaryGuid::from_bytes(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    }

    #[derive(Debug)]
    struct TestProtocolEmpty;

    // SAFETY: provides a unique GUID for unit tests; zero-sized marker protocol.
    unsafe impl ProtocolInterface for TestProtocolEmpty {
        const PROTOCOL_GUID: BinaryGuid =
            BinaryGuid::from_bytes(&[16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_install_then_uninstall_recovers_interface() {
        let mut installed_interface: *mut c_void = ptr::null_mut();

        let (handle, key) = install_protocol_interface(Box::new(TestProtocol(42)), |protocol, interface| {
            assert_eq!(TestProtocol::PROTOCOL_GUID, *protocol);
            assert!(!interface.is_null());
            installed_interface = interface;
            Ok(7_usize as efi::Handle)
        })
        .unwrap();
        assert_eq!(7, handle as usize);

        let recovered = uninstall_protocol_interface(key, |protocol, interface| {
            assert_eq!(TestProtocol::PROTOCOL_GUID, *protocol);
            assert_eq!(installed_interface, interface);
            Ok(())
        })
        .unwrap();
        assert_eq!(42, recovered.0);
    }

    #[test]
    fn test_install_propagates_error_and_reclaims() {
        let result = install_protocol_interface(Box::new(TestProtocol(1)), |_protocol, _interface| {
            Err(efi::Status::OUT_OF_RESOURCES)
        });

        assert!(matches!(result, Err(efi::Status::OUT_OF_RESOURCES)));
    }

    #[test]
    fn test_uninstall_propagates_error() {
        let interface = Box::new(TestProtocol(1));
        let key = interface.metadata();

        let result = uninstall_protocol_interface(key, |_protocol, _interface| Err(efi::Status::NOT_FOUND)).map(|_| ());

        assert_eq!(Err(efi::Status::NOT_FOUND), result);
        drop(interface);
    }

    #[test]
    fn test_protocol_ref_returns_typed_reference() {
        let leaked = Box::into_raw(Box::new(TestProtocol(7)));
        // SAFETY: `leaked` is a valid, uniquely-owned pointer; `protocol_ref` yields the sole reference.
        let interface = unsafe { protocol_ref::<TestProtocol>(|_protocol| Ok(leaked as *mut c_void)) }.unwrap();
        assert_eq!(7, interface.0);
        // SAFETY: reclaim and drop the leaked allocation; `interface` is not used afterwards.
        drop(unsafe { Box::from_raw(leaked) });
    }

    #[test]
    fn test_protocol_ref_maybe_empty_maps_null_to_none() {
        // SAFETY: the closure returns null, so no reference is created.
        let interface = unsafe { protocol_ref_maybe_empty::<TestProtocol>(|_protocol| Ok(ptr::null_mut())) }.unwrap();
        assert!(interface.is_none());
    }

    #[test]
    fn test_protocol_ref_zero_sized_marker_yields_reference() {
        // SAFETY: ZST marker; the closure returns null and a dangling reference to a ZST is valid.
        let result = unsafe { protocol_ref::<TestProtocolEmpty>(|_protocol| Ok(ptr::null_mut())) };
        assert!(result.is_ok());
    }

    #[test]
    fn test_protocol_ref_propagates_error() {
        // SAFETY: the closure returns an error, so no reference is created.
        let result = unsafe { protocol_ref::<TestProtocol>(|_protocol| Err(efi::Status::NOT_FOUND)) };
        assert_eq!(Err(efi::Status::NOT_FOUND), result.map(|_| ()));
    }
}
