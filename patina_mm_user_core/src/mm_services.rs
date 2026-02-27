//! MM System Table (MMST) Construction — User Core Implementation
//!
//! This module builds the concrete `EfiMmSystemTable` instance that is passed
//! to dispatched MM drivers. The *type definitions* (`EfiMmSystemTable`,
//! `MmServices` trait, `StandardMmServices`, etc.) live in the Patina SDK at
//! [`patina::mm_services`] — this module only provides the `extern "efiapi"`
//! thunk functions, the global databases they route to, and the one-time
//! `init_mm_system_table()` entry point.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

extern crate alloc;

use alloc::{boxed::Box, vec::Vec};
use core::ffi::c_void;

use r_efi::efi;
use spin::Once;

use patina::mm_services::{
    EfiMmSystemTable, MmCpuIoAccess, MmCpuIoProtocol, MmiHandlerEntryPoint,
    MM_MMST_SIGNATURE, MM_SYSTEM_TABLE_REVISION,
};
use patina_internal_mm_alloc::PageAllocatorBackend;

// =============================================================================
// Global system table pointer
// =============================================================================

/// Wrapper around a raw pointer so it can live in a `static Once<>`.
struct SendSyncPtr(*mut EfiMmSystemTable);

// SAFETY: The pointer is only written once (in `init_mm_system_table`) and read
// immutably afterwards.  All mutable state behind it is protected by locks.
unsafe impl Send for SendSyncPtr {}
unsafe impl Sync for SendSyncPtr {}

/// The heap-allocated MM System Table.  Initialized once in [`init_mm_system_table`].
static MM_SYSTEM_TABLE: Once<SendSyncPtr> = Once::new();

/// Initialize the MM System Table.
///
/// Allocates the table on the heap and populates it with service function pointers
/// that route to the user core's databases. Must be called once during
/// `StartUserCore`, **after** the heap is available.
///
/// Returns a raw pointer to the table suitable for passing to driver entry points.
pub fn init_mm_system_table() -> *mut EfiMmSystemTable {
    MM_SYSTEM_TABLE.call_once(|| {
        let table = EfiMmSystemTable {
            hdr: efi::TableHeader {
                signature: MM_MMST_SIGNATURE,
                revision: MM_SYSTEM_TABLE_REVISION,
                header_size: core::mem::size_of::<EfiMmSystemTable>() as u32,
                crc32: 0,
                reserved: 0,
            },
            mm_firmware_vendor: core::ptr::null_mut(),
            mm_firmware_revision: 0,

            mm_install_configuration_table: mm_install_configuration_table_impl,

            mm_io: MmCpuIoProtocol {
                mem: MmCpuIoAccess {
                    read: mm_io_not_available,
                    write: mm_io_not_available,
                },
                io: MmCpuIoAccess {
                    read: mm_io_not_available,
                    write: mm_io_not_available,
                },
            },

            mm_allocate_pool: mm_allocate_pool_impl,
            mm_free_pool: mm_free_pool_impl,
            mm_allocate_pages: mm_allocate_pages_impl,
            mm_free_pages: mm_free_pages_impl,

            mm_startup_this_ap: mm_startup_this_ap_not_available,

            currently_executing_cpu: 0,
            number_of_cpus: 0,
            cpu_save_state_size: core::ptr::null_mut(),
            cpu_save_state: core::ptr::null_mut(),

            number_of_table_entries: 0,
            mm_configuration_table: core::ptr::null_mut(),

            mm_install_protocol_interface: mm_install_protocol_interface_impl,
            mm_uninstall_protocol_interface: mm_uninstall_protocol_interface_impl,
            mm_handle_protocol: mm_handle_protocol_impl,
            mm_register_protocol_notify: mm_register_protocol_notify_impl,
            mm_locate_handle: mm_locate_handle_impl,
            mm_locate_protocol: mm_locate_protocol_impl,

            mmi_manage: mmi_manage_impl,
            mmi_handler_register: mmi_handler_register_impl,
            mmi_handler_unregister: mmi_handler_unregister_impl,
        };

        let ptr = Box::into_raw(Box::new(table));
        log::info!("MM System Table allocated at {:p}", ptr);
        SendSyncPtr(ptr)
    }).0
}

/// Returns the MM System Table pointer, or null if not yet initialized.
pub fn get_mm_system_table() -> *mut EfiMmSystemTable {
    MM_SYSTEM_TABLE.get().map(|p| p.0).unwrap_or(core::ptr::null_mut())
}

// =============================================================================
// Service implementations — I/O (stubs)
// =============================================================================

unsafe extern "efiapi" fn mm_io_not_available(
    _this: *const MmCpuIoAccess,
    _width: usize,
    _address: u64,
    _count: usize,
    _buffer: *mut c_void,
) -> efi::Status {
    efi::Status::UNSUPPORTED
}

// =============================================================================
// Service implementations — Configuration Table
// =============================================================================

unsafe extern "efiapi" fn mm_install_configuration_table_impl(
    _system_table: *const EfiMmSystemTable,
    guid: *const efi::Guid,
    table: *mut c_void,
    _table_size: usize,
) -> efi::Status {
    if guid.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }

    let guid = unsafe { &*guid };
    crate::config_table::GLOBAL_CONFIG_TABLE_DB.install_configuration_table(guid, table)
}

// =============================================================================
// Service implementations — Memory services (syscall-backed)
// =============================================================================

unsafe extern "efiapi" fn mm_allocate_pool_impl(
    _pool_type: efi::MemoryType,
    size: usize,
    buffer: *mut *mut c_void,
) -> efi::Status {
    if buffer.is_null() || size == 0 {
        return efi::Status::INVALID_PARAMETER;
    }

    let layout = match core::alloc::Layout::from_size_align(size, 8) {
        Ok(l) => l,
        Err(_) => return efi::Status::INVALID_PARAMETER,
    };

    let ptr = unsafe { alloc::alloc::alloc(layout) };
    if ptr.is_null() {
        return efi::Status::OUT_OF_RESOURCES;
    }

    unsafe { *buffer = ptr as *mut c_void };
    efi::Status::SUCCESS
}

unsafe extern "efiapi" fn mm_free_pool_impl(
    buffer: *mut c_void,
) -> efi::Status {
    if buffer.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }

    let layout = unsafe { core::alloc::Layout::from_size_align_unchecked(1, 1) };
    unsafe { alloc::alloc::dealloc(buffer as *mut u8, layout) };
    efi::Status::SUCCESS
}

unsafe extern "efiapi" fn mm_allocate_pages_impl(
    _alloc_type: efi::AllocateType,
    _memory_type: efi::MemoryType,
    pages: usize,
    memory: *mut efi::PhysicalAddress,
) -> efi::Status {
    if memory.is_null() || pages == 0 {
        return efi::Status::INVALID_PARAMETER;
    }

    match crate::mm_mem::SYSCALL_PAGE_ALLOCATOR.allocate_pages(pages) {
        Ok(addr) => {
            unsafe { *memory = addr };
            efi::Status::SUCCESS
        }
        Err(_) => efi::Status::OUT_OF_RESOURCES,
    }
}

unsafe extern "efiapi" fn mm_free_pages_impl(
    memory: efi::PhysicalAddress,
    pages: usize,
) -> efi::Status {
    if memory == 0 || pages == 0 {
        return efi::Status::INVALID_PARAMETER;
    }

    match crate::mm_mem::SYSCALL_PAGE_ALLOCATOR.free_pages(memory, pages) {
        Ok(()) => efi::Status::SUCCESS,
        Err(_) => efi::Status::INVALID_PARAMETER,
    }
}

// =============================================================================
// Service implementations — MP service (stub)
// =============================================================================

unsafe extern "efiapi" fn mm_startup_this_ap_not_available(
    _procedure: usize,
    _cpu_number: usize,
    _proc_arguments: *mut c_void,
) -> efi::Status {
    efi::Status::UNSUPPORTED
}

// =============================================================================
// Service implementations — Protocol services
// =============================================================================

unsafe extern "efiapi" fn mm_install_protocol_interface_impl(
    handle: *mut efi::Handle,
    protocol: *mut efi::Guid,
    _interface_type: efi::InterfaceType,
    interface: *mut c_void,
) -> efi::Status {
    if handle.is_null() || protocol.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }

    let guid = unsafe { &*protocol };
    let caller_handle = unsafe { *handle };

    match GLOBAL_PROTOCOL_DB.install_protocol(caller_handle, guid, interface) {
        Ok((new_handle, pending_notifies)) => {
            unsafe { *handle = new_handle };
            // Fire notifications outside the DB lock.
            for notify in pending_notifies {
                unsafe {
                    (notify.function)(
                        &notify.guid as *const efi::Guid,
                        notify.interface,
                        notify.handle,
                    );
                }
            }
            efi::Status::SUCCESS
        }
        Err(status) => status,
    }
}

unsafe extern "efiapi" fn mm_uninstall_protocol_interface_impl(
    handle: efi::Handle,
    protocol: *mut efi::Guid,
    interface: *mut c_void,
) -> efi::Status {
    if handle.is_null() || protocol.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }

    let guid = unsafe { &*protocol };

    match GLOBAL_PROTOCOL_DB.uninstall_protocol(handle, guid, interface) {
        Ok(()) => efi::Status::SUCCESS,
        Err(status) => status,
    }
}

unsafe extern "efiapi" fn mm_handle_protocol_impl(
    handle: efi::Handle,
    protocol: *mut efi::Guid,
    interface: *mut *mut c_void,
) -> efi::Status {
    if protocol.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }

    if interface.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }
    // C reference: *Interface = NULL before lookup.
    unsafe { *interface = core::ptr::null_mut() };

    if handle.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }

    let guid = unsafe { &*protocol };

    match GLOBAL_PROTOCOL_DB.handle_protocol(handle, guid) {
        Some(iface) => {
            unsafe { *interface = iface };
            efi::Status::SUCCESS
        }
        None => efi::Status::UNSUPPORTED,
    }
}

unsafe extern "efiapi" fn mm_register_protocol_notify_impl(
    protocol: *const efi::Guid,
    function: usize,
    registration: *mut *mut c_void,
) -> efi::Status {
    if protocol.is_null() || registration.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }

    let guid = unsafe { &*protocol };

    if function == 0 {
        // Function is NULL → unregister the notification identified by *Registration.
        let reg = unsafe { *registration };
        match GLOBAL_PROTOCOL_DB.unregister_protocol_notify(guid, reg) {
            Ok(()) => efi::Status::SUCCESS,
            Err(status) => status,
        }
    } else {
        // Register a new notification.
        // SAFETY: function is an `EFI_MM_NOTIFY_FN` function pointer passed as usize.
        let notify_fn: MmNotifyFn = unsafe { core::mem::transmute(function) };
        let token = GLOBAL_PROTOCOL_DB.register_protocol_notify(guid, notify_fn);
        unsafe { *registration = token };
        efi::Status::SUCCESS
    }
}

unsafe extern "efiapi" fn mm_locate_handle_impl(
    search_type: efi::LocateSearchType,
    protocol: *mut efi::Guid,
    _search_key: *mut c_void,
    buffer_size: *mut usize,
    buffer: *mut efi::Handle,
) -> efi::Status {
    if buffer_size.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }

    let handles = match search_type {
        efi::ALL_HANDLES => GLOBAL_PROTOCOL_DB.all_handles(),
        efi::BY_PROTOCOL => {
            if protocol.is_null() {
                return efi::Status::INVALID_PARAMETER;
            }
            let guid = unsafe { &*protocol };
            GLOBAL_PROTOCOL_DB.locate_handle_by_protocol(guid)
        }
        _ => {
            log::warn!("MmLocateHandle: search type {} not yet supported", search_type);
            return efi::Status::UNSUPPORTED;
        }
    };

    if handles.is_empty() {
        return efi::Status::NOT_FOUND;
    }

    let required_size = handles.len() * core::mem::size_of::<efi::Handle>();
    let caller_size = unsafe { *buffer_size };
    unsafe { *buffer_size = required_size };

    if caller_size < required_size {
        return efi::Status::BUFFER_TOO_SMALL;
    }

    if buffer.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }

    unsafe {
        core::ptr::copy_nonoverlapping(handles.as_ptr(), buffer, handles.len());
    }
    efi::Status::SUCCESS
}

unsafe extern "efiapi" fn mm_locate_protocol_impl(
    protocol: *mut efi::Guid,
    _registration: *mut c_void,
    interface: *mut *mut c_void,
) -> efi::Status {
    if protocol.is_null() || interface.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }

    let guid = unsafe { &*protocol };

    match GLOBAL_PROTOCOL_DB.locate_protocol(guid) {
        Some(iface) => {
            unsafe { *interface = iface };
            efi::Status::SUCCESS
        }
        None => efi::Status::NOT_FOUND,
    }
}

// =============================================================================
// Service implementations — MMI management
// =============================================================================

unsafe extern "efiapi" fn mmi_manage_impl(
    handler_type: *const efi::Guid,
    context: *const c_void,
    comm_buffer: *mut c_void,
    comm_buffer_size: *mut usize,
) -> efi::Status {
    let guid = if handler_type.is_null() {
        None
    } else {
        Some(unsafe { &*handler_type })
    };

    GLOBAL_MMI_DB.mmi_manage(guid, context, comm_buffer, comm_buffer_size)
}

unsafe extern "efiapi" fn mmi_handler_register_impl(
    handler: MmiHandlerEntryPoint,
    handler_type: *const efi::Guid,
    dispatch_handle: *mut efi::Handle,
) -> efi::Status {
    if dispatch_handle.is_null() {
        return efi::Status::INVALID_PARAMETER;
    }

    let guid = if handler_type.is_null() {
        None
    } else {
        Some(unsafe { &*handler_type })
    };

    match GLOBAL_MMI_DB.mmi_handler_register(handler, guid) {
        Ok(handle) => {
            unsafe { *dispatch_handle = handle };
            efi::Status::SUCCESS
        }
        Err(status) => status,
    }
}

unsafe extern "efiapi" fn mmi_handler_unregister_impl(
    dispatch_handle: efi::Handle,
) -> efi::Status {
    match GLOBAL_MMI_DB.mmi_handler_unregister(dispatch_handle) {
        Ok(()) => efi::Status::SUCCESS,
        Err(status) => status,
    }
}

// =============================================================================
// Global databases accessed by service thunks
// =============================================================================

use crate::mmi::MmiDatabase;

/// Global MMI handler database used by the system table services.
pub static GLOBAL_MMI_DB: MmiDatabase = MmiDatabase::new();

// =============================================================================
// Protocol database — handle-aware, with notify support
// =============================================================================

/// `EFI_MM_NOTIFY_FN` — callback invoked when a protocol is installed.
///
/// ```c
/// typedef EFI_STATUS (EFIAPI *EFI_MM_NOTIFY_FN)(
///   IN CONST EFI_GUID  *Protocol,
///   IN VOID            *Interface,
///   IN EFI_HANDLE      Handle
/// );
/// ```
type MmNotifyFn = unsafe extern "efiapi" fn(
    *const efi::Guid,
    *mut c_void,
    efi::Handle,
) -> efi::Status;

/// Per-handle state: all protocol interfaces installed on one handle.
struct MmHandle {
    protocols: Vec<(efi::Guid, *mut c_void)>,
}

/// A registered protocol notification.
struct ProtocolNotifyEntry {
    guid: efi::Guid,
    function: MmNotifyFn,
    /// Unique token returned via `*Registration`.
    token: usize,
}

/// Info collected under the lock, fired after the lock is released.
pub struct PendingNotify {
    pub function: MmNotifyFn,
    pub guid: efi::Guid,
    pub interface: *mut c_void,
    pub handle: efi::Handle,
}

struct MmProtocolDatabaseInner {
    /// All handles: (opaque id, per-handle data).
    handles: Vec<(usize, MmHandle)>,
    /// Registered protocol notifications.
    notifications: Vec<ProtocolNotifyEntry>,
    /// Next monotonic id for handle allocation (starts at 1 to avoid null).
    next_handle_id: usize,
    /// Next monotonic id for registration tokens (starts at 1 to avoid null).
    next_registration_id: usize,
}

/// Handle-aware protocol database with notification support.
///
/// Mirrors the C `StandaloneMmCore` handle/protocol infrastructure
/// (`IHANDLE`, `PROTOCOL_ENTRY`, `PROTOCOL_INTERFACE`, `PROTOCOL_NOTIFY`).
pub struct MmProtocolDatabase {
    inner: spin::Mutex<MmProtocolDatabaseInner>,
}

// SAFETY: All mutable state is behind a spin::Mutex.
unsafe impl Send for MmProtocolDatabase {}
unsafe impl Sync for MmProtocolDatabase {}

impl MmProtocolDatabase {
    pub const fn new() -> Self {
        Self {
            inner: spin::Mutex::new(MmProtocolDatabaseInner {
                handles: Vec::new(),
                notifications: Vec::new(),
                next_handle_id: 1,
                next_registration_id: 1,
            }),
        }
    }

    // -----------------------------------------------------------------
    // Install / Uninstall
    // -----------------------------------------------------------------

    /// Install a protocol interface onto a handle.
    ///
    /// If `handle` is null a new handle is allocated.  Returns the
    /// (possibly new) handle and any pending notify callbacks that must
    /// be invoked **after** the caller has released the lock.
    pub fn install_protocol(
        &self,
        handle: efi::Handle,
        guid: &efi::Guid,
        interface: *mut c_void,
    ) -> Result<(efi::Handle, Vec<PendingNotify>), efi::Status> {
        let mut inner = self.inner.lock();

        let handle_id = if handle.is_null() {
            // Allocate a new handle.
            let id = inner.next_handle_id;
            inner.next_handle_id += 1;
            inner.handles.push((id, MmHandle { protocols: Vec::new() }));
            id
        } else {
            let id = handle as usize;
            // Validate the handle exists.
            if !inner.handles.iter().any(|(h, _)| *h == id) {
                return Err(efi::Status::INVALID_PARAMETER);
            }
            // Reject duplicate: same protocol already on this handle.
            let mm_handle = &inner.handles.iter().find(|(h, _)| *h == id).unwrap().1;
            if mm_handle.protocols.iter().any(|(g, _)| g == guid) {
                return Err(efi::Status::INVALID_PARAMETER);
            }
            id
        };

        // Add the protocol interface to the handle.
        let mm_handle = &mut inner.handles.iter_mut().find(|(h, _)| *h == handle_id).unwrap().1;
        mm_handle.protocols.push((*guid, interface));

        // Collect pending notifications.
        let actual_handle = handle_id as efi::Handle;
        let notifies: Vec<PendingNotify> = inner
            .notifications
            .iter()
            .filter(|n| n.guid == *guid)
            .map(|n| PendingNotify {
                function: n.function,
                guid: *guid,
                interface,
                handle: actual_handle,
            })
            .collect();

        log::debug!("MmInstallProtocolInterface: {:?} on handle {:p}", guid, actual_handle);
        Ok((actual_handle, notifies))
    }

    /// Uninstall a protocol interface from a handle.
    ///
    /// If the handle has no remaining protocols it is removed from the
    /// database (matching the C `MmUninstallProtocolInterface` behaviour).
    pub fn uninstall_protocol(
        &self,
        handle: efi::Handle,
        guid: &efi::Guid,
        interface: *mut c_void,
    ) -> Result<(), efi::Status> {
        let mut inner = self.inner.lock();
        let id = handle as usize;

        let mm_handle = match inner.handles.iter_mut().find(|(h, _)| *h == id) {
            Some((_, h)) => h,
            None => return Err(efi::Status::INVALID_PARAMETER),
        };

        if let Some(pos) = mm_handle.protocols.iter().position(|(g, i)| g == guid && *i == interface) {
            mm_handle.protocols.remove(pos);
        } else {
            return Err(efi::Status::NOT_FOUND);
        }

        // Remove the handle entirely when it has no more protocols.
        if inner.handles.iter().find(|(h, _)| *h == id).unwrap().1.protocols.is_empty() {
            inner.handles.retain(|(h, _)| *h != id);
        }

        Ok(())
    }

    // -----------------------------------------------------------------
    // Lookup
    // -----------------------------------------------------------------

    /// Look up a specific protocol on a specific handle (`HandleProtocol`).
    pub fn handle_protocol(
        &self,
        handle: efi::Handle,
        guid: &efi::Guid,
    ) -> Option<*mut c_void> {
        let inner = self.inner.lock();
        let id = handle as usize;
        let mm_handle = &inner.handles.iter().find(|(h, _)| *h == id)?.1;
        mm_handle.protocols.iter().find(|(g, _)| g == guid).map(|(_, i)| *i)
    }

    /// Locate the first installed interface for a GUID across all handles.
    pub fn locate_protocol(&self, guid: &efi::Guid) -> Option<*mut c_void> {
        let inner = self.inner.lock();
        for (_, mm_handle) in &inner.handles {
            if let Some((_, iface)) = mm_handle.protocols.iter().find(|(g, _)| g == guid) {
                return Some(*iface);
            }
        }
        None
    }

    /// Return all handles that support a given protocol.
    pub fn locate_handle_by_protocol(&self, guid: &efi::Guid) -> Vec<efi::Handle> {
        let inner = self.inner.lock();
        inner
            .handles
            .iter()
            .filter(|(_, mm_handle)| mm_handle.protocols.iter().any(|(g, _)| g == guid))
            .map(|(id, _)| *id as efi::Handle)
            .collect()
    }

    /// Return all handles in the database.
    pub fn all_handles(&self) -> Vec<efi::Handle> {
        let inner = self.inner.lock();
        inner.handles.iter().map(|(id, _)| *id as efi::Handle).collect()
    }

    // -----------------------------------------------------------------
    // Notify
    // -----------------------------------------------------------------

    /// Register a notification callback for a protocol GUID.
    ///
    /// If an identical `(GUID, function)` pair is already registered the
    /// existing token is returned (matching the C implementation).
    pub fn register_protocol_notify(
        &self,
        guid: &efi::Guid,
        function: MmNotifyFn,
    ) -> *mut c_void {
        let mut inner = self.inner.lock();
        let fn_addr = function as usize;

        // De-duplicate: same GUID + same function pointer.
        if let Some(existing) = inner
            .notifications
            .iter()
            .find(|n| n.guid == *guid && (n.function as usize) == fn_addr)
        {
            return existing.token as *mut c_void;
        }

        let token = inner.next_registration_id;
        inner.next_registration_id += 1;
        inner.notifications.push(ProtocolNotifyEntry {
            guid: *guid,
            function,
            token,
        });

        token as *mut c_void
    }

    /// Unregister a notification by its registration token.
    pub fn unregister_protocol_notify(
        &self,
        guid: &efi::Guid,
        registration: *mut c_void,
    ) -> Result<(), efi::Status> {
        let mut inner = self.inner.lock();
        let token = registration as usize;

        if let Some(pos) = inner
            .notifications
            .iter()
            .position(|n| n.guid == *guid && n.token == token)
        {
            inner.notifications.remove(pos);
            Ok(())
        } else {
            Err(efi::Status::NOT_FOUND)
        }
    }

    // -----------------------------------------------------------------
    // Depex helpers (backward-compatible public API)
    // -----------------------------------------------------------------

    /// Check if a protocol GUID is installed on any handle.
    pub fn is_protocol_installed(&self, guid: &efi::Guid) -> bool {
        let inner = self.inner.lock();
        inner
            .handles
            .iter()
            .any(|(_, mm_handle)| mm_handle.protocols.iter().any(|(g, _)| g == guid))
    }

    /// Return all unique installed protocol GUIDs.
    pub fn registered_protocols(&self) -> Vec<efi::Guid> {
        let inner = self.inner.lock();
        let mut guids = Vec::new();
        for (_, mm_handle) in &inner.handles {
            for (g, _) in &mm_handle.protocols {
                if !guids.contains(g) {
                    guids.push(*g);
                }
            }
        }
        guids
    }
}

/// Global protocol database used by the system table services.
pub static GLOBAL_PROTOCOL_DB: MmProtocolDatabase = MmProtocolDatabase::new();

// =============================================================================
// Helper: update CPU context from MmEntryContext
// =============================================================================

/// Update the system table's CPU information from a new `EfiMmEntryContext`.
///
/// Called at the start of each `UserRequest` handling to reflect the current
/// processor state.
pub fn update_cpu_context(
    currently_executing_cpu: usize,
    number_of_cpus: usize,
) {
    let ptr = get_mm_system_table();
    if ptr.is_null() {
        return;
    }
    // SAFETY: The table is heap-allocated and we are the only writer of these fields.
    unsafe {
        (*ptr).currently_executing_cpu = currently_executing_cpu;
        (*ptr).number_of_cpus = number_of_cpus;
    }
}
