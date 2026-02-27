//! Protocol/Handle Database
//!
//! This module provides a simplified protocol database for the MM User Core.
//! It tracks installed protocols for depex evaluation and driver service use.
//!
//! In the DXE Core, the protocol database is a full handle-protocol mapping
//! (handles can have multiple protocols, protocols can be on multiple handles).
//! The MM User Core simplifies this to a flat set of installed protocol GUIDs
//! since MM drivers primarily need:
//! - `MmInstallProtocolInterface`: Register that a protocol is available
//! - `MmLocateProtocol`: Check if a protocol is available (for depex evaluation)
//! - `registered_protocols()`: Get the list of all installed protocols (for depex eval)
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

use alloc::vec::Vec;

use r_efi::efi;
use spin::Mutex;

// =============================================================================
// Protocol Database
// =============================================================================

/// A simplified protocol/handle database for the MM User Core.
///
/// Tracks installed protocol GUIDs for depex evaluation and protocol location.
pub struct ProtocolDatabase {
    /// Internal state protected by a mutex.
    inner: Mutex<ProtocolDatabaseInner>,
}

struct ProtocolDatabaseInner {
    /// The set of installed protocol GUIDs.
    protocols: Vec<efi::Guid>,
}

impl ProtocolDatabase {
    /// Creates a new empty `ProtocolDatabase`.
    pub const fn new() -> Self {
        Self {
            inner: Mutex::new(ProtocolDatabaseInner {
                protocols: Vec::new(),
            }),
        }
    }

    /// Install a protocol interface.
    ///
    /// Registers the given protocol GUID as available. Duplicate GUIDs are allowed
    /// (multiple instances of the same protocol can be installed on different handles).
    pub fn install_protocol(&self, protocol_guid: &efi::Guid) -> Result<(), efi::Status> {
        let mut inner = self.inner.lock();
        inner.protocols.push(*protocol_guid);
        log::debug!("Installed protocol: {:?}", protocol_guid);
        Ok(())
    }

    /// Uninstall a protocol interface.
    ///
    /// Removes the first occurrence of the given protocol GUID.
    pub fn uninstall_protocol(&self, protocol_guid: &efi::Guid) -> Result<(), efi::Status> {
        let mut inner = self.inner.lock();
        if let Some(pos) = inner.protocols.iter().position(|g| g == protocol_guid) {
            inner.protocols.remove(pos);
            log::debug!("Uninstalled protocol: {:?}", protocol_guid);
            Ok(())
        } else {
            log::warn!("Protocol {:?} not found for uninstall.", protocol_guid);
            Err(efi::Status::NOT_FOUND)
        }
    }

    /// Check if a protocol is installed.
    pub fn is_protocol_installed(&self, protocol_guid: &efi::Guid) -> bool {
        let inner = self.inner.lock();
        inner.protocols.contains(protocol_guid)
    }

    /// Get the list of all installed protocol GUIDs.
    ///
    /// This is used by the depex evaluator to determine which dependencies are satisfied.
    pub fn registered_protocols(&self) -> Vec<efi::Guid> {
        let inner = self.inner.lock();
        inner.protocols.clone()
    }
}
