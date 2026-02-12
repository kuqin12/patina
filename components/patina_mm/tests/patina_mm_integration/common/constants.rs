//! Test constants for Patina MM integration tests
//!
//! ## License
//!
//! Copyright (C) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0

use patina::base::SIZE_4KB;

/// Standard test buffer size
pub const TEST_BUFFER_SIZE: usize = SIZE_4KB;

/// MM Supervisor constants and definitions for testing.
///
/// Protocol constants (SIGNATURE, REVISION, request types) are re-exported from the
/// shared [`patina_mm::protocol::mm_supervisor_request`] module. Test-specific values
/// (VERSION, PATCH_LEVEL, etc.) are defined here as mock data.
pub mod mm_supv {
    // Re-export shared protocol constants
    pub use patina_mm::protocol::mm_supervisor_request::{
        SIGNATURE, REVISION,
        requests, responses,
    };

    /// Request signature as a DWORD (same as shared SIGNATURE, kept for test compatibility)
    pub const REQUEST_SIGNATURE: u32 = SIGNATURE;

    /// Mock supervisor version for testing
    pub const VERSION: u32 = 0x00130008;

    /// Mock supervisor patch level for testing
    pub const PATCH_LEVEL: u32 = 0x00010001;

    /// Mock maximum request level supported
    pub const MAX_REQUEST_LEVEL: u64 = 0x0000000000000004; // COMM_UPDATE
}

/// Test GUIDs for different handlers
///
/// Provides predefined GUIDs used throughout the patina_mm test framework for registering
/// and identifying different types of test handlers.
pub mod test_guids {
    use r_efi::efi;

    /// Echo handler GUID for testing
    pub const ECHO_HANDLER: efi::Guid =
        efi::Guid::from_fields(0x12345678, 0x1234, 0x5678, 0x12, 0x34, &[0x56, 0x78, 0x90, 0xab, 0xcd, 0xef]);

    /// Version handler GUID for testing
    /// Note: Not used now but the GUID is reserved for future usage
    #[allow(dead_code)]
    pub const VERSION_HANDLER: efi::Guid =
        efi::Guid::from_fields(0x87654321, 0x4321, 0x8765, 0x43, 0x21, &[0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54]);

    /// MM Supervisor GUID for supervisor protocol testing
    pub const MM_SUPERVISOR: efi::Guid =
        efi::Guid::from_fields(0x8c633b23, 0x1260, 0x4ea6, 0x83, 0x0F, &[0x7d, 0xdc, 0x97, 0x38, 0x21, 0x11]);
}

// Convenience re-exports for common usage
pub use test_guids::ECHO_HANDLER as TEST_COMMUNICATION_GUID;
