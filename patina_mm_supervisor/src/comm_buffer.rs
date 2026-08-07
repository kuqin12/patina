//! Communication Buffer Configuration for the MM Supervisor Core
//!
//! Defines the communication buffer layout extracted from the MM Supervisor PassDown HOB
//! and shared across the supervisor for routing user- and supervisor-targeted requests.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

/// Communication buffer configuration extracted from PassDown HOB.
#[derive(Debug, Clone, Copy, Default)]
pub struct CommBufferConfig {
    /// MM Supervisor communication buffer (external interface).
    pub supv_comm_buffer: u64,
    /// MM Supervisor internal communication buffer.
    pub supv_comm_buffer_internal: u64,
    /// Size of supervisor communication buffer.
    pub supv_comm_buffer_size: u64,
    /// MM User communication buffer (external interface).
    pub user_comm_buffer: u64,
    /// MM User internal communication buffer.
    pub user_comm_buffer_internal: u64,
    /// Size of user communication buffer.
    pub user_comm_buffer_size: u64,
    /// `MmCommBufferStatus` mailbox for user-targeted requests.
    pub user_status_buffer: u64,
    /// `MmCommBufferStatus` mailbox for supervisor-targeted requests.
    pub supv_status_buffer: u64,
    /// MM Supervisor to User buffer.
    pub supv_to_user_buffer: u64,
    /// Size of Supervisor to User buffer.
    pub supv_to_user_buffer_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comm_buffer_config_default_is_zeroed() {
        let cfg = CommBufferConfig::default();
        assert_eq!(cfg.supv_comm_buffer, 0);
        assert_eq!(cfg.supv_comm_buffer_internal, 0);
        assert_eq!(cfg.supv_comm_buffer_size, 0);
        assert_eq!(cfg.user_comm_buffer, 0);
        assert_eq!(cfg.user_comm_buffer_internal, 0);
        assert_eq!(cfg.user_comm_buffer_size, 0);
        assert_eq!(cfg.user_status_buffer, 0);
        assert_eq!(cfg.supv_status_buffer, 0);
        assert_eq!(cfg.supv_to_user_buffer, 0);
        assert_eq!(cfg.supv_to_user_buffer_size, 0);
    }

    #[test]
    fn test_comm_buffer_config_is_copy() {
        let cfg = CommBufferConfig { supv_comm_buffer: 0x1000, user_comm_buffer: 0x2000, ..Default::default() };
        let copied = cfg; // relies on `Copy`
        assert_eq!(copied.supv_comm_buffer, 0x1000);
        assert_eq!(copied.user_comm_buffer, 0x2000);
        // `cfg` is still usable after the copy.
        assert_eq!(cfg.supv_comm_buffer, 0x1000);
    }
}
