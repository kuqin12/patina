//! MM Supervisor Secure Policy Library
//!
//! This crate provides a comprehensive policy management library for the MM Supervisor,
//! including policy data structures, access validation (policy gate), and helper utilities.
//!
//! ## Overview
//!
//! The MM Supervisor uses security policies to control what resources user-mode MM drivers
//! can access. This crate provides:
//!
//! - **Data Structures**: Rust definitions matching the C structures in `SmmSecurePolicy.h`
//! - **Policy Gate**: Runtime access validation for I/O, MSR, instruction, and save state
//! - **Helpers**: Dump, compare, and page table walking utilities
//!
//! ## Example
//!
//! ```rust,ignore
//! use patina_mm_policy::{PolicyGate, AccessType, IoWidth};
//!
//! // Initialize the policy gate with a policy buffer
//! let gate = unsafe { PolicyGate::new(policy_ptr) }?;
//!
//! // Check if I/O access is allowed
//! if gate.is_io_allowed(0x3F8, IoWidth::Byte, AccessType::Read).is_ok() {
//!     // Perform the I/O operation
//! }
//! ```
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!

#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]
#![allow(dead_code)]

mod types;
mod gate;
mod helpers;

pub use types::*;
pub use gate::*;
pub use helpers::*;
