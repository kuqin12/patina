# patina_mm_policy

MM Supervisor secure policy library for Rust UEFI environments.

This crate provides a comprehensive policy management library for the MM Supervisor,
including policy data structures, access validation (policy gate), and helper utilities.

## Features

### Policy Gate
Initialize with a policy buffer pointer, then query whether operations are allowed:
- `is_io_allowed()` - Check I/O port access
- `is_msr_allowed()` - Check MSR access  
- `is_instruction_allowed()` - Check privileged instruction execution
- `is_save_state_read_allowed()` - Check save state read access

### Helper Functions
- `dump_policy()` - Print policy contents for debugging
- `compare_policies()` - Compare two policies (order-independent)
- `populate_memory_policy_from_page_table()` - Walk page tables to generate memory policy

## Usage

```rust,ignore
use patina_mm_policy::{PolicyGate, AccessType, IoWidth};

// Initialize the policy gate with a policy buffer
let gate = unsafe { PolicyGate::new(policy_ptr) }?;

// Check if I/O access is allowed
let result = gate.is_io_allowed(0x3F8, IoWidth::Byte, AccessType::Read);
if result.is_ok() {
    // Access allowed
}

// Dump policy for debugging
gate.dump_policy();
```

## License

Copyright (c) Microsoft Corporation.

SPDX-License-Identifier: Apache-2.0
