# Patina MM Supervisor Core

A pure Rust implementation of the MM Supervisor Core for standalone MM mode environments.

## Overview

This crate provides the core functionality for the MM (Management Mode) Supervisor in a standalone MM environment. It is designed to run on x64 systems where:

- Page tables are already set up by the pre-MM phase
- All images are loaded and ready to execute
- The BSP (Bootstrap Processor) orchestrates incoming requests
- APs (Application Processors) wait in a holding pen, checking a mailbox for work

## Memory Model

**This is a core component that does not use heap allocation.** All data structures use fixed-size arrays with compile-time constants provided via const generics in the `PlatformInfo` trait:

- `MAX_CPU_COUNT` - Maximum number of CPUs supported
- `MAX_HANDLERS` - Maximum number of request handlers

This allows the entire supervisor to be instantiated as a `static` with no runtime allocation.

## Building a PE/COFF Binary

### Prerequisites

1. Install the Rust UEFI target:
   ```bash
   rustup target add x86_64-unknown-uefi
   ```

2. Ensure you have the nightly toolchain (required for `#![feature(...)]`):
   ```bash
   rustup override set nightly
   ```

### Build Command

Build the example MM Supervisor binary:

```bash
cargo build --release --target x86_64-unknown-uefi --bin example_mm_supervisor
```

The output PE/COFF binary will be at:
```
target/x86_64-unknown-uefi/release/example_mm_supervisor.efi
```

### Entry Point

The MM Supervisor exports `MmSupervisorMain` as its entry point, matching the EDK2 convention:

```rust
#[unsafe(export_name = "MmSupervisorMain")]
pub extern "efiapi" fn mm_supervisor_main(hob_list: *const c_void) -> ! {
    SUPERVISOR.entry_point(hob_list)
}
```

The MM IPL (Initial Program Loader) calls this entry point on **all processors** after:
1. Loading the supervisor image into MMRAM
2. Setting up page tables
3. Constructing the HOB list with MMRAM ranges

## Architecture

### Entry Point Model

The entry point is executed on all cores simultaneously:

1. **BSP (Bootstrap Processor)**: 
   - First CPU to arrive (determined by atomic counter)
   - Performs one-time initialization
   - Sets up the request handling infrastructure
   - Enters the main request serving loop

2. **APs (Application Processors)**:
   - All other CPUs
   - Wait for BSP initialization to complete
   - Enter a holding pen and poll mailboxes for commands

### Mailbox System

The mailbox system provides inter-processor communication:

- Each AP has a dedicated mailbox (cache-line aligned to avoid false sharing)
- BSP sends commands to APs via mailboxes
- APs respond with results through the same mailbox
- Supports synchronization primitives for coordinated operations

## Usage

### Basic Platform Implementation

```rust
#![no_std]
#![no_main]

use core::{ffi::c_void, panic::PanicInfo};
use patina_mm_supervisor_core::*;

struct MyPlatform;

impl CpuInfo for MyPlatform {
    fn ap_poll_timeout_us() -> u64 { 1000 }
}

impl PlatformInfo for MyPlatform {
    type CpuInfo = Self;
    const MAX_CPU_COUNT: usize = 8;
    const MAX_HANDLERS: usize = 32;
}

// Static instance - no heap allocation required
static SUPERVISOR: MmSupervisorCore<MyPlatform> = MmSupervisorCore::new();

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop { core::hint::spin_loop(); }
}

#[unsafe(export_name = "MmSupervisorMain")]
pub extern "efiapi" fn mm_supervisor_main(hob_list: *const c_void) -> ! {
    SUPERVISOR.entry_point(hob_list)
}
```

### Registering Request Handlers

Handlers must be defined as static references:

```rust
use patina_mm_supervisor_core::*;

struct MyHandler;

impl RequestHandler for MyHandler {
    fn guid(&self) -> r_efi::efi::Guid {
        // Your handler's GUID
        r_efi::efi::Guid::from_fields(0x12345678, 0x1234, 0x5678, 0x12, 0x34, &[0; 6])
    }

    fn handle(&self, context: &mut RequestContext) -> RequestResult {
        // Handle the request
        RequestResult::Success
    }

    fn name(&self) -> &'static str {
        "MyHandler"
    }
}

static MY_HANDLER: MyHandler = MyHandler;

// Register before calling entry_point, or during BSP initialization
SUPERVISOR.register_handler(&MY_HANDLER);
```

### Integration with MM IPL

The MM IPL (from EDK2/MmSupervisorPkg) loads this binary and calls the entry point. The HOB list passed contains:

- `gEfiMmPeiMmramMemoryReserveGuid` - MMRAM ranges
- `gMmCommBufferHobGuid` - Communication buffer information
- `gMmCommonRegionHobGuid` - Common memory regions
- FV HOBs for MM driver firmware volumes

## Example Binary

See [bin/example_mm_supervisor.rs](bin/example_mm_supervisor.rs) for a complete example platform implementation.

## License

Copyright (c) Microsoft Corporation.

SPDX-License-Identifier: Apache-2.0
