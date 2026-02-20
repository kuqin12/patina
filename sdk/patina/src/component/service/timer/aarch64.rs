//! AArch64-specific timer calibration routines.
//!
//! On AArch64 the generic timer (`CNTPCT_EL0` / `CNTFRQ_EL0`) typically
//! reports its frequency directly, so platform-specific calibration is
//! rarely required.
//!
//! This module is reserved for any future AArch64 calibration helpers that
//! may be needed on specific platforms.
//!
//! ## License
//!
//! Copyright (c) Microsoft Corporation.
//!
//! SPDX-License-Identifier: Apache-2.0
//!
