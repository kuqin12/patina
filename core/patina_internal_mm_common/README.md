# patina_internal_mm_common

Shared type definitions used by both the MM Supervisor Core and MM User Core.

This crate provides the communication structures and enumerations that define
the ABI between the supervisor (ring 0) and user (ring 3) MM modules,
including:

- `EfiMmEntryContext` — PI specification entry context
- `MmCommBufferStatus` — Communication buffer status flags
- `EfiMmCommunicateHeader` — MMI communication header
- `MmCommonBufferHobData` — Communication buffer HOB data
- `UserCommandType` — Supervisor-to-user command enumeration
- `MM_COMM_BUFFER_HOB_GUID` — Shared GUID for the communication buffer HOB
