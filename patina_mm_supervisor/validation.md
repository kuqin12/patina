# MM Supervisor Initialization Validation

The supervisor runs on state MM Init leaves behind: the HOB list, the page table, the loaded images, the MMI entry code,
and each core's GDT, TSS and stacks. It does not rebuild that state. The BSP validates it during initialization, before
any Ring 3 code runs, and fails the boot on any mismatch.

This document lists what is validated, what MM Init must do for the checks to pass, and what is left.

## Approach

- The supervisor adopts MM Init's page table (`init_page_table`) and checks it rather than building its own. Every check
  uses data that already exists: the HOB list, the `PassDown` HOB, the MMI entry fixups, the PSD, the GDT and TSS, and
  CPU state. No HOB fields were added for validation.
- Images are checked per page from their allocation HOBs. Ring 0 does not parse PE headers.
- Nothing here relies on SEA, which reports long after initialization. Where SEA checks the same thing, it is a second
  opinion.
- Anything inside MMRAM that the supervisor cannot name must be supervisor-only and non-executable. That replaces the
  need for per-allocation owner and type data.

## When the Checks Run

The BSP, in `bsp_init` (`src/mmcore/init.rs`):

1. `validate_hob_list_structure`, before any HOB is read.
2. `validate_incoming_hobs_pre_paging_init` (`src/hob/validation.rs`), before anything is written into MMRAM.
3. `validate_incoming_hobs_post_paging_init` (`src/hob/validation.rs`), once the page table is adopted.
4. `validate_page_table_post_activation` (`src/memory/validation.rs`), the sweep over the whole page table.

Then `validate_mmi_entry` (`src/mmcore/validation.rs`) runs right after `bsp_init`, on the BSP for its own entry and for
every core's page protections and distinctness. Each AP repeats it for itself on its first entry.

## What Is Validated

### HOB List

- **Structure:** the headers form a well-formed list that ends within 1 MiB.
- **MMRAM:** the MMRAM regions form one contiguous, non-overlapping span at or above 1 MiB that holds the supervisor.
  The HOB list and the MP Information HOB payload lie inside it.
- **Memory allocation HOBs:** none overlaps MMRAM.
- **Module HOBs:** Init, Core and User are each described exactly once. Every module allocation lies inside MMRAM, is
  page aligned, overlaps no other, and holds its entry point. At most 64 are accepted.
- **Resource descriptors:** none overlaps another in the same address space. One system or reserved memory descriptor
  holds all of MMRAM, so no MMIO or firmware-device descriptor reaches into it. V2 descriptors carry valid attributes.
  Only V2 descriptors are checked, or only V1 with the `v1_resource_descriptor_support` feature.
- **`PassDown` HOB:** the revision matches. Its buffers, sized for the CPU count, lie in MMRAM pages the producer marked
  allocated. Together with the module allocations, the HOB list, the SMBASE area and MSEG, no two share a page. The
  Ring 3 stacks start and end on page boundaries, and the SMBASE array starts on one. Inside the SMBASE area, no CPU's
  SMI entry code or SMM descriptor and save state overlaps another's.
- **MSEG HOB:** when present, it describes a non-empty, page-aligned MSEG below 4 GiB.
- **Unblock-region and protected-region HOBs:** each is non-empty, page aligned and outside MMRAM. Unblock regions name
  their requester with a non-zero GUID. Protected regions are of the IOMMU type only.
- **CPU HOB:** the physical address width does not exceed the processor's MAXPHYADDR.
- **Communication buffers:** the external buffers and status mailboxes lie outside MMRAM and are supervisor-only
  (`require_external_comm_buffer`, when the channels are built).

### Named Regions in the Page Table

Checked page by page in `validate_incoming_hobs_post_paging_init`. S is supervisor-only, U user-accessible, NP not
present. No page may be both writable and executable, and every SMBASE must be page aligned.

| Region                                                                                        | Required                                                                                                       |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Supervisor core                                                                               | S, present; entry point on a read-only, executable page                                                        |
| User core and drivers                                                                         | U, present; entry point on a read-only, executable page                                                        |
| MM Init image                                                                                 | S, present. The entry point is not checked: MM Init maps its own image as data and never runs under this table |
| Each CPU's SMI entry code (`SMBASE + 0x8000`)                                                 | S, read-only, executable                                                                                       |
| Each CPU's SMM descriptor and save state (`SMBASE + 0xFB00` to `+ 0x10000`)                   | S, non-executable                                                                                              |
| Firmware policy                                                                               | S, read-only, non-executable                                                                                   |
| SMBASE array                                                                                  | S, non-executable                                                                                              |
| MM-initialized buffer                                                                         | S, writable, non-executable                                                                                    |
| Ring 3 stacks                                                                                 | U, writable, non-executable                                                                                    |
| MSEG                                                                                          | NP                                                                                                             |
| Page 0, and everything from the physical address width to the top of the lower canonical half | NP                                                                                                             |

### Whole Page Table

`validate_page_table_post_activation` walks every present leaf:

- The root at CR3 lies inside MMRAM, and every leaf is an identity mapping.
- Adjacent identity mappings with equal attributes form one blanket, checked as a whole. A blanket never spreads across
  an MMRAM boundary.
- **Outside MMRAM,** every present page lies in a region a HOB names, with the attributes MM Init gives that region.
  Every region that reaches into a blanket must allow its attributes, and together they must cover it. System memory
  and the gaps between resource descriptors are therefore not present.

  | Source                                        | Required                                                    |
  | --------------------------------------------- | ----------------------------------------------------------- |
  | MMIO and firmware-device resource descriptors | U, non-executable                                           |
  | Unblock-region HOBs                           | Non-executable; S exactly when the HOB sets `EFI_MEMORY_SP` |
  | Communication buffers and status mailboxes    | S, non-executable                                           |
  | IOMMU protected regions                       | U, read-only, non-executable                                |

- **Inside MMRAM,** no page is both writable and executable, user-accessible pages lie only in the user core, a driver
  or the Ring 3 stacks, and executable pages lie only in the supervisor core, the user core, a driver or the SMI entry
  code. Every other page is supervisor-only and non-executable.

The unblocked memory tracker is seeded later from this same table (`record_unblocked_memory`). Nothing between the
sweep and that walk maps memory outside MMRAM, so the tracker holds only validated ranges.

### MMI Entry and Per-Core State

`validate_mmi_entry` checks the V5 MMI entry each core arrived through, and the state it left:

- **Fixups:** header version 5. The CR3 slot holds the live CR3, the rendezvous slot the supervisor entry point, and the
  HOB slot the HOB list the entry code passed. The SMBASE slot matches the SMBASE array and, when readable,
  `IA32_SMBASE`. XD is set, `MSR_IA32_MISC_ENABLE` support matches the vendor, the 5-level paging flag matches CR4.LA57,
  and the debug, CET and reserved slots are zero.
- **Registers:** IDTR and GDTR hold what the entry code loaded, with the expected limits, and the IDT lies inside MMRAM.
  CR4 has UMIP, SMEP and SMAP set.
- **GDT and TSS:** the segments, the call gate and the TSS descriptor match the `SmiException.nasm` template, and the
  TSS directly follows the GDT. The TSS limit is byte-granular and its I/O map base lies at or beyond the limit, so it
  has no I/O permission bitmap and every Ring 3 port access faults.
- **PSD** at `SMBASE + 0xFB00`: the signature, and the CR3 and GDT descriptor the entry code loads.
- **Ring 0 stack:** the slot at its top holds the core's index.
- **Page protections:** the entry code is S, read-only and executable. The IDT and the IDT and GDT descriptors are S and
  non-executable. The GDT and TSS are S, read-only and non-executable. The top pages of the Ring 0 and IST1 stacks are
  S, writable and non-executable. IST1 is page aligned and the page it starts is not present: that is the stack guard.

On the BSP it also checks, for every core, before any Ring 3 code runs:

- every core's entry loads the same CR3;
- no two cores share a GDT, a GDT descriptor, a Ring 0 stack or an IST1 stack;
- every core's page protections, as above. Each core's entry code reads its GDT and writes its stack before turning
  paging on, where neither page protection nor SMAP applies, so this cannot wait for the AP's own first entry.

## Requirements on MM Init

The checks above fail unless MM Init (`MmSupervisorPkg`) does the following.

| Requirement                                                                                                                                               | Check that depends on it                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Free its FFS copies of the drivers and of the user core image before `LockMmCoreBeforeExit` snapshots the memory map                                      | User pages inside MMRAM                                                                      |
| Allocate no internal copy of the user communication buffer                                                                                                | User pages inside MMRAM                                                                      |
| With MM heap page guard enabled, make every recorded guard page not present before handoff (`MmEntryPointMemoryManagementHook` in `LockMmCoreBeforeExit`) | User pages inside MMRAM: otherwise the guard below the Ring 3 stacks is mapped user-writable |
| Map the GDT read-only (`PatchGdtIdtMap`)                                                                                                                  | GDT and TSS protections                                                                      |
| Assemble the TSS without an I/O permission bitmap: G = 0, I/O map base = TSS size (`SmiException.nasm`)                                                   | TSS check                                                                                    |
| Build with `PcdCpuSmmStackGuard`, enforced by a `STATIC_ASSERT`                                                                                           | IST1 inside MMRAM; stack guard not present                                                   |
| Leave MSEG not present (`SEA_MSEG_ATTRIBUTE` = `EFI_MEMORY_RP`)                                                                                           | MSEG not present                                                                             |
| Allocate the SMBASE array with `AllocatePages`                                                                                                            | SMBASE array page aligned                                                                    |
| Use the V5 MMI entry code (`MmiEntrySeaV5`)                                                                                                               | Fixup checks; CR4 SMAP                                                                       |

## What Is Left

| Item                  | What is missing                                                                                                                                                                                                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Paging structures     | Only the root is shown to lie inside MMRAM. Not yet checked: every non-leaf entry's target lies inside MMRAM before it is read, and every table page is supervisor-only, non-executable, read-only and outside every named region. This needs `patina_paging` to expose non-leaf entries; its iterator yields only leaves |
| Root self-map entries | The root must have no entry at index 0x1FE or 0x1FF. `patina_paging`'s iterator skips both when 0x1FF points back to the root, so a crafted self-map would be invisible to the sweep. MM Init never creates one                                                                                                           |
