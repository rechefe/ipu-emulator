---
name: arch-spec
description: Write or rewrite an IPU pipeline-stage architecture spec (docs/content/specs/ARCH-stage-*.md) in the house style of ARCH-stage-aaq-str.md - overview, table of contents, block diagram, interfaces with CR field layouts, then a per-instruction ISA section with Syntax/Operands/Operation/Example. Use this whenever the user asks to write, draft, create, restructure, clean up or extend an ARCH spec, an architecture spec, a stage spec or the ISA section of a stage (MULT, ACC, CTRL, LR, AaQ, STORE, COND...), or wants a stage's instructions documented from instruction_spec.py, the emulator code, notes or a register-layout picture - even if they only say "write the spec for the ACC stage" or "document these instructions".
---

# Writing ARCH stage specs

An ARCH spec tells hardware and software engineers **what a pipeline stage does
at its boundary**: its inputs and outputs, the configuration registers it reads,
and exactly what each of its instructions computes. It is read to build RTL and
to write programs, so every sentence must carry a fact the reader acts on.

The worked example is [docs/content/specs/ARCH-stage-aaq-str.md](../../../docs/content/specs/ARCH-stage-aaq-str.md).
Read it before writing. Its structure was reached through many rounds of user
feedback; match it. `references/skeleton.md` has the same shape as a template
with placeholders.

## Workflow

1. **Gather the facts.** Sources, in the order you should read them:
   - The user's notes, messages and pictures (register layouts, diagrams).
     These describe the **target architecture** and win over everything else.
   - The slot's entries in `src/ipu-isa/src/ipu_isa/instruction_spec.py`:
     mnemonics, opcode order, operand `name` and `type`, the `InstructionDoc`.
   - The handler `execute_<name>` in `src/ipu-emu/src/ipu_emu/slots/<slot>.py`
     for the exact behavior.
   - the generated operand-type page (`docs/content/reference/isa/operand-types.md`, linked from a spec as `../reference/isa/operand-types.md`) for operand-type anchors.
   - Existing specs in `docs/content/specs/` (MAS-*, stage-*) for context.

   The code often lags the architecture. For example, the emulator's
   `ACTIVATE.QUANTIZE` has 2 operands while the ARCH spec has 5. When sources
   disagree, write the architecture the user describes, keep a list of every
   disagreement, and report it at the end. Don't edit the code.

2. **Write the spec** in the structure below, at
   `docs/content/specs/ARCH-stage-<stage>.md`. Put drawings in
   `docs/content/specs/assets/<stage>-<topic>.svg`.

3. **Run the checker** and fix every error, then go through each review item:

   ```bash
   bazel run //tools:check_arch_spec -- "$PWD/docs/content/specs/ARCH-stage-<stage>.md"
   ```

   It catches broken ToC and in-page links, missing images, sections
   duplicated by a bad edit, tiny headings, and sentences that may break the
   style rules. The review items are prompts, not orders. "Lanes that are not
   active output `NaN`" is a fact, so keep it.

4. **Register a new spec** in `docs/mkdocs.yml` under the specs nav, next to
   the stage's other specs (for example, `- AaQ Stage ARCH: specs/ARCH-stage-aaq-str.md`).

5. **Report back** in your reply, not in the doc:
   - every TBD left in the spec
   - every fact you inferred rather than read (for example, an XMEM address
     formula you derived from operand descriptions)
   - every place the spec and the code disagree

## Structure

| Section | Contents | Why |
|---------|----------|-----|
| `# <Stage> Stage` | Title | |
| `## 1. Overview` | Bullet list of the stage's tasks (bold task name, one or two sentences). Optional `> **Note:**`. | The reader learns what the stage is for in five lines. |
| `## Table of Contents` | Hand-written numbered list with anchor links, down to the deepest numbered heading. | The mkdocs sidebar shows only three levels (`toc_depth: 3`), but readers jump to 4.3.2.1. |
| `## 2. Block Diagram` | Mermaid `block-beta` diagram in the AaQ style (teal `blk` class, invisible `io` class), with every input in the left column. When an input feeds a unit in the middle of the stage, use `flowchart LR` with a `subgraph` for the stage instead (as in the MULT spec). | A `block-beta` input in its own top row renders far above the stage with its arrow detached. |
| `## 3. Interfaces` | Name / Direction / Description table. Then `### 3.N` for each CR layout and each **shared concept**. | Everything the instructions refer to is defined before the ISA. |
| `## 4. <STAGE>-<Name> ISA` | Intro (slot resolution, opcode enum source), register-width note, "Instructions in this section" table with links, then one `### 4.N` per instruction, separated by `---`. | |

There is no closing summary table and no separate "Operations" section after
the ISA. Each algorithm lives under the instruction step that uses it. The
quantization algorithm, for example, is 4.3.2.x under Quantize.

### Section 3: interfaces, CR layouts, shared concepts

- **CR layouts:** one `### 3.N \`<operand>\` Layout` per configuration register
  an instruction names. Use a Bits / Field / Description table, MSB first
  (`[31:24]` down to `[7:0]`), with unused bits as field `N/A`, description
  "Not used.". When a field has coded values, add an encoding table (Value /
  Meaning) under it. The CR layout is the **only** definition of those fields.
  Don't add a second "derived format" table, because two tables drift apart.
- **Shared concepts:** a rule that more than one instruction uses (for
  example, which elements are active) goes in section 3, before the ISA. Write it
  in general terms, not tied to one instruction. Give it a numbered rule list,
  a one- or two-line formula, a concrete example and a drawing.

### Section 4: one instruction

Order the instructions `NOP` first, then loading and configuration instructions,
then the main compute instruction. If the user names an order, use theirs.

```markdown
### 4.N `MNEMONIC`: Title

One to three sentences on what it does. (No "Summary" heading: the title
already is one.)

#### Syntax
#### Operands
#### Operation
#### Example
```

- **Syntax:** a `text` code block. Operands are separated by **commas**, which
  matches `instruction_spec.py` (`ACTIVATE.QUANTIZE activation_fn, ds, qf, ...`).
- **Operands:** "The Type column links to the [operand type reference](../operand-types.md)."
  followed by a Name | Type | Details table. Link the type to its anchor
  (`[`CrIdx`](../operand-types.md#cridx)`). Details start with the legal range
  (`CR0`-`CR15`, `0`-`4`) and then say what the operand supplies, with a
  section reference.

  **Operand types** (the Type column):
  - A register operand uses its register type: `LrIdx`, `CrIdx`, `LcrIdx`,
    `MultStageReg`, `LrdIdx`.
  - The data structure (`ds`) and the quantization format (`qf`) are 16-bit
    halves of a `CR`, type `HalfCrIdx`, written `CRn.L` (bits `[15:0]`) or
    `CRn.H` (bits `[31:16]`). Their layouts are 16-bit tables (`[15:0]`).
  - An immediate whose values are an enum (`activation_fn`, `elements_in_row`,
    `horizontal_stride`, `vertical_stride`, `lut_addr`) is `Imm`, not its enum
    name. Link `Imm` to the enum's entry in `operand-types.md`
    (`[`Imm`](../operand-types.md#horizontalstride)`), and list the enum
    values in Details.
  The names in `instruction_spec.py` and `operand-types.md` can differ from
  these. The spec follows these rules.

  For a multi-step instruction add a **Used by** column naming the step. List
  only the legal values. Leave out invalid encodings ("`5`-`7` are invalid").
- **Operation:** a `text` pseudocode block with `//` comments that name the
  section where each step is defined. Name the stage's inputs by their
  interface names (`input_data[i]`), never by internal registers (`r_acc`).
- **Example:** a `text` block ending with `;;`, with a short `//` comment if
  it helps.
- **Mechanisms and steps:** numbered `#### 4.N.k` subsections after the
  Example. An instruction that does several things gets a step table
  (Step | What happens | Section) in its description, then one subsection per
  step. `ACTIVATE.QUANTIZE` has 4.3.1 Activate, 4.3.2 Quantize and 4.3.3 Store.
  A structure only one instruction uses (for example, the LUT that `LOAD_LUT`
  fills) lives under that instruction.

## Style rules

The user reads every sentence and cuts the ones that waste their time. These
rules come from that feedback.

**Say each fact once.** Before writing a sentence, check that a table row, an
earlier definition, or another sentence doesn't already carry it. That excludes:
- pointers back like "as in section 4.3.2.2"
- pseudocode that repeats a table
- "for the 8-bit case, `fm1 = 6`" when a formats table already shows `e1m6`

When two forms say the same thing, keep the one that is easier to read. A table
of branches plus a table of edge cases is easier to read than pseudocode.

**No meta.** Leave out:
- roadmaps ("Both modes use the same scale; dynamic mode adds...")
- "This section describes..."
- history ("renamed from", "used to", "the old text")
- the reason a fact is absent ("the format is not written because...")

**No "it is A, not B"** unless the reader would otherwise believe B. If deleting
a qualifier leaves the same information, delete it. "Activation applies to every
element" needs no "it does not depend on `ds`".

**Architectural level.** Describe the bit layout of fields the architecture
exposes. Leave out:
- how those fields map onto implementation words or buses (word boundaries,
  `input_data[8][31:8]`)
- what unused or leftover bits contain
- implementation choices

When the user asks "why do we care", that content was below the architecture
line.

**Tables and short lines over paragraphs.** Each edge case is a table row
(Function | Input | Result). Each mode is a row. Use bold lead-ins (**Source.**,
**Scale.**) for short parts that don't need a heading. Break long paragraphs
into bullets or numbered steps.

**Readable headings.**
- Instructions are `###`.
- Their parts (Syntax, numbered steps, value tables, special cases) are `####`.
- Use `#####` only below a numbered `####`.
- Avoid `######`: it renders as small as body text.

Use `---` between instructions and between an instruction's main parts.

**Names.**
- Field and variable names are lower_case_with_underscores, with whole words:
  `num_partition`, `valid_elements`, `nx_format`, `num_bits`, not `num_part`
  or `valid_elem`. The exception is `N/A`.
- Prefer descriptive names (`group_size`) to one-letter ones (`G`). Keep
  established symbols (`s`, `E`, `fe`, `fm`).
- Mnemonics and registers are upper case (`LOAD_LUT`, `CR15`, `LR0`).
  Operands are lower case.
- One name per thing everywhere: text, tables, pseudocode and drawings. When
  you rename something, update every occurrence, including the SVG labels.

**Unknowns.** Never invent an encoding, polarity or width. Write
`> **TBD:** <precise question>`. If the user names who knows, write
`> **TBD - ask <name>:**`. Tell the user what you inferred in your reply,
never inside the doc.

**Notes** are `> **Note:**` blockquotes, one fact each. The `NaN`/`inf`
exclusion from the scale and the explicit XMEM address are examples.

## Drawings

Draw when a layout, a partition of elements, or a function shape is easier to
see than read. That covers the active-elements grid and the `generic` LUT
versions.
- Block diagrams are Mermaid in the markdown.
- Other drawings are SVG files in `docs/content/specs/assets/`, with:
  - a white background `<rect>`
  - Helvetica or Arial text
  - teal `#2e9e8c` / `#1a7060` for the highlighted parts
  - labels that use the spec's exact names
- Check that every SVG parses:

  ```bash
  python3 -I -c "import xml.dom.minidom as m, sys; [m.parse(f) for f in sys.argv[1:]]" docs/content/specs/assets/<stage>-*.svg
  ```

## Editing an existing spec safely

Specs are long, and a splice that matches the wrong place can silently
duplicate half the file.
- Anchor every scripted replacement on a string that occurs exactly once, and
  assert that it does.
- When cutting from one heading to the next, make sure the end marker comes
  after the start.
- Run `//tools:check_arch_spec` after every structural edit. It reports a repeated section
  number.
- When you renumber a section, update the Table of Contents and every
  "section X.Y" reference in the same edit.

Anchor slugs follow python-markdown: lower case, punctuation (including
backticks, dots and colons) removed, spaces become `-`. For example,
`### 4.3 \`ACTIVATE.QUANTIZE\`: Activate and Quantize` becomes
`#43-activatequantize-activate-and-quantize`. The checker computes these, so
trust it over your head.
