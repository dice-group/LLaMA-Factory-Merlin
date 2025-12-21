# Merlin Fork Changes (LLaMA-Factory + PEFT)

This document records the changes made in this fork relative to the upstream
LLaMA-Factory repo and upstream PEFT. It is meant to be a precise, searchable
reference for what we added or modified to support the multi-project workflow
and custom PEFT tuners.

If you need a short summary: this fork vendors PEFT v0.17.1, adds custom PEFT
tuners (CoLA, HydraLoRA, AdaMole, MoLA, MoELPR), and extends LLaMA-Factory to
support language routing metadata and extra loss terms used by those tuners.

## Baseline

- LLaMA-Factory base: the current state of this repository before these changes
  were applied.
- PEFT base: vendored copy of upstream PEFT v0.17.1 under `src/peft/`.

## High-level differences

1) **PEFT is vendored in-tree**
   - Added: `src/peft/` (full upstream PEFT v0.17.1 source).
   - Removed: `peft` dependency from `requirements.txt`.
   - Rationale: keep a single, shared, controlled PEFT implementation for both
     downstream projects and avoid version drift.

2) **Custom PEFT tuners integrated**
   - Added tuners and configs:
     - CoLA: `ColaConfig`, `ColaModel`
     - HydraLoRA: `HydraLoraConfig`, `HydraLoraModel`
     - AdaMole: `AdaMoleConfig`, `AdaMoleModel`
     - MoLA: `MolaConfig`, `MolaModel` (registered under `adamoe_1` path)
     - MoELPR: `MoelprConfig`, `MoelprModel`
   - Added metrics helpers: `src/peft/metrics.py`
   - Registered new PEFT types:
     - `COLA`, `HYDRALORA`, `ADAMOLE`, `MOLA`, `MOELPR`
   - All of these are wired into PEFT registries and `peft/__init__.py`.

3) **Language routing / multilingual support**
   - Added language metadata plumbing in LLaMA-Factory:
     - dataset fields, collator behavior, and model adapter config use
       `language_ids` to route tokens to experts.
   - Added `src/llamafactory/extras/language.py` with helpers for language maps
     and ID conversions.

4) **Training loop support for additional losses**
   - Extra loss terms used by AdaMole / MoLA / MoELPR added to SFT trainer.

## Detailed change log (file-level)

### PEFT vendoring and tuning extensions

**New / updated files**

- `src/peft/` (vendored upstream PEFT v0.17.1)
- `src/peft/metrics.py`
  - Adds metrics recording helpers.
  - Includes `record_cola_metrics` and `record_hydralora_metrics`.

**Updated PEFT type system**

- `src/peft/utils/peft_types.py`
  - Adds the new PEFT method types: `COLA`, `HYDRALORA`, `ADAMOLE`, `MOLA`, `MOELPR`.

**New tuners and configs**

- `src/peft/tuners/cola/`
  - Implements CoLA layers and model wrapper.
- `src/peft/tuners/hydralora/`
  - Implements HydraLoRA layers and model wrapper.
- `src/peft/tuners/adamole/`
  - Implements AdaMole layers and model wrapper.
- `src/peft/tuners/adamoe_1/`
  - Implements MoLA layers and model wrapper (registered as `mola`).
- `src/peft/tuners/moelpr/`
  - Implements MoELPR layers and model wrapper.

**Registry updates**

- `src/peft/tuners/__init__.py`
  - Exposes all custom tuners and configs.
- `src/peft/__init__.py`
  - Exports new config/model classes at the top level.
- `src/peft/tuners/*/__init__.py`
  - Registers tuners via `register_peft_method`.
  - Uses `prefix="lora_"` for all new tuners to match adapter naming.

**Loading/saving and forward args**

- `src/peft/utils/save_and_load.py`
  - Adds new tuners into LoRA-like handling.
  - Uses `getattr(config, "use_dora", False)` to avoid missing fields.
- `src/peft/peft_model.py`
  - Adds `language_ids` and `family_ids` to `special_peft_forward_args`.

**CoLA / Hydra-specific changes**

- `src/peft/tuners/cola/model.py`
  - Passes `language_to_subgroup_ids` into the layer forward.
- `src/peft/tuners/hydralora/model.py`
  - Same `language_to_subgroup_ids` support as CoLA.
- `src/peft/tuners/cola/config.py`
  - Accepts `random_ab/random_ba` and maps `random` to `random_ab`.

### LLaMA-Factory changes for language routing and new PEFT types

**Language metadata helpers**

- `src/llamafactory/extras/language.py`
  - Implements:
    - `load_language_map`
    - `load_language_groupings`
    - `build_language_vocab`
    - `language_to_ids`
    - `LANGUAGE_PAD_ID`

**Data args**

- `src/llamafactory/hparams/data_args.py`
  - Adds:
    - `language_column`
    - `language_map`
    - `_language_metadata` cache
    - `get_language_metadata`

**Dataset parsing and conversion**

- `src/llamafactory/data/parser.py`
  - Adds dataset attribute `language`.
  - Extends column parsing to include it.
- `src/llamafactory/data/converter.py`
  - Adds `_extract_language`.
  - Adds `_language` to outputs for Alpaca / ShareGPT / OpenAI formats.

**Data processor**

- `src/llamafactory/data/processor/supervised.py`
  - Adds `language_ids` in features using `language_to_ids`.
  - Adds `LANGUAGE_PAD_ID` for packed sequences.

**Collator**

- `src/llamafactory/data/collator.py`
  - Collects `language_ids` and includes them in the batch tensor when present.

**Finetuning args**

- `src/llamafactory/hparams/finetuning_args.py`
  - Adds finetuning types: `cola`, `hydralora`, `adamole`, `mola`, `moelpr`.
  - Adds configuration fields for:
    - CoLA and HydraLoRA
    - AdaMole, MoLA, MoELPR
    - Language routing and auxiliary loss weights
  - Adds validation for MoELPR stage settings and LoRA-like tuners.

**Argument parsing sync**

- `src/llamafactory/hparams/parser.py`
  - Adds `_sync_language_metadata` to align `language_map` and
    `language_column` between `data_args` and `finetuning_args`.
  - Loosens adapter validation to allow new finetuning types.

**Adapter setup for new tuners**

- `src/llamafactory/model/adapter.py`
  - Adds imports for the new PEFT configs.
  - Adds helper functions:
    - `_build_language_metadata`
    - `_parse_optional_int_list`
  - Adds setup functions:
    - `_setup_cola_tuning`
    - `_setup_hydralora_tuning`
    - `_setup_adamole_tuning`
    - `_setup_mola_tuning`
    - `_setup_moelpr_tuning`
  - Wires these into `init_adapter`.
  - Disables Unsloth/KTransformers for custom tuners to avoid incompatibility.

**Trainer support**

- `src/llamafactory/train/sft/trainer.py`
  - Adds extra loss terms for AdaMole/MoLA/MoELPR.
  - Adds language prior loss and MoELPR masks where required.
  - Imports `torch.nn.functional as F`.

- `src/llamafactory/train/sft/workflow.py`
  - Sets up MoELPR stage-2 language IDs using `data_args.language_map`.

### Requirements / dependency changes

- `requirements.txt`
  - Removed `peft` dependency because it is vendored in `src/peft/`.

## Notes and compatibility

- This fork intentionally keeps the LLaMA-Factory base version unchanged
  while upgrading PEFT to v0.17.1 and merging custom tuners.
- The custom tuners are LoRA-like and are registered with `prefix="lora_"`
  for compatibility with adapter naming and saving/loading behavior.
- Language routing is opt-in; it uses `language_column` and `language_map`
  to generate `language_ids` used by custom tuners.

## Suggested verification (manual)

- Import test:
  - `python -c "import llamafactory, peft; print('ok')"`
- Sanity check that new tuning types parse:
  - `finetuning_type=cola` or `finetuning_type=moelpr`

## Where the changes came from

- CoLA / HydraLoRA code: derived from `HTYLLM-PG` extensions.
- AdaMole / MoLA / MoELPR code: derived from `moe-study` extensions.

