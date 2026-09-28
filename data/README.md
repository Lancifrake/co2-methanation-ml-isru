# Data

## `catalyst_methanation_dataset.csv`

108 CO₂ methanation experiments (104 with both targets) compiled by hand from
four published articles on Ni catalysts supported on waste-derived materials.
Values were read from the articles' text, tables and figures and harmonised to
common units. `references.csv` maps the `Reference` codes to the sources:

| Code | Source | Support family |
|---|---|---|
| [1] | Bahraminia & Anbia, *Int. J. Hydrogen Energy* 83 (2024) 842–855. doi:10.1016/j.ijhydene.2024.08.022 | zeolite from coal gangue |
| [2] | Wan et al., *Fuel* 319 (2022) 123627. doi:10.1016/j.fuel.2022.123627 | slag-based geopolymer microspheres |
| [3] | Chen et al., *Fuel* 338 (2023) 127309. doi:10.1016/j.fuel.2022.127309 | blast-furnace slag (hBFS) |
| [4] | Uttamaprakrom et al., *J. Environ. Chem. Eng.* 9 (2021) 106150. doi:10.1016/j.jece.2021.106150 | kaolin-derived Al-MCM-41 |

The compilation is shared under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The underlying measurements belong to their authors: cite the articles above
if you reuse the data. Values digitised from figures carry reading uncertainty.

## Corrections applied to the original spreadsheet

`scripts/build_dataset.py` regenerates the CSV from the original Excel file and
applies these fixes:

| Problem | Cells | Fix | Source of the correct value |
|---|---|---|---|
| Oxide contents auto-converted to dates (e.g. 18.01 → "18 Jan") | 112 (SiO₂, Al₂O₃, K₂O, MgO in [2], [3], [4]) | restored numeric values | [2] Table S1, [3] Table 1, [4] Table 1 |
| CO₂/H₂ ratio typed as text "1:4" | 66 | 0.25 | same value |
| Drag-to-fill created names "Ni-30Ce/Al-MCM-39/42/43" | 3 | "Ni-30Ce/Al-MCM-41" | [4] |

Before these fixes the models read the affected cells as missing, so the
exploratory results in the project slides were computed on a partly empty
oxide table.

## Schema

| Column | Type | Role |
|---|---|---|
| `Reference` | str | literature source → LOGO group (never a feature) |
| `Number of data`, `Experiment No` | — | bookkeeping, dropped |
| `Catalyst source`, `Catalyst type`, `Active phase`, `Raw material treatment`, `Active Space` | categorical | features |
| `% wt AP` | wt% | active-phase (Ni) loading |
| `SiO2 %wt` … `TiO2 %wt` (8 oxides) | wt% | support composition |
| `RM calcincation T° (C)`, `RM calcincation time (H)` | °C, h | raw-material pretreatment |
| `Calcination Temperature (C)`, `Calcination time (h)` | °C, h | catalyst calcination |
| `Reduction Temperature (C)`, `Reduction Pressure (bar)`, `Reduction time (h)`, `Reduction H2 %` | | activation |
| `Temperature (C)`, `Pressure (bar)`, `W/F (mgcat/minml)`, `WHSV (mL.g−1.h−1)`, `Time on Stream (h)` | | reaction conditions |
| `Ration CO2/H2`, `CO % in feed`, `Inert % in feed`, `CH4 % in feed`, `H2O % in feed` | | feed |
| `CO2 Conversion %` | % | target |
| `CH4 selectivity %` | % | target |

Empty cells mean "not reported in the source". Header typos ("calcincation",
"Ration") are kept to match the original spreadsheet.

## Synthetic data for tests

`scripts/make_synthetic_data.py` generates a file with the same columns and
invented values; the test suite uses it so tests do not depend on the real data.
