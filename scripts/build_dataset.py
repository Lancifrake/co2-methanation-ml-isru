"""Build data/catalyst_methanation_dataset.csv from the original spreadsheet.

The original Excel file had three silent data-entry problems (the first two
were read as missing values by the exploratory notebooks):

1. Spreadsheet auto-formatting turned some decimal oxide contents into dates
   (e.g. 18.01 -> 18 January). The true values are restored below from the
   composition tables of the source articles.
2. The CO2/H2 ratio was typed as "1:4" in some rows; it is stored as 0.25.
3. Drag-to-fill created catalyst names that do not exist (Al-MCM-39/42/43).

Usage:
    python scripts/build_dataset.py --xlsx path/to/Database_catalyst.xlsx
"""

import argparse
import datetime as dt

import pandas as pd

# (Catalyst source, Catalyst type or None for all types) -> {column: value}
# with the table the value comes from.
OXIDE_FIXES = {
    ("S-SGS", None): ({"MgO %wt": 8.11}, "[2] Wan et al. 2022, Table S1"),
    ("hBFS", None): ({"Al2O3 %wt": 18.01, "MgO %wt": 5.50}, "[3] Chen et al. 2023, Table 1"),
    ("Kaolin-derived Al-MCM-41", "Ni-0Ce/Al-MCM-41"): ({"Al2O3 %wt": 10.8}, "[4] Table 1"),
    ("Kaolin-derived Al-MCM-41", "Ni-10Ce/Al-MCM-41"): ({"Al2O3 %wt": 10.4}, "[4] Table 1"),
    ("Kaolin-derived Al-MCM-41", "Ni-20Ce/Al-MCM-41"): ({"Al2O3 %wt": 10.9}, "[4] Table 1"),
    ("Kaolin-derived Al-MCM-41", "Ni-30Ce"): ({"Al2O3 %wt": 6.3}, "[4] Table 1"),
    ("Kaolin-derived Al-MCM-41", "Ni-50Ce/Al-MCM-41"): (
        {"SiO2 %wt": 29.1, "Al2O3 %wt": 4.1, "K2O %wt": 1.03}, "[4] Table 1"),
}

REFERENCES = pd.DataFrame([
    ("[1]", "Bahraminia S., Anbia M. Carbon dioxide methanation using Ni catalysts supported "
            "on low-cost Lynde Type A zeolite synthesized from pretreated Iranian coal gangue. "
            "Int. J. Hydrogen Energy 83 (2024) 842-855.", "10.1016/j.ijhydene.2024.08.022"),
    ("[2]", "Wan H., He Y., Su Q., Liu L., Cui X. Slag-based geopolymer microspheres as a "
            "support for CO2 methanation. Fuel 319 (2022) 123627.", "10.1016/j.fuel.2022.123627"),
    ("[3]", "Chen X., He Y., Cui X., Liu L. High value utilization of waste blast furnace slag: "
            "New Ni-CeO2/hBFS catalyst for low temperature CO2 methanation. Fuel 338 (2023) "
            "127309.", "10.1016/j.fuel.2022.127309"),
    ("[4]", "Uttamaprakrom W. et al. Development of Ni-Ce/Al-MCM-41 catalysts prepared from "
            "natural kaolin for CO2 methanation. J. Environ. Chem. Eng. 9 (2021) 106150.",
     "10.1016/j.jece.2021.106150"),
], columns=["Reference", "Citation", "DOI"])


def build(xlsx: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name="Data")

    for (source, ctype), (values, _) in OXIDE_FIXES.items():
        mask = df["Catalyst source"].eq(source)
        if ctype is not None:
            mask &= df["Catalyst type"].astype(str).str.startswith(ctype)
        for col, val in values.items():
            df.loc[mask, col] = val

    # drag-to-fill incremented "Al-MCM-41" to -39/-42/-43 in some rows; only -41 exists in [4]
    df["Catalyst type"] = df["Catalyst type"].str.replace(r"Al-MCM-(39|42|43)$", "Al-MCM-41",
                                                          regex=True)

    ratio = df["Ration CO2/H2"].astype(str).str.strip()
    df["Ration CO2/H2"] = ratio.where(ratio != "1:4", "0.25")

    leftover = df.map(lambda v: isinstance(v, (dt.datetime, pd.Timestamp))).sum().sum()
    if leftover:
        raise ValueError(f"{leftover} cells are still dates; add them to OXIDE_FIXES")

    df = df.drop(columns=["Train/Test Set"])  # constant ("Train"), not meaningful
    for c in ["SiO2 %wt", "Al2O3 %wt", "K2O %wt", "MgO %wt", "Ration CO2/H2"]:
        df[c] = pd.to_numeric(df[c])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", default="data/catalyst_methanation_dataset.csv")
    args = ap.parse_args()
    build(args.xlsx).to_csv(args.out, index=False)
    REFERENCES.to_csv("data/references.csv", index=False)
    print(f"Wrote {args.out} and data/references.csv")
