"""Generate a SYNTHETIC dataset with the same schema as the real one.

The real literature-compiled dataset is not distributed. This file only lets
the pipeline run end-to-end. Values come from a made-up response function
and carry no chemical meaning: do not interpret results obtained on it.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {
    "[A]": dict(source="Support A", SiO2=45, Al2O3=38, CaO=1, t_offset=0),
    "[B]": dict(source="Support B", SiO2=30, Al2O3=14, CaO=40, t_offset=25),
    "[C]": dict(source="Support C", SiO2=35, Al2O3=15, CaO=35, t_offset=-20),
    "[D]": dict(source="Support D", SiO2=80, Al2O3=5, CaO=1, t_offset=10),
}


def make(n_per_source: int = 26, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for ref, s in SOURCES.items():
        loading = rng.choice([5, 10, 15, 20])
        for _ in range(n_per_source):
            k += 1
            T = rng.choice(np.arange(200, 551, 50))
            ox = {o: max(v + rng.normal(0, 1.5), 0) for o, v in
                  [("SiO2", s["SiO2"]), ("Al2O3", s["Al2O3"]), ("CaO", s["CaO"])]}
            # made-up bell-shaped conversion curve + noise
            x_co2 = 85 * np.exp(-((T - 400 - s["t_offset"]) / 110) ** 2) + loading * 0.4
            x_co2 = float(np.clip(x_co2 + rng.normal(0, 5), 0, 100))
            s_ch4 = float(np.clip(99 - 0.02 * abs(T - 380) + rng.normal(0, 1.5), 55, 100))
            rows.append({
                "Number of data": k, "Reference": ref, "Experiment No": k,
                "Train/Test Set": "", "Catalyst source": s["source"],
                "Catalyst type": f"{loading}%Ni/{s['source']}", "Active phase": "Ni",
                "% wt AP": loading,
                "SiO2 %wt": ox["SiO2"], "Al2O3 %wt": ox["Al2O3"], "Fe2O3 %wt": np.nan,
                "SO3 %wt": np.nan, "K2O %wt": np.nan, "MgO %wt": np.nan,
                "CaO %wt": ox["CaO"], "TiO2 %wt": np.nan,
                "Raw material treatment": "none",
                "RM calcincation T° (C)": 600, "RM calcincation time (H)": 2,
                "Active Space": "nan",
                "Calcination Temperature (C)": 500, "Calcination time (h)": 3,
                "Reduction Temperature (C)": 450, "Reduction Pressure (bar)": 1,
                "Reduction time (h)": 2, "Reduction H2 %": 50,
                "Temperature (C)": T, "Pressure (bar)": 1, "W/F (mgcat/minml)": 3,
                "Time on Stream (h)": 1, "Ration CO2/H2": 4, "CO % in feed": 0,
                "Inert % in feed": 20, "CH4 % in feed": 0, "H2O % in feed": 0,
                "WHSV (mL.g−1.h−1)": 20000,
                "CH4 selectivity %": s_ch4, "CO2 Conversion %": x_co2,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/sample/synthetic_example.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    make(seed=args.seed).to_csv(args.out, index=False)
    print(f"Wrote {args.out} (SYNTHETIC data)")
