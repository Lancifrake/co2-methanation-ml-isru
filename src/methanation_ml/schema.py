"""Column names of the catalyst dataset.

Headers are kept exactly as they appear in the original spreadsheet
(including historical typos such as "calcincation" and "Ration") so that the
code can read the file without renaming anything.
"""

TARGET_CO2 = "CO2 Conversion %"
TARGET_CH4 = "CH4 selectivity %"
GROUP_COL = "Reference"  # literature source -> one LOGO group

# Bookkeeping columns, never used as features
ID_COLS = ["Number of data", "Experiment No", "Train/Test Set"]

OXIDE_COLS = [
    "SiO2 %wt", "Al2O3 %wt", "Fe2O3 %wt", "SO3 %wt",
    "K2O %wt", "MgO %wt", "CaO %wt", "TiO2 %wt",
]

NUMERIC_COLS = [
    "% wt AP",  # active-phase loading
    *OXIDE_COLS,
    "RM calcincation T° (C)", "RM calcincation time (H)",  # raw-material pretreatment
    "Calcination Temperature (C)", "Calcination time (h)",
    "Reduction Temperature (C)", "Reduction Pressure (bar)",
    "Reduction time (h)", "Reduction H2 %",
    "Temperature (C)", "Pressure (bar)", "W/F (mgcat/minml)", "Time on Stream (h)",
    "Ration CO2/H2", "CO % in feed", "Inert % in feed", "CH4 % in feed",
    "H2O % in feed", "WHSV (mL.g−1.h−1)",
]

CATEGORICAL_COLS = [
    "Catalyst source", "Catalyst type", "Active phase",
    "Raw material treatment", "Active Space",
]

DERIVED_COLS = ["Temperature_sq", "inv_WF", "pCO2_bar", "pH2_bar"]
