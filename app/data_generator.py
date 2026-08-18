import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

np.random.seed(42)

N = 10000

machine_ids = [
    f"M{str(i).zfill(3)}"
    for i in range(1, 101)
]

machine_types = [
    "CNC",
    "Lathe",
    "Press",
    "Milling",
    "Grinding"
]


# --------------------------------------------------
# 2. Generate base machine data
# --------------------------------------------------

df = pd.DataFrame({
    "Machine_ID": np.random.choice(
        machine_ids, N
    ),

    "Timestamp": pd.date_range(
        start="2025-01-01",
        periods=N,
        freq="h"
    ),

    "Machine_Type": np.random.choice(
        machine_types,
        N
    ),

    "Machine_Age_Years": np.random.randint(
        1, 16, N
    ),

    "Operating_Hours": np.random.randint(
        500, 50000, N
    ),

    "Temperature_C": np.random.normal(
        70, 10, N
    ),

    "Vibration_mm_s": np.random.normal(
        4, 1.5, N
    ),

    "Pressure_bar": np.random.normal(
        6, 1, N
    ),

    "Motor_Current_A": np.random.normal(
        20, 4, N
    ),

    "Production_Count": np.random.randint(
        100, 2000, N
    ),

    "Production_Rate": np.random.normal(
        85, 10, N
    ),

    "Last_Maintenance_Days": np.random.randint(
        1, 365, N
    ),

    "Maintenance_Count": np.random.randint(
        0, 20, N
    ),

    "Failure_Count": np.random.poisson(
        2, N
    )
})


# --------------------------------------------------
# 3. Keep sensor values realistic
# --------------------------------------------------

df["Temperature_C"] = df["Temperature_C"].clip(
    40, 110
)

df["Vibration_mm_s"] = df["Vibration_mm_s"].clip(
    0.5, 12
)

df["Pressure_bar"] = df["Pressure_bar"].clip(
    2, 10
)

df["Motor_Current_A"] = df["Motor_Current_A"].clip(
    5, 40
)

df["Production_Rate"] = df["Production_Rate"].clip(
    40, 120
)


# --------------------------------------------------
# 4. Create normalized risk factors
# --------------------------------------------------

age_risk = (
    df["Machine_Age_Years"] - 1
) / 14

hours_risk = (
    df["Operating_Hours"] - 500
) / 49500

temperature_risk = np.maximum(
    df["Temperature_C"] - 65,
    0
) / 45

vibration_risk = np.maximum(
    df["Vibration_mm_s"] - 2.5,
    0
) / 9.5

maintenance_risk = (
    df["Last_Maintenance_Days"] / 365
)

failure_history_risk = (
    df["Failure_Count"] / 8
)


# --------------------------------------------------
# 5. Build nonlinear failure risk
# --------------------------------------------------

risk_score = (
    1.8 * age_risk
    + 2.2 * hours_risk
    + 2.8 * temperature_risk
    + 3.0 * vibration_risk
    + 2.0 * maintenance_risk
    + 1.5 * failure_history_risk
)


# Interaction effects

risk_score += (
    2.0
    * temperature_risk
    * vibration_risk
)

risk_score += (
    1.5
    * hours_risk
    * maintenance_risk
)


# --------------------------------------------------
# 6. Convert risk score to probability
# --------------------------------------------------

failure_probability = (
    1 /
    (
        1 +
        np.exp(
            -(risk_score - 7.0)
        )
    )
)


# Add controlled randomness

failure_probability = (
    0.85 * failure_probability
    + 0.15 * np.random.random(N)
)


failure_probability = np.clip(
    failure_probability,
    0.01,
    0.95
)


# --------------------------------------------------
# 7. Generate target
# --------------------------------------------------

df["Failure_Next_30_Days"] = (
    np.random.random(N)
    < failure_probability
).astype(int)


# --------------------------------------------------
# 8. Generate realistic failure types
# --------------------------------------------------

df["Failure_Type"] = "No Failure"


# High vibration → bearing failure

condition = (
    (df["Vibration_mm_s"] > 6)
    &
    (df["Failure_Next_30_Days"] == 1)
)

df.loc[
    condition,
    "Failure_Type"
] = "Bearing Failure"


# High temperature → overheating

condition = (
    (df["Temperature_C"] > 90)
    &
    (df["Failure_Next_30_Days"] == 1)
    &
    (df["Failure_Type"] == "No Failure")
)

df.loc[
    condition,
    "Failure_Type"
] = "Overheating"


# High motor current → electrical failure

condition = (
    (df["Motor_Current_A"] > 28)
    &
    (df["Failure_Next_30_Days"] == 1)
    &
    (df["Failure_Type"] == "No Failure")
)

df.loc[
    condition,
    "Failure_Type"
] = "Electrical Failure"


# High pressure → hydraulic failure

condition = (
    (df["Pressure_bar"] > 8)
    &
    (df["Failure_Next_30_Days"] == 1)
    &
    (df["Failure_Type"] == "No Failure")
)

df.loc[
    condition,
    "Failure_Type"
] = "Hydraulic Failure"


# Remaining failures → motor failure

condition = (
    (df["Failure_Next_30_Days"] == 1)
    &
    (df["Failure_Type"] == "No Failure")
)

df.loc[
    condition,
    "Failure_Type"
] = "Motor Failure"


# --------------------------------------------------
# 9. Save dataset
# --------------------------------------------------

output_path = (
    "data/raw/manufacturing_data.csv"
)

df.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# 10. Display summary
# --------------------------------------------------

print("=" * 50)
print("Manufacturing Dataset Created")
print("=" * 50)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\nFailure distribution:")

print(
    df[
        "Failure_Next_30_Days"
    ].value_counts()
)

print("\nFailure percentage:")

print(
    df[
        "Failure_Next_30_Days"
    ].value_counts(
        normalize=True
    ) * 100
)

print("\nFailure types:")

print(
    df[
        "Failure_Type"
    ].value_counts()
)

print("\nMissing values:")

print(
    df.isnull().sum().sum()
)

print("\nDataset saved to:")

print(output_path)