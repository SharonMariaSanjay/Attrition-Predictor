"""
Generates a synthetic employee dataset with realistic attrition patterns,
so the project runs fully offline with no dataset download or signup
(no Kaggle account needed). The structure mirrors the well-known IBM HR
Analytics Employee Attrition dataset, so this is a drop-in replacement —
swap in the real Kaggle CSV later if you want to use real-world data
instead (see README).

Attrition is NOT random here — it's generated to genuinely correlate with
overtime, low satisfaction, low income, and long commutes, so the model
has real signal to learn from (not just noise).
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 800

age = np.random.randint(21, 60, N)
monthly_income = np.random.randint(15000, 150000, N)
years_at_company = np.clip((age - 21) - np.random.randint(0, 5, N), 0, None)
job_satisfaction = np.random.randint(1, 5, N)  # 1=Low, 4=Very High
distance_from_home = np.random.randint(1, 40, N)  # km
overtime = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])
work_life_balance = np.random.randint(1, 5, N)  # 1=Bad, 4=Best
num_promotions = np.random.randint(0, 4, N)

# Build attrition probability from real, interpretable HR drivers
risk = (
    0.35 * (overtime == "Yes").astype(int)
    + 0.30 * (job_satisfaction <= 2).astype(int)
    + 0.20 * (monthly_income < 35000).astype(int)
    + 0.15 * (distance_from_home > 25).astype(int)
    + 0.15 * (work_life_balance <= 2).astype(int)
    + 0.10 * (years_at_company < 2).astype(int)
    - 0.15 * (num_promotions >= 2).astype(int)
)
risk = np.clip(risk, 0, 1)
attrition = np.random.binomial(1, risk)

df = pd.DataFrame({
    "Age": age,
    "MonthlyIncome": monthly_income,
    "YearsAtCompany": years_at_company,
    "JobSatisfaction": job_satisfaction,
    "DistanceFromHome": distance_from_home,
    "OverTime": overtime,
    "WorkLifeBalance": work_life_balance,
    "NumPromotions": num_promotions,
    "Attrition": np.where(attrition == 1, "Yes", "No"),
})

df.to_csv("data/employee_data.csv", index=False)
print(f"Generated {len(df)} employee records -> data/employee_data.csv")
print(f"Attrition rate: {(df['Attrition'] == 'Yes').mean():.1%}")
