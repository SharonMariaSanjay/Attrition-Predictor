"""
Employee Attrition Predictor
------------------------------
Trains a logistic regression model to predict whether an employee is likely
to leave, based on tenure, satisfaction, overtime, income, commute distance,
and promotion history. Runs entirely locally — no API key, no cost.

Usage:
    python generate_data.py   # creates data/employee_data.csv (run once)
    python train_model.py     # trains + evaluates the model
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

DATA_PATH = "data/employee_data.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Encode categorical fields
    df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    feature_cols = [
        "Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction",
        "DistanceFromHome", "OverTime", "WorkLifeBalance", "NumPromotions",
    ]
    X = df[feature_cols]
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features so coefficients are comparable to each other
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("=" * 60)
    print("MODEL PERFORMANCE (on held-out test set)")
    print("=" * 60)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.1%}")
    print(f"Precision: {precision_score(y_test, y_pred):.1%}  "
          f"(of predicted leavers, how many actually left)")
    print(f"Recall:    {recall_score(y_test, y_pred):.1%}  "
          f"(of actual leavers, how many we caught)")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion matrix:")
    print(f"                 Predicted Stay  Predicted Leave")
    print(f"Actual Stay      {cm[0][0]:>13}  {cm[0][1]:>15}")
    print(f"Actual Leave     {cm[1][0]:>13}  {cm[1][1]:>15}")

    # Feature importance: coefficient magnitude (since features are scaled,
    # coefficients are directly comparable)
    print("\n" + "=" * 60)
    print("TOP ATTRITION DRIVERS (by model weight)")
    print("=" * 60)
    importance = pd.Series(model.coef_[0], index=feature_cols).sort_values(
        key=abs, ascending=False
    )
    for feature, coef in importance.items():
        direction = "increases" if coef > 0 else "decreases"
        print(f"  {feature:<18} {coef:+.2f}   ({direction} attrition risk)")

    # Show a few example predictions with probability, so it's demoable
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS (first 5 test employees)")
    print("=" * 60)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    sample = X_test.copy()
    sample["ActualAttrition"] = y_test.map({1: "Yes", 0: "No"}).values
    sample["PredictedRisk"] = [f"{p:.0%}" for p in probs]
    print(sample.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
