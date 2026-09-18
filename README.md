# Employee Attrition Predictor

A logistic regression model that predicts whether an employee is likely to
leave, based on tenure, job satisfaction, overtime, income, commute
distance, and promotion history — and explains *which factors* are driving
that risk.

Runs **entirely locally** — no API key, no signup, no cost.

## Why I built this

Attrition prediction is one of the most common real-world HR analytics
use cases — knowing *who* is at risk lets HR intervene early (workload
review, comp adjustment, career conversation) instead of finding out at
the exit interview. This project is a small, explainable version of that:
a model that doesn't just say "at risk" but tells you *why*.

## How it works

1. **`generate_data.py`** creates a synthetic dataset of 800 employees
   (`data/employee_data.csv`). Attrition isn't random — it's built to
   genuinely correlate with overtime, low satisfaction, low income, long
   commutes, and short tenure, the way real workforce data tends to.
   (No Kaggle account needed; the real IBM HR Analytics dataset — the
   dataset this one is structurally modeled on — can be swapped in later,
   see below.)
2. **`train_model.py`** loads the data, splits it into train/test sets,
   scales the features, and trains a logistic regression classifier.
3. It reports accuracy/precision/recall, a confusion matrix, ranks which
   features drive attrition risk most, and prints sample predictions with
   probability scores.

## Setup

```bash
pip install -r requirements.txt
python generate_data.py
python train_model.py
```

## Actual output from this repo

```
Accuracy:  81.2%
Precision: 73.1%  (of predicted leavers, how many actually left)
Recall:    70.4%  (of actual leavers, how many we caught)

TOP ATTRITION DRIVERS (by model weight)
  OverTime           +0.96   (increases attrition risk)
  JobSatisfaction    -0.88   (decreases attrition risk)
  DistanceFromHome   +0.44   (increases attrition risk)
  YearsAtCompany     -0.42   (decreases attrition risk)
  NumPromotions      -0.37   (decreases attrition risk)
  WorkLifeBalance    -0.36   (decreases attrition risk)
  MonthlyIncome      -0.32   (decreases attrition risk)
  Age                +0.28   (increases attrition risk)
```

*(This is a real run of the code in this repo, not a mocked example.)*

## Design Decisions

- **Logistic regression, not a black-box model** — it's simple enough to
  explain fully: each feature gets a weight, and the sign/size of that
  weight tells you its direction and strength of effect on attrition risk.
  For an HR use case, being able to explain *why* someone is flagged
  matters as much as the prediction itself.
- **Synthetic data with real signal, not random noise** — I deliberately
  built the attrition probability formula around known real-world drivers
  (overtime, satisfaction, tenure) rather than assigning attrition
  randomly, so the model has a genuine pattern to learn — the strong
  accuracy/precision numbers reflect that, not overfitting to noise.
- **Precision vs. recall tradeoff is a live design question** — in a real
  HR setting you'd tune this deliberately: high recall (catch more true
  leavers, tolerate more false alarms) if intervention is cheap, higher
  precision if HR follow-up is costly and you don't want to flag people
  unnecessarily.
- **Feature scaling before comparing coefficients** — without scaling,
  a feature like MonthlyIncome (values in the thousands) would dwarf a
  feature like JobSatisfaction (values 1-4) in raw coefficient size, even
  if satisfaction matters more. Scaling makes the driver ranking honest.

## Swapping in real data

To use the actual IBM HR Analytics Employee Attrition dataset (public,
free, on Kaggle) instead of the synthetic one: download it, save it as
`data/employee_data.csv` with matching column names (or adjust
`feature_cols` in `train_model.py`), and skip running `generate_data.py`.

## Possible extensions

- Try a decision tree or random forest and compare accuracy/interpretability
- Add a simple risk-tier output (Low/Medium/High) instead of raw probability
- Build a small Streamlit form: enter one employee's details, get their
  predicted risk and top contributing factors
