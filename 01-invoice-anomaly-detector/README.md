# Invoice Anomaly Detector

Automatically detects anomalies in invoice data — combining 
rule-based checks with Machine Learning (IsolationForest).

---

## The Problem

Finance teams manually review hundreds of invoices every month.
Duplicate payments, policy violations and statistical outliers
often go undetected — wasting time and money.

---

## What This Tool Does

Runs 4 detection layers automatically:

| Module | Method | Risk Level |
|--------|--------|------------|
| Duplicate Invoice IDs | Rule-based | HIGH |
| Weekend Bookings | Rule-based | MEDIUM |
| Amount Outliers | IsolationForest (ML) | HIGH |
| Round Number Bias | Rule-based | LOW |

Outputs a formatted Excel report and a dashboard chart —
ready to share with a finance or audit team.

---

## Results on Sample Data

- 530 invoices analyzed
- 51 anomalies detected (9.6%)
- Runtime: under 2 minutes

---

## Quick Start

pip install pandas scikit-learn openpyxl matplotlib numpy

python invoice_detector.py

---

## Output

- output/anomaly_report.xlsx — Excel report with risk levels
- output/dashboard.png — chart for presentations
- data/invoices_sample.csv — sample data (auto-generated)

---

## Using Your Own Data

Replace the data generation step with:

df = pd.read_csv("your_data.csv", parse_dates=["date"])

Required columns:
invoice_id, date, vendor, category, amount, cost_center, approved_by

---

## Tech Stack

pandas · scikit-learn · openpyxl · matplotlib · numpy

---

Feedback welcome via Issues.