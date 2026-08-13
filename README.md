\# Customer Churn Prediction API



Predicts whether a telecom customer is likely to churn, based on account and service details. Built end-to-end: from raw data exploration to a deployed, live API.



🔗 \*\*Live API:\*\* https://churn-prediction-api-wox2.onrender.com/docs



\## Problem

Customer churn is expensive to fix after the fact — predicting it in advance lets a business act early with retention offers. This project uses the IBM Telco Customer Churn dataset (\~7,000 customers) to build a model that flags at-risk customers.



\## Key Findings from EDA

\- \~26.5% of customers churned — a meaningfully imbalanced problem, so accuracy alone is a misleading metric

\- Month-to-month contracts show far higher churn than one/two-year contracts

\- Churn is heavily concentrated in customers with low tenure (early months)



\## Approach

1\. Cleaned and encoded raw customer data (handled a hidden data quality issue in `TotalCharges`)

2\. Trained a baseline Logistic Regression model

3\. Trained XGBoost — interestingly, the \*untuned\* version underperformed the baseline

4\. Tuned XGBoost via GridSearchCV, which then outperformed both



| Model | ROC-AUC | Recall (churners) |

|---|---|---|

| Logistic Regression | 0.84 | 0.79 |

| XGBoost (default) | 0.82 | 0.66 |

| \*\*XGBoost (tuned)\*\* | \*\*0.843\*\* | \*\*0.81\*\* |



Recall was prioritized over precision — in a churn use case, missing an at-risk customer is costlier than a false alarm.



\## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Docker, deployed on Render



\## Run It Locally

\\`\\`\\`

git clone https://github.com/nyasha0434/churn-prediction.git

cd churn-prediction

pip install -r requirements.txt

uvicorn app:app --reload

\\`\\`\\`

Then visit `http://127.0.0.1:8000/docs`



\## What I'd Improve With More Time

\- Add automated tests

\- Try SHAP-based feature selection to simplify the model

\- Add a simple frontend instead of relying on the `/docs` page

