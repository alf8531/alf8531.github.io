# Final Project Outline

## 1. Project Title
> **Strategic Framework for Predictive Modeling of Clinical Readmission Risk in Diabetic Populations**

## 2. Problem Statement or Research Question
> **Research Question:** Can a machine learning architecture, trained on a decade of clinical encounters, significantly outperform traditional risk assessment tools (like the LACE index) in forecasting 30-day hospital readmissions for diabetic patients?
>
> **The Problem:** Unplanned readmissions represent a failure in the healthcare continuum, costing the US healthcare system estimated $20 billion annually. Under the CMS Hospital Readmissions Reduction Program (HRRP), hospitals face severe financial penalties for excessive readmission rates. The goal is to shift from a reactive symptom-based model to a proactive data-driven ecosystem.

## 3. Description of the Project
> This project involves the end-to-end analysis of over **100,000 patient records** across 130 hospitals.The scope is specifically limited to inpatient encounters involving patients diagnosed with diabetes (length of stay 1–14 days).
>
> By analyzing demographic factors, clinical measurements, and historical utilization, the project aims to build a predictive framework to identify high-risk individuals *before* discharge. This allows for targeted interventions to bridge the gap between inpatient care and community recovery.

## 4. Data Source(s)
> **UCI Machine Learning Repository**
> - **Dataset:** Diabetes 130-US Hospitals for Years 1999–2008
> - **URL:** [https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
> - **Scale:** 101,766 unique patient encounters, 50+ attributes (Demographics, Encounter Dynamics, Diagnostic Profiles, Medication Context).

## 5. Tools & Technologies
> - **Core Stack:** Python 3.9+ (All analysis and modeling will be performed in a Python environment).
> - **Data Processing:** Pandas & NumPy (Used for data cleaning, aggregation, and vectorization instead of SQL).
> - **Visualization:** Matplotlib & Seaborn (For static statistical plots and correlation heatmaps).
> - **Machine Learning:** Scikit-learn (Baseline models and preprocessing), **XGBoost** (Gradient Boosting for high performance).
> - **Interpretability:** Built-in Feature Importance measures (visualizing which variables, like 'Number of Lab Procedures', drive the model's decisions).
> - **Dashboard/Deployment:** **Streamlit** (A Python-based framework to build an interactive web interface for the model directly from the code).

## 6. Planned Workflow / Methods
> 1.  **Data Sanitization (Pandas):** Use Python to filter out patients with `discharge_disposition_id` relating to hospice or expiration. Map the 700+ distinct ICD-9 diagnostic codes into 9 high-level clinical categories (e.g., Circulatory, Respiratory, Diabetes) to reduce noise.
> 2.  **Exploratory Analysis (EDA):** Use Seaborn/Matplotlib to visualize the "Utilization Gradient" (relationship between previous ER visits and readmission risk) and analyze the impact of A1C test results on patient outcomes.
> 3.  **Feature Engineering:** Handle the 11.2% class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic examples of readmitted patients in the training set.
> 4.  **Modeling:** Train and compare a Baseline Logistic Regression model against a Gradient Boosting model (**XGBoost**). We will use Stratified K-Fold Cross-Validation to ensure the model performs consistently across different data splits.
> 5.  **Model Interpretation:** Instead of "black box" predictions, we will extract **Feature Importance scores** directly from the XGBoost model. This will rank which variables (e.g., "Time in Hospital" vs. "Number of Lab Procedures") are the strongest predictors of readmission.

## 7. Expected Outcome or Deliverables
> - **Primary Metric:** Achieve a **Recall (Sensitivity) of >60%** for 30-day readmissions. We are prioritizing "Recall" because in healthcare, missing a high-risk patient (False Negative) is worse than falsely flagging a low-risk one.
> - **Deliverable 1:** A **Streamlit Web Application**. This will be a user-friendly interface running in the browser where a user can adjust patient inputs (e.g., Age, Number of Diagnosis) and see the predicted Readmission Probability in real-time.
> - **Deliverable 2:** A comprehensive **Jupyter Notebook** containing all code for data cleaning, EDA, model training, and evaluation metrics.
> - **Deliverable 3:** A **Strategic Insight Report** identifying the top 5 drivers of readmission based on the Feature Importance analysis (e.g., confirming if "Number of Inpatient Visits" is the leading indicator). history) and ROI analysis on cost savings.

## 8. Team Info
**Is this project:**
- [x] Individual
- [ ] Group



--[← Back to Home](./)
