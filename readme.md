### App flow

1. User opens the app

2. User uploads dataset (dropdown / upload box)

3. App shows raw data insights (basic EDA + visuals)

4. App performs data processing (cleaning, outliers, transforms)

5. App shows post-processed analysis & visuals

6. User selects:

    -> target column

    ->prediction type

    ->model

7. App trains, evaluates, and shows results






1. Basic info

shape

column names

data types

2. Descriptive statistics

describe() for numeric columns only

3. Missing values

count missing per column

apply your chosen strategy (drop / fill)

4. Outlier analysis

numeric columns only

IQR method

show before/after counts

5. EDA plots

histogram → numeric

bar plot → categorical

box plot → outliers

6. Correlation

correlation matrix

heatmap (numeric only)






Step 2: Decide action (rules)

≤ 5% missing → impute

5–30% missing → impute only if important

> 30% missing → drop column




# 📊 DAML – Data Analysis & Machine Learning App

DAML is an interactive Streamlit-based application that allows users to upload datasets, perform complete data analysis, and build machine learning models for regression and classification without writing code.

---

## 🚀 Features

### Dataset Upload
- Upload CSV or Excel files
- Automatic dataset preview
- Displays shape and column details

### Data Analysis
- Descriptive statistics
- Intelligent missing value handling (imputation & column removal)
- Outlier handling using IQR (clipping)
- Exploratory Data Analysis (EDA)
  - Histograms
  - Bar plots
  - Box plots
- Correlation heatmap
- Cleaned dataset stored for modeling

### Prediction Module

#### Regression
- Models:
  - Linear Regression
  - Ridge
  - Lasso
  - ElasticNet
  - Decision Tree
  - Random Forest
- Metrics:
  - R² Score
  - Mean Squared Error
- Visualizations:
  - Actual vs Predicted plot
- Download predictions as CSV

#### Classification
- Models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Metrics & Outputs:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - ROC Curve (binary classification)
  - Precision–Recall Curve
  - Model comparison table
- Download predictions as CSV

---

## 🧱 Project Structure

myprojects/
│
├── main.py
├── app.py
├── data_analysis.py
├── prediction.py
├── style.css


---

## ▶️ How to Run

### 1. Install dependencies


pip install streamlit pandas numpy scikit-learn matplotlib seaborn


### 2. Run the application


streamlit run main.py


---

## 🎥 Demo Flow

1. Upload dataset (CSV / Excel)
2. View dataset preview
3. Perform data analysis (EDA, cleaning, correlation)
4. Choose prediction type (Regression / Classification)
5. Select model and target column
6. View metrics, visualizations, and predictions
7. Download prediction results

---

## 🎯 Use Cases
- Machine learning learning projects
- Data analysis demonstrations
- Academic mini / major projects
- Interview portfolio showcase

---

## 🔮 Future Enhancements
- Auto-detection of prediction type
- Hyperparameter tuning
- Model saving and loading
- Cloud deployment