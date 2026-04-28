# 🏠 House Price Prediction using Elastic Net Regression

> A machine learning project focused on maximizing regression performance through advanced feature engineering, where **Elastic Net outperformed more complex ensemble models**.

---

## 🎯 Project Goal

The primary objective of this project is not just to predict house prices, but to **achieve the best possible regression performance** through:

- Careful **feature engineering**
- Systematic **model selection**
- Effective use of **regularization techniques**

Unlike typical approaches that rely heavily on complex ensemble models, this project explores whether a well-prepared dataset can allow simpler models to perform competitively—or even better.

The final results show that **Elastic Net Regression achieved superior performance**, highlighting the impact of strong feature engineering and proper handling of multicollinearity.

## 📊 Dataset Overview

- **Source:** Kaggle Housing Dataset *(https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)*
- **Number of Samples:** *[insert number]*
- **Number of Features:** *[insert number]*
- **Target Variable:** `SalePrice`

### 🧾 Data Characteristics

- The dataset contains a mix of **numerical and categorical features**
- Presence of **missing values** in several columns
- The target variable (`SalePrice`) is **right-skewed**, requiring transformation
- Some features show **strong multicollinearity**, making regularization important

---

## 🔍 Exploratory Data Analysis (EDA)

### 📈 Target Variable Distribution

Understanding the distribution of the target variable is critical for regression performance.

#### Before Transformation
![Target Distribution Before](images/target_distribution_before.png)

- The distribution is **heavily right-skewed**
- This can negatively impact model performance, especially for linear models

#### After Log Transformation
![Target Distribution After](images/target_distribution_after.png)

- Applying a **log transformation** results in a more **normal distribution**
- This improves model stability and prediction accuracy

---

### 🔥 Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)

- Strong correlations observed between:
  - `OverallQual` and `SalePrice`
  - `GrLivArea` and `SalePrice`
- Presence of **multicollinearity** among features
- Justifies the use of **Elastic Net regularization**, which handles correlated predictors effectively

---

### 📉 Feature Relationships

#### Example: Ground Living Area vs Sale Price
![GrLivArea vs SalePrice](images/grlivarea_vs_price.png)

- Clear **positive relationship** between living area and price
- A few **extreme outliers** detected and removed during preprocessing

---

### 🧠 Key EDA Takeaways

- Log transformation of the target significantly improves distribution
- Several features have strong predictive power
- Multicollinearity is present → supports use of **regularized linear models**
- Outlier handling is necessary for robust performance