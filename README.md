# 📊 Data Preprocessing

## 🚀 Introduction
Data preprocessing is an essential step in the data science and machine learning pipeline. Raw data is often noisy, incomplete, or inconsistent, making it unsuitable for direct analysis. Preprocessing ensures the data is clean, structured, and ready for modeling — ultimately improving accuracy, efficiency, and reliability.

---

## ⚙️ Common Data Preprocessing Techniques
1. **Data Cleaning**
   - Handling missing values
   - Removing duplicates
   - Outlier detection
   - Correcting inconsistencies

2. **Data Transformation**
   - Normalization (Min-Max scaling)
   - Standardization (Z-score scaling)
   - Log / Box-Cox transformations

3. **Feature Engineering**
   - Feature extraction & selection
   - Polynomial features
   - Dimensionality reduction (PCA, LDA)

4. **Encoding Categorical Data**
   - Label encoding
   - One-hot encoding
   - Ordinal encoding
   - Target encoding

5. **Data Integration & Reduction**
   - Merging multiple sources
   - Reducing data volume without losing info

6. **Discretization & Binning**
   - Converting continuous variables into intervals

7. **Data Splitting**
   - Train-test split
   - Cross-validation

---

## 🐍 Example Python Code

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    "Age": [25, 30, np.nan, 22, 28],
    "Salary": [50000, 60000, 52000, 58000, np.nan],
    "Department": ["HR", "IT", "Finance", "IT", "HR"]
}

df = pd.DataFrame(data)
print("🔹 Raw Data:\n", df)

# 1. Handling Missing Values
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Salary"].fillna(df["Salary"].median(), inplace=True)

# 2. Encoding Categorical Data
label_encoder = LabelEncoder()
df["Department_Label"] = label_encoder.fit_transform(df["Department"])

one_hot = pd.get_dummies(df["Department"], prefix="Dept")
df = pd.concat([df, one_hot], axis=1)

# 3. Normalization (Min-Max Scaling)
scaler = MinMaxScaler()
df["Age_Scaled"] = scaler.fit_transform(df[["Age"]])

# 4. Standardization (Z-score)
std_scaler = StandardScaler()
df["Salary_Standardized"] = std_scaler.fit_transform(df[["Salary"]])

# 5. Dimensionality Reduction (PCA Example)
pca = PCA(n_components=1)
df["Salary_PCA"] = pca.fit_transform(df[["Salary"]])

# 6. Train-Test Split
train, test = train_test_split(df, test_size=0.2, random_state=42)

print("\n✅ Preprocessed Data:\n", df)
print("\n📂 Train Split:\n", train)
print("\n📂 Test Split:\n", test)
