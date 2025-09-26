# 🩺 ML-Diabetes-Prediction-App

An end-to-end **Machine Learning project** that predicts the likelihood of diabetes in patients using diagnostic health parameters.  
The project includes:
- Data preprocessing (handling missing values, feature scaling)
- Class balancing with **SMOTE**
- Hyperparameter tuning with **GridSearchCV**
- Optimized **SVM model**
- A user-friendly **Streamlit web application**

---

## 📊 Dataset
The dataset used is the **PIMA Indians Diabetes Dataset** (`diabetes.csv`) with **768 records** and 9 attributes:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0 → Non-Diabetic, 1 → Diabetic)

---

## 📈 Model Performance

Accuracy: 74%

F1-Score: 0.65

ROC-AUC: 0.83

Balanced classes with SMOTE:
Before: Non-Diabetic=400, Diabetic=214

After: Non-Diabetic=400, Diabetic=400


## ⚙️ Features
1. **Data Preprocessing**
   - Replaced invalid zero values with `NaN` and imputed using column medians.
   - Standardized features with `StandardScaler`.

2. **Balancing**
   - Applied **SMOTE** (Synthetic Minority Oversampling Technique) to fix class imbalance.

3. **Model Training**
   - Trained **Support Vector Machine (SVM)** with hyperparameter tuning using `GridSearchCV`.
   - Best Parameters:  
     ```
     C = 100
     gamma = 0.01
     kernel = rbf
     ```
   - Achieved **F1-Score: 0.65** and **ROC-AUC: 0.83** on test data.

4. **Streamlit App**
   - Interactive web app to enter patient details.
   - Predicts **Diabetic / Not Diabetic** in real-time.
   - Shows model test performance metrics.

---

## 🖥️ Streamlit App UI

### 🔹 Non Diabetic
<img width="907" height="1009" alt="d1" src="https://github.com/user-attachments/assets/08f53916-4da7-4ebe-9eb4-dec1ea209a94" />

### 🔹 Diabetic
<img width="866" height="970" alt="d2" src="https://github.com/user-attachments/assets/5b71957b-3642-401c-9926-b25986c99e9d" />

---

## 🚀 How to Run Locally

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/ML-Diabetes-Prediction-App.git
cd ML-Diabetes-Prediction-App
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows
```

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn streamlit
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```

