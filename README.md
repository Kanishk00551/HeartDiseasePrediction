



# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Problem Statement

Can we predict the presence of heart disease in a patient given a set of clinical parameters?

This project applies supervised machine learning techniques to analyze clinical data and predict whether a patient is likely to have heart disease.

---

## 📂 Data Sources

* **UCI Machine Learning Repository (Cleveland dataset)**
  📎 [UCI Link](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

* **Kaggle Heart Disease Dataset**
  📎 [Kaggle Link](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset)

---

## 🎯 Project Objective

Develop a binary classification model to predict heart disease using various clinical features.
**Success Metric**: Achieve at least **95% accuracy** during the proof of concept phase.

---

## 🧠 Technologies & Skills Used

* **Programming Language**: Python
* **Libraries**:

  * `NumPy` for numerical computations
  * `Pandas` for data handling and manipulation
  * `Matplotlib` & `Seaborn` for data visualization
  * `Scikit-learn` for model building, preprocessing, and evaluation
* **Concepts**:

  * Exploratory Data Analysis (EDA)
  * Feature engineering and selection
  * Classification algorithms (Logistic Regression, Random Forest, etc.)
  * Cross-validation and performance evaluation (confusion matrix, accuracy, precision, recall, F1-score)

---

## 📊 Features Description

| Feature      | Description                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------- |
| **age**      | Age of the patient in years                                                                  |
| **sex**      | Gender (1 = male, 0 = female)                                                                |
| **cp**       | Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal, 3 = asymptomatic) |
| **trestbps** | Resting blood pressure (mm Hg)                                                               |
| **chol**     | Serum cholesterol in mg/dl                                                                   |
| **fbs**      | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)                                        |
| **restecg**  | Resting ECG results (0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy)     |
| **thalach**  | Maximum heart rate achieved                                                                  |
| **exang**    | Exercise-induced angina (1 = yes, 0 = no)                                                    |
| **oldpeak**  | ST depression induced by exercise compared to rest                                           |
| **slope**    | Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)             |
| **ca**       | Number of major vessels (0–3) visualized by fluoroscopy                                      |
| **thal**     | Thalium stress test result (1 = normal, 6 = fixed defect, 7 = reversible defect)             |
| **target**   | Presence of heart disease (1 = yes, 0 = no)                                                  |

---

## 📈 Workflow

1. **Load and Understand the Data**
2. **Data Cleaning and Preprocessing**

   * Handle missing values
   * Encode categorical variables
   * Feature scaling
3. **Exploratory Data Analysis (EDA)**

   * Visualize distributions, correlations, and patterns
4. **Model Building**

   * Try various models: Logistic Regression, K-Nearest Neighbors, Random Forest, etc.
   * Tune hyperparameters using GridSearchCV or RandomizedSearchCV
5. **Model Evaluation**

   * Evaluate using accuracy, precision, recall, F1-score, ROC-AUC
   * Aim for ≥ 95% accuracy
6. **Interpretation and Insights**

   * Identify key features that influence heart disease prediction
   * Provide model explanation using SHAP or feature importance

---

## ✅ Results

> *Model performance and evaluation metrics will be added here after experimentation.*

---

## 🚀 Future Scope

* Deploy as a web application using **Flask** or **Streamlit**
* Integrate with real-time health data via APIs or wearables
* Add model explainability using **SHAP** or **LIME**
* Test on other heart disease datasets for generalization

---

## 📚 References

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
* [Kaggle Dataset](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

---



