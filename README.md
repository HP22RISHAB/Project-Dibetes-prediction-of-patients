# 🧠 Diabetes Prediction using Machine Learning

This project builds a machine learning model to predict whether a patient has diabetes based on diagnostic health data. It uses the **Pima Indians Diabetes dataset** and applies a **Random Forest Classifier** for classification.

---

## 📊 Dataset

* Source: Pima Indians Diabetes Dataset
* Features include:

  * Pregnancies
  * Glucose
  * Blood Pressure
  * Skin Thickness
  * Insulin
  * BMI
  * Diabetes Pedigree Function
  * Age
* Target:

  * `Outcome` (0 = No Diabetes, 1 = Diabetes)

---

## ⚙️ Tech Stack

* Python
* Pandas & NumPy (data handling)
* Matplotlib & Seaborn (visualization)
* Scikit-learn (ML model + preprocessing)

---

## 🚀 Workflow

1. **Data Loading**

   * Dataset loaded directly from a public URL

2. **Exploratory Data Analysis**

   * Preview dataset
   * Check for missing values
   * Correlation heatmap visualization

3. **Data Preprocessing**

   * Feature/target split
   * Train-test split (80/20)
   * Feature scaling using `StandardScaler`

4. **Model Training**

   * Random Forest Classifier

5. **Evaluation**

   * Accuracy Score
   * Confusion Matrix
   * Classification Report

6. **Feature Importance**

   * Visualized using bar chart

---

## 📈 Results

* Model performance is evaluated using:

  * Accuracy
  * Precision / Recall / F1-score
* Feature importance highlights the most influential medical indicators for prediction

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/diabetes-prediction.git

# Navigate into the project
cd diabetes-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the script
python "Project Dibetes prediction of patients.py"
```

---

## 🧩 Future Improvements

* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
* Try advanced models (XGBoost, LightGBM)
* Handle zero/invalid values more robustly
* Deploy as a web app (Flask / FastAPI)
* Add cross-validation

---

## 📌 Key Insight

Random Forest works well here because:

* It handles non-linear relationships
* It is robust to noise
* It gives feature importance (useful for interpretation)

---

## 🧑‍💻 Author

* Your Name

---

## 📜 License

This project is open-source and available under the MIT License.
