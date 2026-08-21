# ❤️ Heart Disease Prediction Using Machine Learning
**IBM SkillsBuild Project 2026 — Team Deadlock**  
*Government College of Engineering, Kannur*

---

## 📌 Project Overview
This project provides an end-to-end Machine Learning web application for predicting the risk of heart disease based on **13 clinical parameters**. It utilizes the **UCI Heart Disease Dataset (1025 patient records)** and a **Random Forest Classifier** achieving high predictive accuracy.

### 👥 Team Members (Team Deadlock)
- **Team Lead:** Aparna E P
- **Member:** Fathima Shadha
- **Member:** Devika T P

---

## 🗂️ Project Structure
```
IBM_PROJECT/
│
├── heart.csv               # UCI Heart Disease Dataset (1025 records, 14 columns)
├── model.py                # ML model training, evaluation & serialization script
├── heart_model.pkl         # Exported trained Random Forest Classifier
├── metrics.json            # Actual calculated evaluation metrics
├── app.py                  # Flask web application server (Render-ready)
├── Procfile                # Gunicorn process definition for Render
├── render.yaml             # Render deployment configuration
│
├── templates/
│   └── index.html          # Frontend interface template
│
├── static/
│   └── style.css           # Styling with responsive design & ECG animation
│
├── requirements.txt        # Python package dependencies (with gunicorn)
└── README.md               # Documentation & usage guide
```

---

## 🔬 13 Medical Parameters (Features)

| # | Feature | Code | Description / Allowed Values |
|---|---------|------|------------------------------|
| 1 | **Age** | `age` | Patient age in years |
| 2 | **Sex** | `sex` | `1` = Male, `0` = Female |
| 3 | **Chest Pain Type** | `cp` | `0`: Typical Angina, `1`: Atypical Angina, `2`: Non-anginal, `3`: Asymptomatic |
| 4 | **Resting Blood Pressure** | `trestbps` | In mm Hg (e.g. 120) |
| 5 | **Serum Cholesterol** | `chol` | In mg/dl (e.g. 210) |
| 6 | **Fasting Blood Sugar** | `fbs` | `1` = > 120 mg/dl (High), `0` = &le; 120 mg/dl (Normal) |
| 7 | **Resting ECG** | `restecg` | `0`: Normal, `1`: ST-T wave abnormality, `2`: LV hypertrophy |
| 8 | **Maximum Heart Rate** | `thalach` | Peak heart rate achieved in bpm (e.g. 165) |
| 9 | **Exercise Induced Angina** | `exang` | `1` = Yes, `0` = No |
| 10 | **ST Depression (Oldpeak)** | `oldpeak` | ST depression induced by exercise relative to rest (e.g. 0.5) |
| 11 | **Slope** | `slope` | `0`: Upsloping, `1`: Flat, `2`: Downsloping |
| 12 | **Major Vessels** | `ca` | Number of major vessels colored by fluoroscopy (`0` - `3`) |
| 13 | **Thalassemia** | `thal` | `1` = Normal, `2` = Fixed Defect, `3` = Reversible Defect |

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and Evaluate the ML Model
```bash
python model.py
```

**Actual Calculated Output:**
- **Accuracy:** `98.54%`
- **Confusion Matrix:**
  ```
  [[102   0]
   [  3 100]]
  ```
- **Precision / Recall / F1-Score:** `0.99` (Weighted Average)

### 3. Start the Flask Web Application
```bash
python app.py
```

### 4. Open in Your Browser
Navigate to `http://127.0.0.1:5000`

---

## ☁️ Deployment on Render

This project is fully configured for zero-configuration deployment on **Render**:

1. **Push your code to GitHub** (create a GitHub repository and push this folder).
2. **Go to [render.com](https://render.com/)** and click **New + > Web Service**.
3. **Connect your GitHub repository**.
4. Configure the settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Click **Deploy Web Service**!

Render will automatically bind to the assigned `$PORT` and serve the application live.

---

## 🧪 Sample Test Cases (Slide 13)

### Example 1: Low Risk Prediction 🟢
- **Inputs:** Age: `45`, Sex: `0` (Female), CP: `1`, trestbps: `120`, chol: `210`, fbs: `0`, restecg: `1`, thalach: `165`, exang: `0`, oldpeak: `0.5`, slope: `2`, ca: `0`, thal: `2`
- **Expected Result:** `Low Risk of Heart Disease` (🟢)

### Example 2: High Risk Prediction 🔴
- **Inputs:** Age: `62`, Sex: `1` (Male), CP: `3`, trestbps: `160`, chol: `290`, fbs: `1`, restecg: `0`, thalach: `105`, exang: `1`, oldpeak: `3.5`, slope: `0`, ca: `2`, thal: `3`
- **Expected Result:** `High Risk of Heart Disease` (🔴)
