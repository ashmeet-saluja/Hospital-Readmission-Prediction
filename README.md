# Hospital-Readmission-Prediction
Here's a comprehensive `README.md` template for your GitHub repository that details the project on hospital readmission prediction:

---

# **Hospital Readmission Prediction**

Predicting hospital readmission within 30 days using electronic health record (EHR) data.

---

## **Project Overview**

This project leverages machine learning to predict patient readmission rates based on historical hospital data. The primary objective is to assist healthcare providers in identifying high-risk patients and reducing readmission rates.

---

## **Dataset**

- **Source**: [Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Description**: The dataset contains records of 101,766 patients admitted to 130 hospitals over 10 years. It includes demographic, diagnostic, and medication details.
- **Target Variable**: `readmitted` (`NO`, `>30`, `<30`)

---

## **Technologies Used**

- **Programming Languages**: Python
- **Libraries**: 
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
- **Model**: Random Forest Classifier

---

## **Features**

- **Age**: Patient age group (e.g., `[0-10)`)
- **Gender**: Patient gender
- **Time in Hospital**: Duration of stay
- **Medications**: Details about prescribed drugs
- **Diagnosis**: Primary, secondary, and tertiary diagnoses
- **Number of Procedures**: Number of medical procedures performed

---

## **Steps in Analysis**

1. **Data Cleaning**:
   - Removed irrelevant columns (`encounter_id`, `patient_nbr`, etc.).
   - Handled missing values (`?` replaced or rows dropped).
   - Encoded categorical variables.

2. **Exploratory Data Analysis (EDA)**:
   - Analyzed patterns in patient demographics and diagnoses.
   - Visualized correlations and readmission distributions.

3. **Model Training**:
   - Used Random Forest Classifier.
   - Optimized hyperparameters with GridSearchCV.

4. **Evaluation**:
   - Achieved ~56% accuracy on the test dataset.
   - Used classification reports to assess model performance.

5. **Feature Importance**:
   - Analyzed which features contributed most to predictions.

---

## **How to Run**

### Prerequisites:
1. Install Python 3.8 or higher.
2. Install required libraries using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hospital-readmission-prediction.git
   cd hospital-readmission-prediction
   ```
2. Run the Jupyter notebook or Python scripts to preprocess data, train the model, and evaluate results.

3. To predict using the trained model:
   ```python
   import joblib
   model = joblib.load('hospital_readmission_model.pkl')
   prediction = model.predict(new_data)
   ```

---

## **Project Files**

- `diabetic_data.csv`: The original dataset.
- `data_preprocessing.py`: Data cleaning and encoding script.
- `model_training.py`: Model training and evaluation script.
- `hospital_readmission_model.pkl`: Trained Random Forest model.
- `README.md`: Project documentation.

---

## **Results**

- **Accuracy**: ~56%
- **Insights**:
  - Patients with multiple diagnoses and longer hospital stays are more likely to be readmitted.
  - Specific age groups and medications also correlate with readmission rates.

---

## **Future Work**

- Improve model accuracy using advanced techniques like deep learning.
- Incorporate additional features for better predictions.
- Deploy the model as a web application for real-time predictions.

---

## **Contributing**

Contributions are welcome! Please fork the repository, make changes, and submit a pull request.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contact**

For any questions or feedback, contact:

- **Name**: Ashmeet Saluja
- **Email**: salujaashmeet179@example.com
- **GitHub**: https://github.com/ashmeet-saluja

---

This `README.md` is customizable to your preferences. Replace placeholders like `your-username` with actual details. Let me know if you need help with any additional sections!
