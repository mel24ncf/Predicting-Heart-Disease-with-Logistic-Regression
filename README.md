# Predicting Heart Disease with Logistic Regression - Framingham Heart Study

## 1. Project Overview
In this project I build a **logistic regression model** to predict whether a patient is at risk of developing **coronary heart disease (CHD) within the next 10 years**. The dataset comes from the **Framingham Heart Study**, which tracks various health indicators and lifestyle factors.

Beyond standard model evaluation, **threshold analysis** was conducted to optimize the decision threshold, improving the balance between **precision and recall** rather than relying on the default 0.50 threshold. This ensures the model is fine-tuned for real-world application in heart disease risk assessment.

## 2. Model & Methodology
- **Logistic Regression:** Chosen for interpretability and its effectiveness in binary classification tasks.
- **Data Preprocessing:** Standardization of numerical features and handling of missing values with imputation. Pandas Profiling used for Exploratory Data Analysis.
- **Class Imbalance Handling:** Undersampling was applied to the No CHD class to address the class imbalance.
- **Feature Importance:** Feature importance was measured.
- **Threshold Optimization:** Conducted an F1-score-based threshold analysis to improve classification performance.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, and ROC-AUC were analyzed to assess model performance.

## 3. Features
| Feature Name     | Description |
|-----------------|-------------|
| **Male**        | Binary indicator for male patient (1 = Male, 0 = Female) |
| **Age**         | Age of the patient in years |
| **Education**   | Categorical variable for education level (1-4, ranging from no formal education to college degree) |
| **currentSmoker** | Binary indicator for current smoker status |
| **cigsPerDay**  | Number of cigarettes smoked per day |
| **BPMeds**      | Binary indicator for use of anti-hypertensive medication |
| **prevalentStroke** | Binary indicator for history of stroke |
| **prevalentHyp** | Binary indicator for hypertension |
| **Diabetes**    | Binary indicator for diabetes status |
| **totChol**     | Total cholesterol level (mg/dL) |
| **sysBP**       | Systolic blood pressure (mmHg) |
| **diaBP**       | Diastolic blood pressure (mmHg) |
| **BMI**         | Body Mass Index (kg/m²) |
| **heartRate**   | Heart rate (bpm) |
| **Glucose**     | Blood glucose level (mg/dL) |
| **TenYearCHD**  | Target variable: Whether CHD occurred within 10 years (1 = Yes, 0 = No) |

## 4. Files & Scripts
| File Name       | Description |
|----------------|-------------|
| **framingham.csv** | Dataset containing patient data |
| **chd_explore.py** | Generates an HTML report using Pandas Profiling for exploratory data analysis |
| **chd_train.py** | Trains the logistic regression model, performs preprocessing, prints accuracy metric, and saves the trained model as a pickle file |
| **chd_test.py** | Evaluates the model, conducts threshold analysis, and plots performance metrics (ROC curve, precision-recall tradeoff, confusion matrix) |
| **util.py** | Holds reusable functions for generating model performance visualizations |

## 5. Model Performance
### **Default Threshold (0.50):**
- **Accuracy:** 72%
- **Precision:** 74%
- **Recall:** 68%
- **F1-score:** 71%

![ConfusionMatrixdef](Model%20Performance/ConfusionMatrix_default.png)

### **Optimized Threshold (0.43):**
- **Accuracy:** 71%
- **Precision:** 66%
- **Recall:** 85%
- **F1-score:** 75%

![ConfusionMatrixopt](Model%20Performance/ConfusionMatrix_optimized.png)

![Thresh](Model%20Performance/ThresholdAnalysis.png)

### AUC (.77)

![ROCAUC](Model%20Performance/ROC_Curve.png)

### **Insights:**
- Age, Mean Arterial Pressure (MAP), and sex (male) were identified as the top three most important features for predicting heart disease.
- The optimized threshold improves the F1-score and results in a significant increase in recall, detecting more heart disease cases without sacrificing too much on precision.
- The ROC curve analysis shows that the model is providing predictive value significantly above random guessing (random classifier), with an AUC of 0.77, suggesting good overall model performance.

![FeatureImp](Model%20Performance/FeatureImportance.png)

## 6. Future Work
- Experiment with other models (e.g., Random Forest, Gradient Boosting) for improved predictive performance.
- Implement SMOTE or other techniques to balance the dataset without reducing the negative class.
- Incorporate additional clinical data or external risk factors to enhance predictions.

---
This project demonstrates a structured approach to **predictive modeling in healthcare** by balancing interpretability, performance, and real-world applicability. ⚕️
