# Supervised Learning Algorithms for Classification (Hand-Crafted Implementations)

[![Semester](https://img.shields.io/badge/Semester-Fall%202023-blue)]() [![Project](https://img.shields.io/badge/Project-Machine%20Learning%20Project%201-orange)]()

🚀 check out the [report](https://github.com/JenLungHsu/Supervised-Learning-Algorithms-for-Classification/blob/main/Supervised%20Learning%20Algorithms%20for%20Classification.pdf) for more detail.

## Project Overview
This project implements various **Supervised Learning Algorithms** for classification tasks. All algorithms are implemented manually without the use of machine learning libraries like `scikit-learn`. The models include:

- **Linear Classifier (Perceptron)**
- **K-Nearest Neighbors (K-NN)**
- **Naive Decision Tree Classifier**
- **Decision Tree with Pruning**

The study emphasizes understanding the internal mechanisms of these algorithms, comparing their performance through cross-validation and feature importance analysis.

## Dataset
- **Source:** [Car Insurance Claim Prediction - Kaggle](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification/data)
- The dataset contains information about policyholders and their associated risk factors, aiming to predict the likelihood of an insurance claim.

## Project Structure
```
├── cross_validation.py                                     # Cross-validation and evaluation
├── data_processing.py                                      # Data preprocessing and feature engineering
├── definition_model.py                                     # Model definitions (Perceptron, K-NN, Decision Tree)
├── feature_importance.py                                   # Feature importance analysis using SHAP values
├── Supervised Learning Algorithms for Classification.pdf   # Research paper with analysis
├── README.md                                               # Project documentation
```

## Methodology
1. **Data Preprocessing:**
   - Cleaned data and handled missing values.
   - Applied feature engineering techniques to enhance model performance.

2. **Model Implementation:**
   - Linear Classifier (Perceptron)
   - K-Nearest Neighbors (K-NN) with different distance metrics
   - Naive Decision Tree
   - Pruned Decision Tree

3. **Feature Importance Analysis:**
   - Computed SHAP values to understand feature impact.

4. **Cross-Validation:**
   - Used k-fold cross-validation (k=3, 5, 10) to assess model stability and generalization.

## Key Findings
- The **Pruned Decision Tree** achieved the best performance in terms of stability and accuracy.
- Cross-validation demonstrated the robustness of K-NN and Pruned Decision Tree models.
- Feature importance analysis showed that certain variables significantly impacted predictions.

## Usage
To execute the models, follow the steps:

```bash
# Data Preprocessing
python data_processing.py

# Model Training and Evaluation
python definition_model.py

# Feature Importance Analysis
python feature_importance.py

# Cross-Validation
python cross_validation.py
```

## Contact
- **Author:** Jen Lung Hsu
- **Email:** RE6121011@gs.ncku.edu.tw
- **Institute:** National Cheng Kung University, Institute of Data Science
