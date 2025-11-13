# 🎬 IMDB Movie Review Sentiment Analysis

This project performs **Sentiment Analysis** on the **IMDB Movie Review dataset** to classify movie reviews as **Positive** or **Negative** using multiple machine learning algorithms.  
It aims to compare the performance of different models in understanding text sentiment through natural language processing (NLP) techniques.

---

## 🧠 Project Overview

Sentiment analysis is an important application of NLP that determines whether a given piece of text expresses a **positive** or **negative** emotion.  
In this project, IMDB movie reviews are preprocessed, vectorized using TF-IDF, and trained using various classification algorithms to predict sentiment polarity.

---

## ⚙️ Workflow

### 1. **Data Preprocessing**
- Removed HTML tags, punctuation, and stopwords  
- Tokenized and lemmatized text using **NLTK**  
- Transformed text into numerical features using **TF-IDF Vectorization**

### 2. **Model Training**
Implemented and trained the following classification algorithms:
- **Logistic Regression**
- **Multinomial Naive Bayes**
- **Linear Support Vector Classifier (SVC)**
- **Decision Tree Classifier**

### 3. **Model Optimization**
- **Cross-Validation (k-fold):** Ensured model stability and reduced bias-variance errors  
- **Hyperparameter Tuning:** Used **GridSearchCV** to find optimal parameters for each model  
- **Overfitting Checking:**  
  - Compared **training vs. validation accuracy** (difference within ±5% indicates no overfitting)  
  - Evaluated **ROC Curves** and **AUC Scores** for model discrimination ability  

### 4. **Model Evaluation**
Performance measured using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **ROC Curve & AUC**

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:----------|:-----------|:--------|:----------|
| Logistic Regression | *88.96%* | *88.99%* | *88.96%* | *88.96%* |
| Multinomial Naive Bayes | *86.16%* | *86.19%* | *86.16%* | *86.16%* |
| Linear SVC | *88.99%* | *89.02%* | *88.99%* | *88.99%* |
| Decision Tree | *75.22%* | *75.22%* | *75.22%* | *75.22%* |

> 🏆 **Linear SVC** and **Logistic Regression** achieved the highest accuracy on text classification tasks.

---

## 🛠️ Technologies Used

- **Python**
- **Scikit-learn**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **NLTK** (Natural Language Toolkit)

---

## 📁 Dataset

The dataset used is the **IMDB Movie Review Dataset**, which contains **50,000 reviews** (25,000 positive and 25,000 negative).  
You can download it from [Kaggle - IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

---

## 💡 Example Output
``` bash
Review: "The movie was fantastic! The performances were stellar."
Predicted Sentiment: Positive ✅
```

``` bash
Review: "The plot was dull and the characters were underdeveloped."
Predicted Sentiment: Negative ❌
```

---

## 🚧 Future Improvements
- Integrate deep learning models (e.g., LSTM, BERT)
- Build a Flask / Streamlit web app for real-time predictions
