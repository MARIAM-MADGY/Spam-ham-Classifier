# 📌 SMS Spam Detection Project


🔹 Overview

This project aims to classify SMS messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) techniques and Machine Learning models.
The dataset contains a collection of SMS messages that are preprocessed, vectorized, and then used to train a classifier for spam detection.

🔹 Objectives

Preprocess raw SMS messages (tokenization, removing stopwords, lemmatization, etc.).

Convert text data into numerical representation using TF-IDF.

Train a classification model (Naive Bayes) on the processed data.

Evaluate the model using Precision, Recall, F1-score, and Accuracy.

🔹 Tools & Libraries

Python

pandas, numpy – data handling

nltk – text preprocessing (tokenization, stopwords, lemmatization)

scikit-learn – TF-IDF, Naive Bayes, train-test split, evaluation metrics

matplotlib, seaborn – visualization

🔹 Workflow

Data Loading & Cleaning

Removed unnecessary columns, handled duplicates, and renamed columns.

Exploratory Data Analysis (EDA)

Checked class distribution (Spam vs Ham).

Visualized imbalance in the dataset.

Text Preprocessing

Converted messages to lowercase.

Tokenized text into words.

Removed punctuation and stopwords.

Applied Lemmatization for better word representation.

Feature Extraction

Converted text into numerical features using TF-IDF Vectorizer.

Model Training

Trained a Multinomial Naive Bayes classifier.

Evaluation

Used Confusion Matrix and calculated:

Precision = 1.0

Recall = 0.74

F1-score = 0.85

Accuracy on Test set = 96%

Accuracy on Train set = 97%

🔹 Results & Insights

The model achieved high precision (1.0), meaning it rarely misclassified a message as Spam when it wasn’t.

Recall was slightly lower (0.74), meaning some Spam messages were not detected.

Overall, the model is robust and well-generalized with balanced performance.

🔹 Future Improvements

Try other models (e.g., Logistic Regression, SVM).

Use Word2Vec / embeddings for richer feature representation.

Apply techniques to handle class imbalance.
