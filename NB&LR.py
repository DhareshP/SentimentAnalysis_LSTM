import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

file_path = "D:\\PyFiles\\imdb.csv"  
data = pd.read_csv(file_path)


print("Columns in dataset:", data.columns)
print(data.head())

data = data[['reviews', 'sentiment']]
data.columns = ['review', 'sentiment']

label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

X = data['review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Naive Bayes
nb_model = MultinomialNB()

# Logistic Regression
lr_model = LogisticRegression(max_iter=200)

nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)

lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)

def plot_confusion_matrix(y_test, y_pred, model_name):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

def plot_roc_curve(y_test, y_pred_proba, model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'{model_name} ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

nb_pred_proba = nb_model.predict_proba(X_test_tfidf)[:, 1]  
plot_confusion_matrix(y_test, nb_pred, 'Naive Bayes')
plot_roc_curve(y_test, nb_pred_proba, 'Naive Bayes')

lr_pred_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]  
plot_confusion_matrix(y_test, lr_pred, 'Logistic Regression')
plot_roc_curve(y_test, lr_pred_proba, 'Logistic Regression')

print("Naive Bayes Classification Report:")
print(classification_report(y_test, nb_pred, target_names=label_encoder.classes_))

print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_pred, target_names=label_encoder.classes_))
