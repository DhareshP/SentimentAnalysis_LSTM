# SentimentAnalysis_LSTM

This project implements a sentiment analysis model using Long Short-Term Memory (LSTM) networks on the IMDB movie review dataset. The model is trained to classify reviews as either positive or negative. Additionally, the project includes various visualizations, such as a Confusion Matrix Heatmap, ROC Curve, and Training/Validation Accuracy and Loss plots, to evaluate the model's performance.

Project Overview
Key Steps:
Data Collection: We use the IMDB dataset, which contains movie reviews along with their associated sentiment labels (positive/negative).
Data Preprocessing:
Handling missing values and encoding sentiment labels.
Tokenization and padding of the movie reviews to prepare them for input into the LSTM model.
Model Development: A deep learning model based on LSTM is developed to capture the sequential dependencies in text data.
Model Training: The model is trained using the processed dataset and evaluated on a test set.
Evaluation Metrics:
Confusion Matrix Heatmap
Classification Report
ROC Curve
Training and Validation Loss/Accuracy plots
Prerequisites
Before running this project, ensure you have the following libraries installed:

numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
You can install these dependencies using pip:

bash
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
Dataset
The dataset used in this project is an IMDB movie reviews dataset. Make sure to have your dataset (imdb.csv) saved locally. It should contain the following columns:

reviews: The text reviews.
sentiment: The labels indicating whether the review is positive or negative.
Project Structure
imdb.csv: The dataset file containing movie reviews and sentiment labels.
sentiment_analysis.py: The main Python script to train the model and generate visualizations.
README.md: This file.
Model Architecture
The LSTM model is designed with the following layers:

Embedding Layer: Maps each word to a dense vector of fixed size.
SpatialDropout1D Layer: Applies dropout to the embeddings to prevent overfitting.
LSTM Layer: A 100-unit LSTM to capture the sequential nature of the text.
Dense Layer: A fully connected layer with a sigmoid activation function for binary classification.
How to Run
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/your-username/sentiment-analysis-lstm.git
Place your dataset file (imdb.csv) in the project directory.

Run the script:

bash
Copy code
python sentiment_analysis.py
After training, the script will display:

A Confusion Matrix Heatmap.
A ROC Curve to visualize the model's performance.
Training and validation accuracy and loss plots to assess overfitting or underfitting.
Visualization Examples
Confusion Matrix Heatmap: Provides insights into correct and incorrect predictions.
ROC Curve: Plots the trade-off between true positive and false positive rates.
Loss and Accuracy Curves: Track the model’s performance over epochs.
Results
The model's performance is evaluated using the following metrics:

Accuracy
Precision
Recall
F1-Score
ROC AUC Score
Future Improvements
Tuning the LSTM architecture (number of units, additional layers, etc.).
Trying other word embeddings such as GloVe or Word2Vec.
Experimenting with different tokenization techniques.
Implementing more advanced models like GRU or BERT for better performance.
License
This project is open-source and available under the MIT License.


