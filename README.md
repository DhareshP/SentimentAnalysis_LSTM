# SentimentAnalysis_LSTM

This project implements a sentiment analysis model using Long Short-Term Memory (LSTM) networks on the IMDB movie review dataset. The model is trained to classify reviews as either **positive** or **negative**. Additionally, the project includes various visualizations, such as a **Confusion Matrix Heatmap**, **ROC Curve**, and **Training/Validation Accuracy and Loss** plots, to evaluate the model's performance.

## Project Overview

### Key Steps:
1. **Data Collection**: We use the IMDB dataset, which contains movie reviews along with their associated sentiment labels (positive/negative).
2. **Data Preprocessing**:
   - Handling missing values and encoding sentiment labels.
   - Tokenization and padding of the movie reviews to prepare them for input into the LSTM model.
3. **Model Development**: A deep learning model based on LSTM is developed to capture the sequential dependencies in text data.
4. **Model Training**: The model is trained using the processed dataset and evaluated on a test set.
5. **Evaluation Metrics**: 
   - Confusion Matrix Heatmap
   - Classification Report
   - ROC Curve
   - Training and Validation Loss/Accuracy plots

## Prerequisites

Before running this project, ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`

You can install these dependencies using pip:

``bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

This project implements a sentiment analysis model using Long Short-Term Memory (LSTM) networks on the IMDB movie review dataset. The model is trained to classify reviews as either **positive** or **negative**. Various visualizations, such as a **Confusion Matrix Heatmap**, **ROC Curve**, and **Training/Validation Accuracy and Loss** plots, are included to evaluate the model's performance.

## Dataset

The dataset used in this project is an IMDB movie reviews dataset. Make sure to have your dataset (`imdb.csv`) saved locally. It should contain the following columns:

- **reviews**: The text reviews.
- **sentiment**: The labels indicating whether the review is positive or negative.

## Project Structure

- **`imdb.csv`**: The dataset file containing movie reviews and sentiment labels.
- **`sentiment_analysis.py`**: The main Python script to train the model and generate visualizations.
- **`README.md`**: This file, containing all the relevant information about the project.

## Model Architecture

The LSTM model is designed with the following layers:

1. **Embedding Layer**: Maps each word to a dense vector of fixed size.
2. **SpatialDropout1D Layer**: Applies dropout to the embeddings to prevent overfitting.
3. **LSTM Layer**: A 100-unit LSTM to capture the sequential nature of the text.
4. **Dense Layer**: A fully connected layer with a sigmoid activation function for binary classification.

## How to Run

### Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/sentiment-analysis-lstm.git

python lstm_sa.py


