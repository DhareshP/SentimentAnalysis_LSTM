# SentimentAnalysis_LSTM

# Sentiment Analysis using LSTM

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

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow


