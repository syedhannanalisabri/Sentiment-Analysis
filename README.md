# Twitter Sentiment Analysis App

## Overview
This is a Streamlit web application for predicting the sentiment (positive, negative, or neutral) of tweets using a Long Short-Term Memory (LSTM) neural network. The model was trained on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle, which contains 1.6 million tweets labeled for sentiment. The app allows users to input a tweet (up to 280 characters) and receive a sentiment prediction with a confidence score.

The project showcases skills in natural language processing (NLP), deep learning with TensorFlow, and web app development with Streamlit. It is deployed on Streamlit Community Cloud and can be accessed [here](https://sentimentanalysisbyhannan.streamlit.app) 
## Features
- **Sentiment Prediction**: Predicts whether a tweet is positive, negative, or neutral using an LSTM model.
- **User-Friendly Interface**: Built with Streamlit, featuring a clean UI with emoji feedback and confidence scores.
- **Real-Time Analysis**: Processes user-input tweets instantly (max 280 characters).
- **Error Handling**: Manages empty inputs and missing model/tokenizer files gracefully.
- **Deployment Ready**: Hosted on Streamlit Community Cloud with automatic updates via GitHub.

## Dataset
The model was trained on the [Sentiment140 dataset]([https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment

## Model Architecture
- **Embedding Layer**: Converts tokenized tweets into 128-dimensional vectors (5,000-word vocabulary).
- **LSTM Layer**: 128 units to capture sequential patterns in tweet text.
- **Dense Layer**: Outputs probabilities for three classes (negative, neutral, positive) using softmax activation.
- **Training**: Trained for 5 epochs with a batch size of 32, using the Adam optimizer and categorical cross-entropy loss.

## Installation
To run the app locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-app.git
   cd sentiment-analysis-app
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   ```
   streamlit
   tensorflow
   pandas
   numpy
   ```

3. **Download Model and Tokenizer**:
   Ensure the following files are in the repository root:
   - `sentiment_model.h5`: Pre-trained LSTM model.
   - `tokenizer.pickle`: Saved tokenizer for text preprocessing.
   These files are included in the repository (model file tracked with Git LFS if >100MB).

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```
   The app will open in your browser at `http://localhost:8501`.

## Deployment
The app is deployed on Streamlit Community Cloud, linked to this GitHub repository. To deploy your own instance:
1. Sign up at [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
2. Create a new app, selecting this repository and `app.py` as the main file.
3. Ensure `sentiment_model.h5` and `tokenizer.pickle` are in the repository root.
4. Deploy and share the generated app URL.

## Usage
1. Open the app (locally or via the deployed URL).
2. Enter a tweet in the text area (max 280 characters).
3. Click **"Analyze Sentiment"** to view the predicted sentiment and confidence score.
4. Results include an emoji (😊👍 for positive, 😔👎 for negative, 😐 for neutral) and the input tweet.

## Example
**Input Tweet**: "I love this movie, it's amazing!"  
**Output**: Predicted Sentiment: Positive 😊👍 (Confidence: 0.9995)

## Project Structure
```
sentiment-analysis-app/
├── app.py               # Streamlit app code
├── sentiment_model.h5   # Pre-trained LSTM model
├── tokenizer.pickle     # Saved tokenizer
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Future Improvements
- Add support for batch tweet analysis.
- Incorporate advanced preprocessing (e.g., emoji handling, hashtag parsing).
- Optimize the model for faster inference on Streamlit Community Cloud.
- Add visualization of model performance metrics (e.g., accuracy, loss).

## Credits
Developed by Syed Hannan Ali Sabri. The project uses the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle, licensed under the [Creative Commons Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
