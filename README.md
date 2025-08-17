# Sentiment-Analysis-using-NLP-
# Multi-Class Sentiment Analysis with BERT and PyTorch

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.0%2B-yellow.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

This project is an end-to-end Natural Language Processing (NLP) pipeline for multi-class sentiment analysis. It uses a pre-trained BERT model from Hugging Face, fine-tuned on the Google Play Store User Reviews dataset to classify app reviews into three categories: **Positive**, **Negative**, or **Neutral**.

The entire project is implemented in a Google Colab notebook, demonstrating the full workflow from data loading and cleaning to model training, evaluation, and inference.

## Key Features

-   **Multi-Class Classification:** Classifies text into Positive, Negative, and Neutral sentiments.
-   **State-of-the-Art Model:** Leverages `bert-base-uncased`, a powerful pre-trained Transformer model, fine-tuned for high accuracy.
-   **End-to-End Pipeline:** Includes all steps: data preprocessing, BERT tokenization, model training, performance evaluation, and a prediction function for new text.
-   **High Performance:** Achieved **~94% accuracy** on the validation set after just 3 epochs of training.
-   **PyTorch-Powered:** Built using PyTorch for model training and management, including the use of GPU acceleration.

## Technologies Used

-   **Python 3.8+**
-   **PyTorch:** For building and training the deep learning model.
-   **Hugging Face Transformers:** For loading the pre-trained BERT model and its tokenizer.
-   **Scikit-learn:** For splitting data and performance evaluation.
-   **Pandas:** For data manipulation and cleaning.
-   **Seaborn & Matplotlib:** For data visualization and plotting training performance.
-   **Google Colab:** As the development and training environment with GPU support.

## Project Structure

```
.
├── bert-sentiment-model/      # Directory for the saved fine-tuned model and tokenizer
├── googleplaystore_user_reviews.csv  # The dataset file (not included in repo)
└── Sentiment_Analysis_with_BERT.ipynb # The main Colab notebook with all the code
└── README.md                  # This file
```

## Setup and Usage

To run this project, you can follow these steps:

1.  **Clone the repository (optional):**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Open in Google Colab:**
    -   Go to [Google Colab](https://colab.research.google.com/).
    -   Click on `File > Upload notebook` and upload the `Sentiment_Analysis_with_BERT.ipynb` file.

3.  **Enable GPU:**
    -   In the Colab notebook, navigate to `Runtime > Change runtime type`.
    -   Select `GPU` from the "Hardware accelerator" dropdown menu.

4.  **Download the Dataset:**
    -   Download the dataset from Kaggle: [Google Play Store User Reviews](https://www.kaggle.com/datasets/lava18/google-play-store-apps).
    -   Upload the `googleplaystore_user_reviews.csv` file to your Colab session using the file-explorer pane on the left.

5.  **Run the Notebook:**
    -   Execute the cells in the notebook sequentially.
    -   The notebook will automatically install all required dependencies. The training process will take approximately 15-20 minutes on a standard Colab GPU.

## How It Works

The notebook is divided into the following key stages:
1.  **Data Loading and Cleaning:** The raw CSV is loaded, and rows with missing reviews or sentiments are dropped.
2.  **Preprocessing:** Sentiment labels ('Positive', 'Negative', 'Neutral') are mapped to numerical values (2, 1, 0).
3.  **BERT Tokenization:** The text is tokenized using the `BertTokenizer`, converting it into a format suitable for the model (Input IDs and Attention Masks).
4.  **Training:** The `BertForSequenceClassification` model is fine-tuned on the training data for 3 epochs.
5.  **Evaluation:** The model's performance is evaluated on the validation set after each epoch.
6.  **Inference:** A prediction function is provided to test the trained model on new, unseen sentences.

### Using the Prediction Function

Once the model is trained, you can easily predict the sentiment of any text:

```python
# Example of using the prediction function from the notebook
review_text = "This app is great, but the latest update has some bugs."
predicted_sentiment = predict_sentiment(review_text)

print(f"Review: '{review_text}'")
print(f"Predicted Sentiment: {predicted_sentiment}")
# Expected Output: 'Neutral' or 'Positive'
```

## Results

The model was trained for 3 epochs and achieved the following performance on the validation set:

-   **Validation Accuracy:** ~94%
-   **Validation Loss:** ~0.20

### Performance Plots

**(Optional: You can save the plots from your notebook as images and add them here)**

*Training vs. Validation Loss*
![Loss Curve](path/to/your/loss_curve_image.png)

*Validation Accuracy*
![Accuracy Curve](path/to/your/accuracy_curve_image.png)


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
