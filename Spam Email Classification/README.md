# Spam Email Classification Using Machine Learning

## Project Overview

This project aims to build a Machine Learning model that classifies text messages into two categories: Spam and Ham (Normal Messages).

The project uses the Multinomial Naive Bayes algorithm to identify spam messages based on their text content.

It is a beginner-level Machine Learning project designed to demonstrate the basic concepts of text classification.

**Note:** The initial dataset used in this project is the SMS Spam Collection dataset, which contains SMS messages rather than actual emails.

## Project Objectives

- Understand the basics of Machine Learning.
- Learn how to work with text datasets.
- Apply text preprocessing and feature extraction.
- Build a classification model using Naive Bayes.
- Evaluate model performance using classification metrics.
- Predict whether a new message is Spam or Ham.

## Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Development environment |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| Joblib | Saving and loading the trained model |

## Machine Learning Algorithm

**Multinomial Naive Bayes**

Multinomial Naive Bayes is a probabilistic classification algorithm commonly used for text classification tasks.

It estimates the probability that a message belongs to a specific category based on its features.

## Dataset

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

Dataset link:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains two main columns:

- `label`: The message category (Spam or Ham).
- `message`: The actual text message.

## Project Workflow

1. Import the required libraries.
2. Download and load the dataset.
3. Explore and understand the data.
4. Prepare the dataset and remove duplicates.
5. Split the data into training and testing sets.
6. Convert text into numerical features using TF-IDF.
7. Train the Multinomial Naive Bayes model.
8. Evaluate the model performance.
9. Test the model with new messages.
10. Save the trained model.

## Feature Extraction

**TF-IDF (Term Frequency-Inverse Document Frequency)**

TF-IDF converts text messages into numerical features that Machine Learning algorithms can process.

It assigns weights to words based on their frequency in a message and their importance across the dataset.

## Model Evaluation

The model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics help measure how effectively the model distinguishes between Spam and Ham messages.

## Project Structure

```text
Spam-Email-Classification/
│
├── README.md
├── Code-Explanation.md
├── spam_classification.ipynb
│
├── dataset/
│
└── models/
    └── spam_classifier.pkl
```

## How to Run the Project

1. Clone this repository:

```bash
git clone YOUR_REPOSITORY_LINK
```

2. Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

3. Open Jupyter Notebook:

```bash
jupyter notebook
```

4. Open `spam_classification.ipynb`.

5. Run the notebook cells in order.

## Learning Outcomes

Through this project, I aim to gain practical experience in:

- Python for Machine Learning.
- Data exploration and preparation.
- Text feature extraction.
- Supervised Learning.
- Classification algorithms.
- Model evaluation and prediction.

## Project Status

In Progress

## Author

Computer Engineering Student

Interested in Artificial Intelligence and Machine Learning.
