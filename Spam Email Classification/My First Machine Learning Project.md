# Spam Classification – Code Explanation

## 1. Project Overview

This project uses Machine Learning to classify text messages into two categories:

- **Spam:** Unwanted or suspicious messages.
- **Ham:** Normal messages.

We use the Multinomial Naive Bayes algorithm to build the classification model.

The dataset used in this project is the SMS Spam Collection from the UCI Machine Learning Repository.

Note: This dataset contains SMS messages rather than actual emails, but it demonstrates the same text-classification techniques.

---

## 2. Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import urllib.request
import zipfile
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import joblib
```

### Explanation

**1. NumPy**

```python
import numpy as np
```

- `import` is used to import a library.
- `numpy` is a Python library for numerical operations and arrays.
- `as np` gives the library a shorter name.

**2. Pandas**

```python
import pandas as pd
```

- Pandas is used to organize and manipulate data.
- It allows us to work with tables called DataFrames.
- We use it to load and explore our dataset.

**3. Matplotlib**

```python
import matplotlib.pyplot as plt
```

- Matplotlib is used to create graphs and visualizations.
- We use it to display the results of our data analysis.

**4. Seaborn**

```python
import seaborn as sns
```

- Seaborn is a visualization library built on Matplotlib.
- It helps create statistical charts easily.

**5. urllib.request**

```python
import urllib.request
```

- This module allows Python to access URLs.
- We use it to download the dataset from the internet.

**6. zipfile**

```python
import zipfile
```

- This library allows Python to work with ZIP files.
- Our dataset is provided inside a compressed ZIP file.

**7. BytesIO**

```python
from io import BytesIO
```

- BytesIO allows us to treat downloaded binary data as a file in memory.
- This means we can open the ZIP file without saving it first.

**8. train_test_split**

```python
from sklearn.model_selection import train_test_split
```

- This function divides the dataset into training and testing sets.
- Training data is used to teach the model.
- Testing data is used to evaluate the model.

**9. TfidfVectorizer**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

- Computers cannot directly understand text like humans.
- TfidfVectorizer converts text into numerical features.
- These numerical features can be used by Machine Learning algorithms.

**10. MultinomialNB**

```python
from sklearn.naive_bayes import MultinomialNB
```

- This imports the Multinomial Naive Bayes algorithm.
- It is a classification algorithm commonly used for text classification.
- It will predict whether a message is Spam or Ham.

**11. Pipeline**

```python
from sklearn.pipeline import Pipeline
```

- Pipeline connects multiple Machine Learning steps.
- In our project, it connects text vectorization and classification.
- This makes the training and prediction process easier to manage.

**12. Evaluation Metrics**

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
```

These functions help us evaluate the model:

- `accuracy_score`: Calculates the percentage of correct predictions.
- `classification_report`: Displays Precision, Recall, F1-score, and Support.
- `confusion_matrix`: Shows correct and incorrect classifications.
- `ConfusionMatrixDisplay`: Displays the confusion matrix as a chart.

**13. Joblib**

```python
import joblib
```

- Joblib is used to save and load trained Machine Learning models.
- We use it to save our model so we can reuse it later.

---

## 3. Download the Dataset

```python
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

response = urllib.request.urlopen(url)

zip_file = zipfile.ZipFile(BytesIO(response.read()))

with zip_file.open("SMSSpamCollection") as file:
    df = pd.read_csv(
        file,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

print("Dataset downloaded successfully!")
```

### Explanation

**1. Store the dataset URL**

```python
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
```

- `url` is a variable that stores the dataset link.
- The link points to the compressed dataset.

**2. Download the dataset**

```python
response = urllib.request.urlopen(url)
```

- `urlopen()` opens the URL and retrieves its content.
- The downloaded response is stored in the variable `response`.

**3. Open the ZIP file**

```python
zip_file = zipfile.ZipFile(BytesIO(response.read()))
```

This line contains three operations:

- `response.read()` reads the downloaded data.
- `BytesIO()` treats the downloaded data as a file in memory.
- `zipfile.ZipFile()` opens the ZIP archive.

The opened ZIP archive is stored in `zip_file`.

**4. Open the dataset file**

```python
with zip_file.open("SMSSpamCollection") as file:
```

- `zip_file.open()` opens the dataset file inside the ZIP archive.
- `with` manages the file and closes it automatically.
- `as file` gives the opened file the name `file`.

**5. Read the dataset**

```python
df = pd.read_csv(
    file,
    sep="\t",
    header=None,
    names=["label", "message"]
)
```

- `pd.read_csv()` reads the dataset and creates a DataFrame.
- `file` is the file we want to read.
- `sep="\t"` means the columns are separated by a tab character.
- `header=None` means the original file does not contain column names.
- `names=["label", "message"]` assigns names to the two columns.

The resulting DataFrame is stored in `df`.

**6. Print a confirmation message**

```python
print("Dataset downloaded successfully!")
```

- `print()` displays a message on the screen.
- It confirms that the dataset-loading code has finished successfully.

---

## 4. Explore the Dataset

```python
print(df.head())
```

- `head()` displays the first five rows of the DataFrame by default.
- It helps us understand how the dataset is organized.

```python
print("Dataset Shape:", df.shape)
```

- `shape` returns the number of rows and columns.
- The first number represents rows.
- The second number represents columns.

```python
print(df.info())
```

- `info()` displays information about the DataFrame.
- It shows column names, data types, and non-null values.

```python
print(df["label"].value_counts())
```

- `df["label"]` selects the label column.
- `value_counts()` counts how many times each category appears.
- It tells us how many Spam and Ham messages exist.

### Visualize the Dataset

```python
sns.countplot(data=df, x="label")

plt.title("Spam vs Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")

plt.show()
```

- `sns.countplot()` creates a chart showing the number of messages in each category.
- `data=df` specifies the DataFrame.
- `x="label"` selects the column used for the horizontal axis.
- `plt.title()` adds a chart title.
- `plt.xlabel()` adds a horizontal-axis label.
- `plt.ylabel()` adds a vertical-axis label.
- `plt.show()` displays the chart.

---

## 5. Prepare the Data

```python
print(df.isnull().sum())
```

- `isnull()` checks for missing values.
- `sum()` counts the missing values in each column.
- This helps us identify incomplete data.

```python
print("Duplicates:", df.duplicated().sum())
```

- `duplicated()` identifies repeated rows.
- `sum()` counts the duplicated rows.

```python
df = df.drop_duplicates()
```

- `drop_duplicates()` removes repeated rows.
- The updated DataFrame is stored again in `df`.

Removing duplicates helps prevent repeated messages from unnecessarily influencing the training process.

---

## 6. Split the Dataset

```python
X = df["message"]
y = df["label"]
```

**X:**

- `X` contains the input data.
- In this project, the input is the message text.

**y:**

- `y` contains the target values.
- In this project, the target is the message category: Spam or Ham.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Explanation

- `train_test_split()` divides the dataset into training and testing sets.
- `X` contains the messages.
- `y` contains their correct labels.
- `test_size=0.2` assigns 20% of the data to testing.
- `random_state=42` makes the split reproducible.
- `stratify=y` maintains approximately the same class proportions in both sets.

The results are stored in four variables:

- `X_train`: Training messages.
- `X_test`: Testing messages.
- `y_train`: Correct training labels.
- `y_test`: Correct testing labels.

```python
print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))
```

- `len()` returns the number of elements.
- These statements display the number of training and testing messages.

---

## 7. Build the Machine Learning Model

```python
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])
```

### Explanation

We create a Machine Learning pipeline containing two steps.

**First step:**

```python
("tfidf", TfidfVectorizer())
```

- `TfidfVectorizer()` converts text into numerical features.
- TF-IDF gives words numerical weights based on their importance in the messages.
- The name `"tfidf"` identifies this pipeline step.

**Second step:**

```python
("classifier", MultinomialNB())
```

- `MultinomialNB()` creates the Naive Bayes classification algorithm.
- It learns patterns from the numerical features.
- The name `"classifier"` identifies this pipeline step.

**Why do we use Pipeline?**

Pipeline connects the two steps:

1. Convert messages into numerical features.
2. Classify those features as Spam or Ham.

It also ensures that TF-IDF is fitted on training data only when we train the pipeline.

---

## 8. Train the Model

```python
model.fit(X_train, y_train)

print("Model trained successfully!")
```

### Explanation

```python
model.fit(X_train, y_train)
```

- `fit()` is used to train the model.
- `X_train` contains the training messages.
- `y_train` contains their correct labels.
- The pipeline learns the text representation and trains the classifier.

```python
print("Model trained successfully!")
```

Displays a confirmation message after training finishes.

---

## 9. Test the Model

```python
y_pred = model.predict(X_test)

print(y_pred[:10])
```

### Explanation

```python
y_pred = model.predict(X_test)
```

- `predict()` uses the trained model to classify new input messages.
- `X_test` contains messages that were not used during training.
- The predicted labels are stored in `y_pred`.

```python
print(y_pred[:10])
```

- `y_pred` contains all predictions.
- `[:10]` selects the first ten predictions.
- `print()` displays them.

---

## 10. Evaluate the Model

### A. Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")
```

**Explanation:**

- `accuracy_score()` compares the actual labels with the predicted labels.
- `y_test` contains the correct answers.
- `y_pred` contains the model's predictions.
- The result is stored in `accuracy`.
- Multiplying by 100 converts the result into a percentage.

Accuracy measures the proportion of correct predictions.

### B. Classification Report

```python
print(classification_report(y_test, y_pred))
```

This function displays several evaluation metrics:

**Precision**

Measures how many messages predicted as Spam were actually Spam.

**Recall**

Measures how many actual Spam messages were correctly detected.

**F1-score**

Combines Precision and Recall into one score.

**Support**

Shows the number of actual samples in each class.

### C. Confusion Matrix

```python
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
)
```

- `confusion_matrix()` compares actual labels with predicted labels.
- `labels=["ham", "spam"]` specifies the order of the categories.
- The resulting matrix is stored in `cm`.

```python
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)
```

- `ConfusionMatrixDisplay()` prepares the matrix for visualization.
- `confusion_matrix=cm` provides the calculated matrix.
- `display_labels` specifies the displayed category names.

```python
display.plot(cmap="Blues")
```

- `plot()` creates the visualization.
- `cmap="Blues"` selects the blue color scheme.

```python
plt.title("Confusion Matrix")
plt.show()
```

- `plt.title()` adds a title.
- `plt.show()` displays the chart.

---

## 11. Test Your Own Messages

```python
def predict_message(message):

    prediction = model.predict([message])

    if prediction[0] == "spam":
        print("This message is SPAM!")

    else:
        print("This message is HAM (Normal Message).")
```

### Explanation

**1. Define a function**

```python
def predict_message(message):
```

- `def` is used to define a function.
- `predict_message` is the function name.
- `message` is the input parameter.
- The function receives a message and predicts its category.

**2. Predict the message**

```python
prediction = model.predict([message])
```

- `model.predict()` classifies the input message.
- `[message]` puts the message inside a list because the model expects multiple samples.
- The result is stored in `prediction`.

**3. Check the prediction**

```python
if prediction[0] == "spam":
```

- `if` checks a condition.
- `prediction[0]` accesses the first prediction.
- `==` compares two values.
- If the prediction is `"spam"`, the condition is true.

**4. Display Spam**

```python
print("This message is SPAM!")
```

Displays a message if the prediction is Spam.

**5. Otherwise**

```python
else:
    print("This message is HAM (Normal Message).")
```

- `else` runs when the `if` condition is false.
- It displays a message indicating that the prediction is Ham.

### Test Examples

```python
predict_message("Congratulations! You won a free iPhone. Claim now!")
```

```python
predict_message("Hello, are we meeting tomorrow?")
```

These examples allow us to test the model with messages that were not part of the original dataset.

---

## 12. Save the Trained Model

```python
joblib.dump(model, "spam_classifier.pkl")

print("Model saved successfully!")
```

### Explanation

```python
joblib.dump(model, "spam_classifier.pkl")
```

- `joblib.dump()` saves a Python object to a file.
- `model` is the trained Machine Learning pipeline.
- `"spam_classifier.pkl"` is the output filename.
- The saved file contains the fitted TF-IDF vectorizer and classifier.

```python
print("Model saved successfully!")
```

Displays a confirmation message.

### Load the Saved Model

```python
loaded_model = joblib.load("spam_classifier.pkl")

message = ["You have won a free gift!"]

result = loaded_model.predict(message)

print("Prediction:", result[0])
```

**Explanation:**

- `joblib.load()` loads the saved model.
- `loaded_model` stores the loaded pipeline.
- `message` contains a new message to classify.
- `predict()` generates a prediction.
- `result[0]` accesses the first prediction.
- `print()` displays the result.

---

## 13. Important Python Concepts Used in This Project

| Concept | Meaning |
|---|---|
| Variable | Stores a value |
| String | Represents text |
| Function | Performs a specific task |
| Library | Collection of reusable code |
| Import | Makes a library or tool available |
| DataFrame | A table containing data |
| Parameter | An input passed to a function |
| `=` | Assignment operator |
| `==` | Equality comparison operator |
| `()` | Used to call functions or pass arguments |
| `[]` | Used to create lists or access elements |
| `.` | Accesses an object's methods or attributes |
| `if` | Executes code when a condition is true |
| `else` | Executes code when the condition is false |
| `def` | Defines a function |

---

## 14. Complete Project Workflow

The project follows these steps:

1. Import the required libraries.
2. Download the dataset.
3. Explore the dataset.
4. Prepare the data.
5. Split the dataset into training and testing sets.
6. Convert text into numerical features using TF-IDF.
7. Train the Multinomial Naive Bayes classifier.
8. Predict the labels of testing messages.
9. Evaluate the model.
10. Test new messages.
11. Save the trained model.

---

## 15. Learning Objectives

After completing this project, you should understand:

- How to work with a text dataset.
- How to use Pandas for data exploration.
- The difference between input features and target labels.
- The difference between training and testing data.
- How text is converted into numerical features.
- How the Naive Bayes algorithm is used for classification.
- How to evaluate a Machine Learning model.
- How to make predictions on new messages.
- How to save and load a trained model.

**Project Goal:** Build a beginner-friendly text classification model that can distinguish between Spam and Ham messages.
