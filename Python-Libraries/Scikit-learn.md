# Scikit-learn

## 1. What is Scikit-learn?
Scikit-learn is a widely used Python library for classical machine learning. It includes tools for preprocessing, supervised and unsupervised learning, model evaluation, and model selection.

## 2. Installation and imports

```bash
pip install scikit-learn
```

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
```

## 3. Typical machine-learning workflow
1. Define the problem and target.
2. Collect and inspect data.
3. Split data into training and test sets.
4. Preprocess features.
5. Choose and train a model.
6. Evaluate on data not used for training.
7. Improve carefully and document results.

## 4. First classification example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

`X` contains features; `y` contains the target labels. `fit` trains the model and `predict` produces predictions.

## 5. Regression example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))
```

For regression, common metrics include MAE, MSE, and RMSE. For classification, common metrics include accuracy, precision, recall, F1-score, and a confusion matrix.

## 6. Preprocessing and pipelines

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)
```

A pipeline applies preprocessing and modeling in a consistent order. Fit preprocessing only on training data; pipelines help prevent data leakage during cross-validation.

## 7. Common algorithms
- Linear Regression: predicts a continuous value.
- Logistic Regression: classification model.
- Decision Tree: learns rule-like splits.
- Random Forest: combines many decision trees.
- K-Nearest Neighbors: predicts using nearby examples.
- Support Vector Machine: finds a separating boundary.
- K-Means: groups data into clusters.

## 8. Model selection

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5)
print(scores.mean())
```

Cross-validation estimates how a model may perform across different training/validation splits. Keep a final test set separate until final evaluation.

## 9. Why it matters
Scikit-learn is often the best starting point for learning practical machine learning because its API is consistent and it covers many core algorithms.

## 10. Practice
Train a classifier on Iris, compare a decision tree with logistic regression, report accuracy and a classification report, and explain why test data must not be used to train the model.

## 11. Common mistakes
- Evaluating only on training data.
- Data leakage from preprocessing before splitting.
- Using accuracy alone on imbalanced datasets.
- Forgetting that model performance depends on data quality and problem definition.
