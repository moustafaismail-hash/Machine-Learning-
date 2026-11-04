# 6. Model Training & Evaluation

Training a model is not enough. We also need to evaluate its ability to make accurate predictions on unseen data.

## 6.1 Dataset Splitting

- **Training Set:** Used to train the model.
- **Validation Set:** Used to tune hyperparameters and select the best model.
- **Test Set:** Used for the final evaluation of the model.

## 6.2 Overfitting vs Underfitting

### Overfitting

Occurs when a model learns the training data too well, including noise and unnecessary details. It performs well on training data but poorly on new data.

### Underfitting

Occurs when a model is too simple to capture the underlying patterns in the data. It performs poorly on both training and testing data.

## 6.3 Cross-Validation

Cross-Validation is a technique used to evaluate a model by dividing the dataset into multiple subsets (folds).

In K-Fold Cross-Validation, the model is trained and evaluated multiple times, using a different fold for validation each time.
