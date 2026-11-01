# TensorFlow and Keras

## 1. What is TensorFlow?
TensorFlow is an open-source machine-learning framework. Keras is its high-level API for building and training neural networks. TensorFlow is commonly used for deep learning and production deployment.

## 2. Installation

```bash
pip install tensorflow
```

Installation can depend on operating system, Python version, and hardware. Follow the official TensorFlow installation instructions if this command is not suitable for your system.

```python
import tensorflow as tf
print(tf.__version__)
```

## 3. Tensors

A tensor is a multidimensional data structure.

```python
a = tf.constant([1, 2, 3])
b = tf.constant([[1.0, 2.0], [3.0, 4.0]])

print(a)
print(tf.shape(b))
print(a + 10)
```

## 4. Build a simple neural network

This example uses the built-in Fashion-MNIST image dataset.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5, validation_split=0.1)
model.evaluate(x_test, y_test)
```

## 5. Understand the main parts
- **Layers:** transform input data.
- **Dense:** fully connected neural-network layer.
- **Activation:** adds non-linearity, such as ReLU or softmax.
- **Loss:** measures prediction error.
- **Optimizer:** updates model parameters to reduce loss.
- **Epoch:** one full pass through the training data.
- **Batch size:** number of examples processed before an update.

## 6. Save and load a model

```python
model.save("fashion_model.keras")
loaded_model = keras.models.load_model("fashion_model.keras")
```

## 7. Why it matters
TensorFlow/Keras can build neural networks for image classification, text tasks, time-series forecasting, and other deep-learning applications.

## 8. Practice
Change the number of hidden units or epochs, compare validation results, and record what changed. Avoid using the test set repeatedly to tune the model.

## 9. Common mistakes
- Feeding unscaled image pixels when the model expects normalized values.
- Choosing a loss function that does not match the label format.
- Training too long and overfitting.
- Installing a framework version incompatible with the Python environment.
