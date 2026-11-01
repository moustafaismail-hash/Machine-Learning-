# PyTorch

## 1. What is PyTorch?
PyTorch is an open-source deep-learning framework. It provides tensors, automatic differentiation, neural-network layers, and tools for training models. It is popular in research and practical deep learning.

## 2. Installation

Install the command that matches your operating system and CPU/GPU from the official PyTorch installation selector. A basic CPU installation may be available with:

```bash
pip install torch
```

```python
import torch
print(torch.__version__)
```

## 3. Tensors

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)
print(x.dtype)
print(x + 2)
print(x @ x)  # matrix multiplication
```

## 4. Automatic differentiation

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()

print(x.grad)  # derivative: 2*x + 3, evaluated at x=2
```

Autograd calculates gradients, which are used to update model parameters.

## 5. Build a simple neural network

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Toy dataset: y is approximately 2*x
X = torch.randn(500, 1)
y = 2 * X + 0.1 * torch.randn(500, 1)

loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(30):
    for batch_X, batch_y in loader:
        prediction = model(batch_X)
        loss = loss_fn(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training complete")
```

## 6. Key concepts
- `torch.Tensor`: stores numerical data.
- `nn.Module`: base class for model components.
- `DataLoader`: creates batches and can shuffle examples.
- Loss function: measures model error.
- Optimizer: updates model parameters.
- `zero_grad()`: clears old gradients before the next update.
- `backward()`: computes gradients.

## 7. Evaluation mode

```python
model.eval()
with torch.no_grad():
    predictions = model(X[:5])
```

Use evaluation mode for layers such as dropout and batch normalization, and `no_grad()` when gradients are not needed.

## 8. Why it matters
PyTorch is useful for experimenting with neural-network architectures, computer vision, natural-language processing, and custom deep-learning research.

## 9. Practice
Train the toy model, change the learning rate, and compare the loss. Then split data into training and test sets and evaluate on held-out data.

## 10. Common mistakes
- Forgetting `optimizer.zero_grad()`.
- Forgetting `model.eval()` for evaluation.
- Mixing CPU and GPU tensors in the same operation.
- Comparing models using training loss alone.
