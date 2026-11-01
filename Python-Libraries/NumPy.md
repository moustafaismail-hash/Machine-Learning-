# NumPy

## 1. What is NumPy?
NumPy (Numerical Python) is a Python library for fast numerical computing. Its central object is the `ndarray`, an efficient multidimensional array. It is widely used in data science and forms a foundation for many machine-learning tools.

**Use it for:** vectors, matrices, numerical calculations, statistics, random data, and reshaping data.

## 2. Installation and import

```bash
pip install numpy
```

```python
import numpy as np
```

## 3. Creating arrays

```python
import numpy as np

a = np.array([10, 20, 30])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

print(a)
print(matrix)
print(matrix.ndim)   # number of dimensions
print(matrix.shape)  # (2, 3): rows, columns
print(matrix.size)   # total elements
print(matrix.dtype)  # element data type
```

Other useful constructors:

```python
np.zeros((2, 3))       # array filled with zeros
np.ones((2, 2))        # array filled with ones
np.full((2, 2), 7)     # array filled with 7
np.arange(0, 10, 2)    # 0, 2, 4, 6, 8
np.linspace(0, 1, 5)  # five evenly spaced values
np.eye(3)              # 3x3 identity matrix
```

## 4. Indexing, slicing, and reshaping

```python
x = np.array([5, 10, 15, 20, 25])
print(x[0])       # first item
print(x[-1])      # last item
print(x[1:4])     # items at indexes 1, 2, 3

m = np.arange(1, 7).reshape(2, 3)
print(m)
print(m[0, 1])    # row 0, column 1
print(m[:, 0])    # all rows, first column
```

A reshape must preserve the total number of elements.

## 5. Array operations and broadcasting

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print(a + b)   # [11 22 33]
print(a * 2)   # [2 4 6]
print(a ** 2)  # [1 4 9]
```

NumPy performs many operations element by element. **Broadcasting** allows compatible shapes to work together, such as adding a scalar to every element. For two dimensions, dimensions are compatible when they are equal or one of them is 1.

## 6. Aggregations and statistics

```python
data = np.array([[1, 2, 3], [4, 5, 6]])

print(np.sum(data))
print(np.mean(data))
print(np.median(data))
print(np.std(data))
print(np.min(data))
print(np.max(data))
print(np.sum(data, axis=0))  # column totals
print(np.sum(data, axis=1))  # row totals
```

## 7. Boolean filtering and missing values

```python
scores = np.array([45, 72, 88, 59, 95])
passed = scores[scores >= 60]
print(passed)

values = np.array([1.0, np.nan, 3.0])
print(np.isnan(values))
print(np.nanmean(values))  # mean while ignoring NaN
```

## 8. Random numbers

```python
rng = np.random.default_rng(seed=42)
print(rng.integers(0, 10, size=5))
print(rng.random((2, 3)))
```

Using a seed makes random examples reproducible.

## 9. Linear algebra

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A @ B)             # matrix multiplication
print(np.linalg.det(A))  # determinant
print(np.linalg.inv(A))  # inverse, if it exists
```

## 10. Why NumPy matters in Machine Learning
- Representing feature data as arrays and matrices.
- Performing vectorized calculations efficiently.
- Calculating means, standard deviations, and normalization values.
- Understanding the matrix operations used in linear models and neural networks.

## 11. Practice
1. Create a 3x3 array containing numbers 1 through 9.
2. Print its second row and third column.
3. Calculate the mean of each column.
4. Select all values greater than 5.
5. Multiply the matrix by 2.

## 12. Common mistakes
- Confusing a Python list with a NumPy array.
- Using incompatible shapes in arithmetic.
- Forgetting that indexing starts at zero.
- Using `*` for matrix multiplication; use `@` for matrix multiplication.
