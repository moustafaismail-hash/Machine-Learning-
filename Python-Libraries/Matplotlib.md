# Matplotlib

## 1. What is Matplotlib?
Matplotlib is a plotting library used to create charts and visualizations. It helps reveal patterns, trends, outliers, and relationships in data.

## 2. Installation and import

```bash
pip install matplotlib
```

```python
import matplotlib.pyplot as plt
```

## 3. Basic line chart

```python
days = [1, 2, 3, 4, 5]
sales = [10, 13, 12, 18, 20]

plt.plot(days, sales, marker="o", label="Sales")
plt.title("Sales over time")
plt.xlabel("Day")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
```

## 4. Common chart types

```python
# Bar chart
plt.bar(["A", "B", "C"], [5, 8, 3])
plt.title("Category comparison")
plt.show()

# Scatter plot
plt.scatter([1, 2, 3, 4], [2, 4, 3, 7])
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

# Histogram
plt.hist([2, 3, 3, 4, 4, 4, 5, 6], bins=5)
plt.title("Distribution")
plt.show()
```

- Line chart: trends over an ordered sequence or time.
- Bar chart: compare categories.
- Scatter plot: relationship between two numeric variables.
- Histogram: distribution of one numeric variable.

## 5. Figure and axes

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([1, 2, 3], [2, 5, 4])
ax.set(title="Example", xlabel="X", ylabel="Y")
fig.tight_layout()
plt.show()
```

The object-oriented `fig, ax` approach is useful for more complex figures.

## 6. Saving a chart

```python
plt.plot([1, 2, 3], [4, 2, 6])
plt.savefig("chart.png", dpi=150, bbox_inches="tight")
plt.show()
```

Call `savefig` before `show` for reliable saving in many workflows.

## 7. Why it matters in Machine Learning
- Visualizing feature distributions.
- Comparing actual and predicted values.
- Plotting learning curves and model performance.
- Detecting outliers and unusual patterns.

## 8. Practice
Create a line chart for weekly sales, a bar chart for sales by product, and a histogram for customer ages. Add a title and axis labels to each.

## 9. Common mistakes
- Forgetting `plt.show()` in a script.
- Choosing a chart type that does not match the question.
- Omitting labels and units.
- Comparing values with misleading axis limits.
