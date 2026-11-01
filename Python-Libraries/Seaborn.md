# Seaborn

## 1. What is Seaborn?
Seaborn is a statistical data-visualization library built on Matplotlib. It works especially well with Pandas DataFrames and makes it easier to create informative statistical plots.

## 2. Installation and import

```bash
pip install seaborn
```

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

## 3. Load a sample dataset

```python
tips = sns.load_dataset("tips")
print(tips.head())
```

The sample dataset may require an internet connection the first time it is downloaded. You can also pass your own DataFrame to Seaborn functions.

## 4. Common plots

```python
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")
plt.show()

sns.histplot(data=tips, x="total_bill", kde=True)
plt.show()

sns.boxplot(data=tips, x="day", y="total_bill")
plt.show()

sns.countplot(data=tips, x="day")
plt.show()
```

- `scatterplot`: relationships between numeric variables.
- `histplot`: distribution of a numeric variable.
- `boxplot`: median, spread, and potential outliers.
- `countplot`: counts in categories.

## 5. Correlation heatmap

```python
numeric = tips.select_dtypes(include="number")
correlation = numeric.corr()

sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation matrix")
plt.show()
```

Correlation measures association, not proof that one variable causes another.

## 6. Pair plot

```python
sns.pairplot(tips, hue="time")
plt.show()
```

Pair plots compare relationships and distributions across several numeric columns. They can become crowded on large datasets.

## 7. Why it matters in Machine Learning
- Exploring relationships between features.
- Comparing feature distributions across classes.
- Finding possible outliers.
- Investigating correlation before modeling.

## 8. Practice
Use a DataFrame to create a histogram, scatter plot, box plot, and correlation heatmap. Write one sentence describing what each plot reveals.

## 9. Common mistakes
- Treating correlation as causation.
- Plotting too many variables at once.
- Forgetting to inspect data types and missing values.
- Assuming a visually strong relationship guarantees predictive usefulness.
