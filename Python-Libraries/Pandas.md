# Pandas

## 1. What is Pandas?
Pandas is a library for loading, cleaning, exploring, transforming, and analyzing structured data. Its main objects are `Series` (one-dimensional labeled data) and `DataFrame` (two-dimensional tabular data).

## 2. Installation and import

```bash
pip install pandas
```

```python
import pandas as pd
```

## 3. Creating Series and DataFrames

```python
ages = pd.Series([20, 25, 30], name="age")

df = pd.DataFrame({
    "name": ["Ali", "Mona", "Omar", "Sara"],
    "age": [20, 25, 30, 22],
    "score": [80, 92, 75, 88]
})
print(df)
```

## 4. Inspecting data

```python
print(df.head())       # first rows
print(df.tail())       # last rows
print(df.shape)        # (rows, columns)
print(df.columns)
print(df.dtypes)
print(df.info())
print(df.describe())   # numeric summary
```

## 5. Selecting rows and columns

```python
print(df["name"])                 # one column
print(df[["name", "score"]])      # multiple columns
print(df.loc[0, "name"])          # label-based selection
print(df.iloc[0, 1])              # position-based selection
print(df[df["score"] >= 85])      # filtering
```

`loc` uses labels; `iloc` uses integer positions.

## 6. Cleaning data

```python
df = pd.DataFrame({
    "name": ["Ali", "Mona", None, "Sara"],
    "age": [20, None, 30, 22],
    "score": [80, 92, None, 88]
})

print(df.isna().sum())             # missing values per column
df["age"] = df["age"].fillna(df["age"].median())
df = df.dropna(subset=["name"])
df = df.drop_duplicates()
```

Choose a missing-value strategy based on what the data means. Do not automatically fill every missing value with zero.

## 7. Transforming columns

```python
df["age_next_year"] = df["age"] + 1
df["passed"] = df["score"] >= 60
df = df.rename(columns={"score": "exam_score"})
```

## 8. Grouping and aggregation

```python
sales = pd.DataFrame({
    "city": ["Cairo", "Giza", "Cairo", "Giza"],
    "amount": [100, 150, 200, 50]
})

summary = sales.groupby("city")["amount"].agg(["sum", "mean", "count"])
print(summary)
```

## 9. Sorting and combining data

```python
df = df.sort_values("age", ascending=False)

left = pd.DataFrame({"id": [1, 2], "name": ["Ali", "Mona"]})
right = pd.DataFrame({"id": [1, 2], "score": [80, 95]})
joined = pd.merge(left, right, on="id", how="inner")
```

Use `concat` to stack DataFrames and `merge` to join them using keys.

## 10. Reading and writing files

```python
data = pd.read_csv("data.csv")
data.to_csv("clean_data.csv", index=False)

# Other common formats:
# pd.read_excel("data.xlsx")
# pd.read_json("data.json")
```

Excel support may require installing `openpyxl`.

## 11. Why Pandas matters in Machine Learning
- Loading datasets from CSV and Excel files.
- Understanding columns, data types, and distributions.
- Handling missing values and duplicate rows.
- Filtering records and creating useful features.
- Preparing clean data before passing it to a model.

## 12. Practice
Load a CSV dataset, inspect it with `head()` and `info()`, count missing values, remove duplicates, group by one category, and save a cleaned copy.

## 13. Common mistakes
- Confusing `loc` and `iloc`.
- Forgetting `index=False` when saving a CSV if an extra index column is unwanted.
- Modifying data without checking whether the change makes sense.
- Data leakage: calculating preprocessing information using test data.
