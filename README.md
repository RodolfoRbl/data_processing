# Weight of Evidence (WOE) & Information Value (IV)

PySpark implementation of Weight of Evidence (WOE) and Information Value (IV) using PySpark.

## Quickstart

Please check the main module for the example.

### 1) Prepare the data

```python
df = <spark_dataframe>
cols_to_woe = <list_of_categorical_columns_to_encode>
label_col = <label_column>
good_label = <good_label>
```

### 2) WOE Encoding

```python
woe = WOE_IV(df, cols_to_woe, label_col, good_label)
woe.fit()

encoded_df = woe.transform(df)
```

### 3) Information Value

```python
ivs = woe.compute_iv()
```

### 4) Save parameters
```python
woe.save_params(<path>)
```

### 5) Transform DataFrame
```python
woe.WOE_IV.load_params(<path>)
transformed = woe.WOE_IV.transform(<DataFrame>)
```

