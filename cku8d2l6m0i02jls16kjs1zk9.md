---
title: "Why do all Machine Learning models follow the same steps?"
seoTitle: "What Machine Learning Models have in common"
seoDescription: "Uncovering the patterns across different Machine Learning models will make you understand how they work in Python, a programming language"
datePublished: Fri Oct 01 2021 12:47:17 GMT+0000 (Coordinated Universal Time)
cuid: cku8d2l6m0i02jls16kjs1zk9
slug: why-do-all-machine-learning-models-follow-the-same-steps
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1633092426398/uk6VplZUp.png
tags: python, data-science, machine-learning, python3, programming-tips

---

## Introduction

It's tough to find things that always work the same way in programming.

The steps of a Machine Learning (ML) model can be an exception.

Each time we want to compute a model _(mathematical equation)_ and make predictions with it, we would always make the following steps:

1. `model.fit()` → to **compute the numbers** of the mathematical equation..
2. `model.predict()` → to **calculate predictions** through the mathematical equation.
3. `model.score()` → to measure **how good the model's predictions are**.

And I am going to show you this with 3 different ML models.

- `DecisionTreeClassifier()`
- `RandomForestClassifier()`
- `LogisticRegression()`

## Load the Data

But first, let's load a dataset from [CIS](https://www.cis.es/cis/opencms/ES/index.html) executing the lines of code below:
> - The goal of this dataset is
> - To predict `internet_usage` of **people** (rows)
> - Based on their **socio-demographical characteristics** (columns)


```python
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/jsulopz/data/main/internet_usage_spain.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>internet_usage</th>
      <th>sex</th>
      <th>age</th>
      <th>education</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Female</td>
      <td>66</td>
      <td>Elementary</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Male</td>
      <td>72</td>
      <td>Elementary</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Male</td>
      <td>48</td>
      <td>University</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Male</td>
      <td>59</td>
      <td>PhD</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Female</td>
      <td>44</td>
      <td>PhD</td>
    </tr>
  </tbody>
</table>
</div>



## Data Preprocessing

We need to transform the categorical variables to **dummy variables** before computing the models:


```python
df = pd.get_dummies(df, drop_first=True)
df.head()
```



![df_dummy_head.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1638403244509/8jU84GN3G.png)




## Feature Selection

Now we separate the variables on their respective role within the model:


```python
target = df.internet_usage
explanatory = df.drop(columns='internet_usage')
```

## ML Models

### Decision Tree Classifier


```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X=explanatory, y=target)

pred_dt = model.predict(X=explanatory)
accuracy_dt = model.score(X=explanatory, y=target)
```

### Support Vector Machines


```python
from sklearn.svm import SVC

model = SVC()
model.fit(X=explanatory, y=target)

pred_sv = model.predict(X=explanatory)
accuracy_sv = model.score(X=explanatory, y=target)
```

### K Nearest Neighbour


```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(X=explanatory, y=target)

pred_kn = model.predict(X=explanatory)
accuracy_kn = model.score(X=explanatory, y=target)
```

The only thing that changes are the results of the prediction. The models are different. But they all follow the **same steps** that we described at the beginning:

1. `model.fit()` → to compute the mathematical formula of the model
2. `model.predict()` → to calculate predictions through the mathematical formula
3. `model.score()` → to get the success ratio of the model

## Comparing Predictions

You may observe in the following table how the *different models make different predictions*, which often doesn't coincide with reality (misclassification).

For example, `model_svm` doesn't correctly predict the row 214; as if this person *used internet* `pred_svm=1`, but they didn't: `internet_usage` for 214 in reality is 0.


```python
df_pred = pd.DataFrame({'internet_usage': df.internet_usage,
                        'pred_dt': pred_dt,
                        'pred_svm': pred_sv,
                        'pred_lr': pred_kn})

df_pred.sample(10, random_state=7)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>internet_usage</th>
      <th>pred_dt</th>
      <th>pred_svm</th>
      <th>pred_lr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>214</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2142</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1680</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1522</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>325</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2283</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1263</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>993</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2190</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Choose Best Model

Then, we could choose the model with a **higher number of successes** on predicting the reality.


```python
df_accuracy = pd.DataFrame({'accuracy': [accuracy_dt, accuracy_sv, accuracy_kn]},
                           index = ['DecisionTreeClassifier()', 'SVC()', 'KNeighborsClassifier()'])

df_accuracy
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DecisionTreeClassifier()</th>
      <td>0.859878</td>
    </tr>
    <tr>
      <th>SVC()</th>
      <td>0.783707</td>
    </tr>
    <tr>
      <th>KNeighborsClassifier()</th>
      <td>0.827291</td>
    </tr>
  </tbody>
</table>
</div>



Which is the best model here?

- Let me know in the comments below ↓