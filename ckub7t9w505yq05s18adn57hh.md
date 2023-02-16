# How to Analyze Data through Visualization

What is a plot?

- A visual representation of the data

Which data? How is it usually structured?

- In a table. For example:

```python
import seaborn as sns

df = sns.load_dataset('mpg', index_col='name')
df.head()
```
 
![head.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1633264393828/lomiNX7Dj.png)

How can you Visualice this `DataFrame`?

- We could make a point for every car based on
    1. weight
    2. mpg


```python
sns.scatterplot(x='weight', y='mpg', data=df);
```

![plot1.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1633264345405/h5XR0hfT0.jpeg)

Which conclusions can you make out of this plot?

- Well, you may observe that the location of the points are descending as we move to the right

- This means that the `weight` of the car may produce a lower capacity to make kilometres `mpg`

How can you measure this relationship?

- Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X=df[['weight']], y=df.mpg)
model.__dict__
```

- Resulting in ↓


    {'fit_intercept': True,
     'normalize': False,
     'copy_X': True,
     'n_jobs': None,
     'n_features_in_': 1,
     'coef_': array([-0.00767661]),
     '_residues': 7474.8140143821,
     'rank_': 1,
     'singular_': array([16873.20281508]),
     'intercept_': 46.31736442026565}


Which is the mathematical formula for this relationship?

$$mpg = 46.31 - 0.00767 \cdot weight$$

- This equation means that the `mpg` gets 0.00767 units lower for **every unit** that `weight` **increases**.

Could you visualise this equation in a plot?

- Absolutely, we could make the predictions from the original data and plot them.

## Predictions


```python
y_pred = model.predict(X=df[['weight']])

dfsel = df[['weight', 'mpg']].copy()
dfsel['prediction'] = y_pred

dfsel.head()
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
      <th>weight</th>
      <th>mpg</th>
      <th>prediction</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>chevrolet chevelle malibu</th>
      <td>3504</td>
      <td>18.0</td>
      <td>19.418523</td>
    </tr>
    <tr>
      <th>buick skylark 320</th>
      <td>3693</td>
      <td>15.0</td>
      <td>17.967643</td>
    </tr>
    <tr>
      <th>plymouth satellite</th>
      <td>3436</td>
      <td>18.0</td>
      <td>19.940532</td>
    </tr>
    <tr>
      <th>amc rebel sst</th>
      <td>3433</td>
      <td>16.0</td>
      <td>19.963562</td>
    </tr>
    <tr>
      <th>ford torino</th>
      <td>3449</td>
      <td>17.0</td>
      <td>19.840736</td>
    </tr>
  </tbody>
</table>
</div>



- Out of this table, you could observe that predictions don't exactly match the reality, but it approximates.

- For example, Ford Torino's `mpg` is 17.0, but our model predicts 19.84.

## Model Visualization


```python
sns.scatterplot(x='weight', y='mpg', data=dfsel)
sns.scatterplot(x='weight', y='prediction', data=dfsel);
```

![plot2.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1633264568851/DNGxO-SKx.jpeg)

1. The blue points represent the actual data.
2. The orange points represent the predictions of the model.

> I teach Python, R, Statistics & Data Science. I like to produce content that helps people to understand these topics better.
>
> Feel free and welcomed to give me feedback as I would like to make my tutorials clearer and generate content that interests you 🤗
>
> You can see my Tutor Profile [here](https://www.superprof.co.uk/online-rstudio-python-anaconda-jupyter-big-data-artificial-intelligence-sport-analytics-business-analytics.html) if you need Private Tutoring lessons.
