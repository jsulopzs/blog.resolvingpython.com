# Summarise Time Series data with the DataFrame.resample() function

❌ Don't think of a `for` loop if you want to summarise your daily Time Series by years.

✅ Instead, use the function `resample()` from pandas.

Let me explain it with an example.

We start by loading a DataFrame from a CSV file that contains information on the TSLA stock from 2017-2022.

```python
import pandas as pd

url = 'https://raw.githubusercontent.com/jsulopzs/data/main/tsla_stock.csv'

df_tsla = pd.read_csv(filepath_or_buffer=url)
df_tsla
```


![picture_03_df.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1667467357642/vIDG3TNhP.jpeg align="center")

> cc: @elonmusk
> 
> You're welcome for the promotion 😉

You must ensure that column `Date's dtype` is DateTime.

❌ It must not be an object as in the picture (often interpreted as a string).

```python
df_tsla.dtypes.to_frame(name='dtype')
```

![picture_04_df.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1667467462433/Xt50-WQHn.jpg align="center")

We need to convert the Date column into a `datetime` dtype. To do so, we can use the function `pd.to_datetime()`:

```python
df_tsla.Date = pd.to_datetime(df_tsla.Date)
df_tsla.dtypes.to_frame(name='dtype')
```

![picture_06_df.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1667467480819/tldRHLD_h.jpg align="center")

Before getting into the resample() function, we need to **set the column Date as the index** of the DataFrame:

```python
df_tsla.set_index('Date', inplace=True)
df_tsla
```

![picture_08_df.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1667467503775/jkJ20qV-7.jpeg align="center")

Now let the magic happen; we'll get the maximum value of each column by each year with this simple line of code:

```python
df_tsla.resample(rule='Y').max()
```


![picture_10_df.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1667467527584/WQYJbo_VW.jpeg align="center")

We can do many other things:

1. Summarise by Quarter.
2. Calculate the average and the standard deviation (volatility).

```python
df_tsla.resample(rule='Q').agg(['mean', 'std'])
```

![picture_13_df.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1667467572783/Uy_LrVEBr.jpeg align="center")

To finish it, I always like to add a `background_gradient()` to the DataFrame:

```python
df_tsla.resample(rule='Y').max().style.background_gradient('Greens')
```
 
![picture_14_df.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1667467632849/9_ggRW_Nos.jpeg align="center")

If you enjoyed this, I'd appreciate it if you could support my work by [spreading the word](https://twitter.com/share?url=https%3A%2F%2Fblog.resolvingpython.com%2Fsummarise-time-series-data-with-the-dataframeresample-function&text=%7B%20by%20%40jsulopzs%20%7D) 😊