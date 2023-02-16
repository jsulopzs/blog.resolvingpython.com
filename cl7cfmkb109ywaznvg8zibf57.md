# #05 | DateTime Object's Potential within Pandas, a Python Library

**© Jesús López**

Ask him any doubt on **[Twitter](https://twitter.com/jsulopzs)** or **[LinkedIn](https://linkedin.com/in/jsulopzs)**

## Possibilities

Look at the following example as an aspiration you can achieve if you fully understand and replicate this whole tutorial with your data.

Let's load a dataset containing information on the Tesla Stock daily (rows) transactions (columns) in the Stock Market.


```python
import pandas as pd

url = 'https://raw.githubusercontent.com/jsulopzs/data/main/tsla_stock.csv'
df_tesla = pd.read_csv(url, index_col=0, parse_dates=['Date'])
df_tesla
```


![df1.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661635894721/ydNL2622_.jpeg align="left")

You may calculate the `.mean()` of each column by the last Business day of each Month (BM):


```python
df_tesla.resample('BM').mean()
```


![df2.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661635900710/Cn8g4orT4.jpeg align="left")

Or the Weekly Average:


```python
df_tesla.resample('W-FRI').mean()
```


![df3.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661635906331/pVnDSPNfU.jpeg align="left")

And many more; see the full list [here](https://pandas.pydata.org/pandas-docs/dev/user_guide/timeseries.html#offset-aliases).

Pretty straightforward compared to other libraries and programming languages.

It's not a casualty they say Python is the future language because its libraries simplify many operations where most people believe they would have needed a `for` loop.

Let's apply other pandas techniques to the DateTime object:


```python
df_tesla['year'] = df_tesla.index.year
df_tesla['month'] = df_tesla.index.month
```

The following values represent the average Close price by each month-year combination:


```python
df_tesla.pivot_table(index='year', columns='month', values='Close', aggfunc='mean').round(2)
```


![df4.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661635914639/YRaEmtvMl.jpeg align="left")

We could even style it to get a better insight by colouring the cells:


```python
df_stl = df_tesla.pivot_table(
    index='year',
    columns='month',
    values='Close',
    aggfunc='mean',
    fill_value=0).style.format('{:.2f}').background_gradient(axis=1)

df_stl
```


![df5.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661635924250/cOH6NRXCU.jpeg align="left")

And they represent the volatility with the standard deviation:


```python
df_stl = df_tesla.pivot_table(
    index='year',
    columns='month',
    values='Close',
    aggfunc='std',
    fill_value=0).style.format('{:.2f}').background_gradient(axis=1)

df_stl
```


![df6.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661635930567/mvNRFyTvC.jpeg align="left")

In this article, we'll dig into the details of the Panda's DateTime-related object in Python to understand the required knowledge to come up with awesome calculations like the ones we saw above.

First, let's reload the dataset to start from the basics.


```python
df_tesla = pd.read_csv(url, parse_dates=['Date'])
df_tesla
```


![df7.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661635936408/TdykkvX-H.jpeg align="left")

## Series DateTime

An essential part of learning something is the practicability and the understanding of counterexamples where we understand the errors.

Let's go with basic thinking to understand the importance of the DateTime object and how to work with it. So, out of all the columns in the DataFrame, we'll now focus on `Date`:


```python
df_tesla.Date
```




    0      2017-01-03
    1      2017-01-04
              ...    
    1378   2022-06-24
    1379   2022-06-27
    Name: Date, Length: 1380, dtype: datetime64[ns]



What information could we get from a `DateTime` object?

- We may think we can get the month, but it turns out we can't in the following manner:


```python
df_tesla.Date.month
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Input In [53], in <cell line: 1>()
    ----> 1 df_tesla.Date.month


    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/generic.py:5575, in NDFrame.__getattr__(self, name)
       5568 if (
       5569     name not in self._internal_names_set
       5570     and name not in self._metadata
       5571     and name not in self._accessors
       5572     and self._info_axis._can_hold_identifiers_and_holds_name(name)
       5573 ):
       5574     return self[name]
    -> 5575 return object.__getattribute__(self, name)


    AttributeError: 'Series' object has no attribute 'month'


Programming exists to simplify our lives, not make them harder.

Someone has probably developed a simpler functionality if you think there must be a simpler way to perform certain operations. Therefore, don't limit programming applications to complex ideas and rush towards a `for` loop, for example; proceed through trial and error without losing hope.

In short, we need to bypass the `dt` instance to access the `DateTime` functions:


```python
df_tesla.Date.dt
```




    <pandas.core.indexes.accessors.DatetimeProperties object at 0x16230a2e0>



### Process the Month


```python
df_tesla.Date.dt.month
```




    0       1
    1       1
           ..
    1378    6
    1379    6
    Name: Date, Length: 1380, dtype: int64



We can use more elements than just `.month`:

### Process the Month Name


```python
df_tesla.Date.dt.month_name()
```




    0       January
    1       January
             ...   
    1378       June
    1379       June
    Name: Date, Length: 1380, dtype: object



### Process the Year, Week & Day


```python
df_tesla.Date.dt.isocalendar()
```


![df8.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661635953117/KanxQvge6.jpeg align="left")

### Process the Quarter


```python
df_tesla.Date.dt.quarter
```




    0       1
    1       1
           ..
    1378    2
    1379    2
    Name: Date, Length: 1380, dtype: int64



### Process the Year-Month for each Date


```python
df_tesla.Date.dt.to_period('M')
```




    0       2017-01
    1       2017-01
             ...   
    1378    2022-06
    1379    2022-06
    Name: Date, Length: 1380, dtype: period[M]



### Process the Weekly Period for each Date


```python
df_tesla.Date.dt.to_period('W-FRI')
```




    0       2016-12-31/2017-01-06
    1       2016-12-31/2017-01-06
                    ...          
    1378    2022-06-18/2022-06-24
    1379    2022-06-25/2022-07-01
    Name: Date, Length: 1380, dtype: period[W-FRI]



## Time Zones

Pandas contain functionality that allows us to place Time Zones into the objects to ease the work of data from different countries and regions.

Before getting deeper into Time Zones, we need to set the `Date` as the `index` (rows) of the `DataFrame`:


```python
df_tesla.set_index('Date', inplace=True)
df_tesla
```


![df9.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661635964373/f_QXrYQJB.jpeg align="left")

We can tell Python the `DateTimeIndex` of the `DataFrame` comes from Madrid:


```python
df_tesla.index = df_tesla.index.tz_localize('Europe/Madrid')
df_tesla
```


![df10.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661635976064/PVAALcp82.jpeg align="left")

And **change** it to another Time Zone, like **Moscow**:


```python
df_tesla.index.tz_convert('Europe/Moscow')
```




    DatetimeIndex(['2017-01-03 02:00:00+03:00', '2017-01-04 02:00:00+03:00',
                   '2017-01-05 02:00:00+03:00', '2017-01-06 02:00:00+03:00',
                   ...
                   '2022-06-22 01:00:00+03:00', '2022-06-23 01:00:00+03:00',
                   '2022-06-24 01:00:00+03:00', '2022-06-27 01:00:00+03:00'],
                  dtype='datetime64[ns, Europe/Moscow]', name='Date', length=1380, freq=None)



We could have applied the transformation in the `DataFrame` object itself:


```python
df_tesla.tz_convert('Europe/Moscow')
```


![df11.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661635994628/uwyqtzchZ.jpeg align="left")

We can observe the hour has changed accordingly.

The **Pandas Time Zone** functionality is useful for combining timed data from different regions around the globe.


%%[newsletter-inline]

## Summarising the Dates

To summarise, for example, the information of daily operations into months, we can apply different functions with each one having its unique ability (it's up to you to select the one that suits your needs):

1. `.groupby()`
2. `.resample()`
3. `.pivot_table()`

Let's show some examples:

### Groupby


```python
df_tesla.groupby(by=df_tesla.index.year).Volume.sum()
```




    Date
    2017     7950157000
    2018    10808194000
    2019    11540242000
    2020    19052912400
    2021     6902690500
    2022     3407576732
    Name: Volume, dtype: int64



The function `.groupby()` packs the rows of the same year:


```python
df_tesla.groupby(by=df_tesla.index.year)
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x1622eecd0>



To later summarise the total volume in each pack as we saw before.

An easier way?

### Resample


```python
df_tesla.Volume.resample('Y').sum()
```




    Date
    2017-12-31 00:00:00+01:00     7950157000
    2018-12-31 00:00:00+01:00    10808194000
    2019-12-31 00:00:00+01:00    11540242000
    2020-12-31 00:00:00+01:00    19052912400
    2021-12-31 00:00:00+01:00     6902690500
    2022-12-31 00:00:00+01:00     3407576732
    Freq: A-DEC, Name: Volume, dtype: int64



We first select the column in which we want to apply the operation:


```python
df_tesla.Volume
```




    Date
    2017-01-03 00:00:00+01:00    29616500
    2017-01-04 00:00:00+01:00    56067500
                                   ...   
    2022-06-24 00:00:00+02:00    31866500
    2022-06-27 00:00:00+02:00    21237332
    Name: Volume, Length: 1380, dtype: int64



And apply the `.resample()` function to take a [Date Offset](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases) to aggregate the `DateTimeIndex`. In this example, we aggregate by year `'Y'`:


```python
df_tesla.Volume.resample('Y')
```




    <pandas.core.resample.DatetimeIndexResampler object at 0x16230abe0>



And apply mathematical operations to the aggregated objects separately as we saw before:


```python
df_tesla.Volume.resample('Y').sum()
```




    Date
    2017-12-31 00:00:00+01:00     7950157000
    2018-12-31 00:00:00+01:00    10808194000
    2019-12-31 00:00:00+01:00    11540242000
    2020-12-31 00:00:00+01:00    19052912400
    2021-12-31 00:00:00+01:00     6902690500
    2022-12-31 00:00:00+01:00     3407576732
    Freq: A-DEC, Name: Volume, dtype: int64



We could have also calculated the `.sum()` for all the columns if we didn't select just the `Volume`:


```python
df_tesla.resample('Y').sum()
```


![df12.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661636004259/G3OlOHgeg.jpeg align="left")

As always, we should strive to represent the information in the clearest manner for anyone to understand. Therefore, we could even visualize the aggregated volume by year with two more words:


```python
df_tesla.Volume.resample('Y').sum().plot.bar();
```


![plot1.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661636009781/0Qj1gCbIT.jpeg align="left")

Let's now try different Date Offsets:

#### Monthly


```python
df_tesla.Volume.resample('M').sum()
```




    Date
    2017-01-31 00:00:00+01:00    503398000
    2017-02-28 00:00:00+01:00    597700000
                                   ...    
    2022-05-31 00:00:00+02:00    649407200
    2022-06-30 00:00:00+02:00    572380932
    Freq: M, Name: Volume, Length: 66, dtype: int64




```python
df_tesla.Volume.resample('M').sum().plot.line();
```


![plot2.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661636016279/8cq_X9rOx.jpeg align="left")

#### Weekly


```python
df_tesla.Volume.resample('W').sum()
```




    Date
    2017-01-08 00:00:00+01:00    142882000
    2017-01-15 00:00:00+01:00    105867500
                                   ...    
    2022-06-26 00:00:00+02:00    141234200
    2022-07-03 00:00:00+02:00     21237332
    Freq: W-SUN, Name: Volume, Length: 287, dtype: int64




```python
df_tesla.Volume.resample('W').sum().plot.area();
```


![plot3.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661636023542/F5FctQZA8.jpeg align="left")


```python
df_tesla.Volume.resample('W-FRI').sum()
```




    Date
    2017-01-06 00:00:00+01:00    142882000
    2017-01-13 00:00:00+01:00    105867500
                                   ...    
    2022-06-24 00:00:00+02:00    141234200
    2022-07-01 00:00:00+02:00     21237332
    Freq: W-FRI, Name: Volume, Length: 287, dtype: int64




```python
df_tesla.Volume.resample('W-FRI').sum().plot.line();
```


![plot4.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661636029694/Zm06lfhpj.jpeg align="left")

#### Quarterly


```python
df_tesla.Volume.resample('Q').sum()
```




    Date
    2017-03-31 00:00:00+02:00    1636274500
    2017-06-30 00:00:00+02:00    2254740000
                                    ...    
    2022-03-31 00:00:00+02:00    1678802000
    2022-06-30 00:00:00+02:00    1728774732
    Freq: Q-DEC, Name: Volume, Length: 22, dtype: int64




```python
df_tesla.Volume.resample('Q').sum().plot.bar();
```


![plot5.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661636035846/56q0r5KGr.jpeg align="left")

### Pivot Table

We can also use Pivot Tables for summarising and nicer represent the information:


```python
df_res = df_tesla.pivot_table(
    index=df_tesla.index.month,
    columns=df_tesla.index.year,
    values='Volume',
    aggfunc='sum'
)

df_res
```


![df13.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661636042780/74ySUxfST.jpeg align="left")

And even apply some style to get more insight on the DataFrame:


```python
df_tesla['Volume_M'] = df_tesla.Volume/1_000_000

dfres = df_tesla.pivot_table(
    index=df_tesla.index.month,
    columns=df_tesla.index.year,
    values='Volume_M',
    aggfunc='sum'
)

df_stl = dfres.style.format('{:.2f}').background_gradient('Reds', axis=1)
df_stl
```


![df14.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1661636048979/dd2gR-7Ux.jpeg align="left")

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
