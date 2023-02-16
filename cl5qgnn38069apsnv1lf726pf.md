# #01 | Getting Started with Pandas

## Introduction

Programming is all about working with data.

We can work with many types of data structures. Nevertheless, the pandas DataFarme is the most useful because it contains functions that automate a lot of work by writing a simple line of code.

This tutorial will teach you how to work with the `pandas.DataFrame` object.

Before, we will demonstrate why working with simple Arrays (what most people do) makes your life more difficult than it should be.


## The Array

An array is any object that can store **more than one object**. For example, the `list`:


```python
[100, 134, 87, 99]
```


Let's say we are talking about the revenue our e-commerce has had over the last 4 months:


```python
list_revenue = [100, 134, 87, 99]
```

We want to calculate the total revenue (i.e., we sum up the objects within the list):


```python
list_revenue.sum()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Input In [3], in <cell line: 1>()
    ----> 1 list_revenue.sum()


    AttributeError: 'list' object has no attribute 'sum'


The list is a *poor* object which doesn't contain powerful functions.

What can we do then?

We convert the list to a powerful object such as the `Series`, which comes from `pandas` library.


```python
import pandas

pandas.Series(list_revenue)
```



    >>>
    0    100
    1    134
    2     87
    3     99
    dtype: int64




```python
series_revenue = pandas.Series(list_revenue)
```

Now we have a powerful object that can perform the `.sum()`:


```python
series_revenue.sum()
```




    >>> 420



## The Series


![Series.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130167821/gJgrKmRJw.jpg align="left")

Within the Series, we can find more objects.


```python
series_revenue
```



    >>>
    0    100
    1    134
    2     87
    3     99
    dtype: int64



### The index


```python
series_revenue.index
```




    >>> RangeIndex(start=0, stop=4, step=1)



Let's change the elements of the index:


```python
series_revenue.index = ['1st Month', '2nd Month', '3rd Month', '4th Month']
```


```python
series_revenue
```



    >>>
    1st Month    100
    2nd Month    134
    3rd Month     87
    4th Month     99
    dtype: int64



### The values


```python
series_revenue.values
```




    >>> array([100, 134,  87,  99])



### The name


```python
series_revenue.name
```

The `Series` doesn't contain a name. Let's define it:


```python
series_revenue.name = 'Revenue'
```


```python
series_revenue
```



    >>>
    1st Month    100
    2nd Month    134
    3rd Month     87
    4th Month     99
    Name: Revenue, dtype: int64



### The dtype

The values of the Series (right-hand side) are determined by their **data type** (alias `dtype`):


```python
series_revenue.dtype
```




    >>> dtype('float64')



Let's change the values' dtype to be `float` (decimal numbers)


```python
series_revenue.astype(float)
```



    >>>
    1st Month    100.0
    2nd Month    134.0
    3rd Month     87.0
    4th Month     99.0
    Name: Revenue, dtype: float64




```python
series_revenue = series_revenue.astype(float)
```

### Awesome Functions 😎

What else could we do with the Series object?


```python
series_revenue.describe()
```



    >>>
    count      4.000000
    mean     105.000000
    std       20.215506
    min       87.000000
    25%       96.000000
    50%       99.500000
    75%      108.500000
    max      134.000000
    Name: Revenue, dtype: float64




```python
series_revenue.plot.bar();
```


![output_39_0.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130197683/fDX_V69Rr.png align="left")
    



```python
series_revenue.plot.barh();
```


    


![output_40_0.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130209613/-VOM_UofB.png align="left")
    



```python
series_revenue.plot.pie();
```


    
![output_41_0.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130220793/gwnLLDoDa.png align="left")
    


## The DataFrame


![DataFrame.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130310253/2iEojId8M.jpg align="left")

The `DataFrame` is a set of Series.

We will create another Series `series_expenses` to later put them together into a DataFrame.


```python
pandas.Series(
    data=[20, 23, 21, 18],
    index=['1st Month','2nd Month','3rd Month','4th Month'],
    name='Expenses'
)
```



    >>>
    1st Month    20
    2nd Month    23
    3rd Month    21
    4th Month    18
    Name: Expenses, dtype: int64




```python
series_expenses = pandas.Series(
    data=[20, 23, 21, 18],
    index=['1st Month','2nd Month','3rd Month','4th Month'],
    name='Expenses'
)
```


```python
pandas.DataFrame(data=[series_revenue, series_expenses])
```



![df1.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130337423/31hwmh0ZH.png align="left")




```python
df_shop = pandas.DataFrame(data=[series_revenue, series_expenses])
```

Let's transpose the DataFrame to have the variables in columns:


```python
df_shop.transpose()
```




![df2.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130366068/OslkBiwIR.png align="left")




```python
df_shop = df_shop.transpose()
```

### The index


```python
df_shop.index
```




    >>> Index(['1st Month', '2nd Month', '3rd Month', '4th Month'], dtype='object')



### The columns


```python
df_shop.columns
```




    >>> Index(['Revenue', 'Expenses'], dtype='object')



### The values


```python
df_shop.values
```



    >>>
    array([[100.,  20.],
           [134.,  23.],
           [ 87.,  21.],
           [ 99.,  18.]])



### The shape


```python
df_shop.shape
```




    >>> (4, 2)



### Awesome Functions 😎

What else could we do with the DataFrame object?


```python
df_shop.describe()
```





![df3.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130419149/Bs2d0TatY.png align="left")




```python
df_shop.plot.bar();
```


    

![output_63_0.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130428176/V4NIj3PO_.png align="left")
    



```python
df_shop.plot.pie(subplots=True);
```


    

![output_64_0.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130439239/GzljhfADi.png align="left")
    



```python
df_shop.plot.line();
```


    

![output_65_0.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130448260/vdYJRuZTa.png align="left")
    



```python
df_shop.plot.area();
```


    

![output_66_0.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130456286/cnTojLxAy.png align="left")
    


We could also export the DataFrame to formatted data files:


```python
df_shop.to_excel('data.xlsx')
```


```python
df_shop.to_csv('data.csv')
```

## Reading Data Tables from Files

### JSON

#### Football Players

```python
url = 'https://raw.githubusercontent.com/jsulopzs/data/main/football_players_stats.json'
pandas.read_json(url, orient='index')
```



![df4.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130508704/Pi69kM3Qx.png align="left")



```python
df_football = pandas.read_json(url, orient='index')
```


```python
df_football.Goals.plot.pie();
```


    

![output_76_0.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130520389/nmuqnZ1VJ.png align="left")
    


#### Tennis Players


```python
url = 'https://raw.githubusercontent.com/jsulopzs/data/main/best_tennis_players_stats.json'
pandas.read_json(path_or_buf=url, orient='index')
```





![df5.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130558622/N3if1xzjn.png align="left")




```python
df_tennis = pandas.read_json(path_or_buf=url, orient='index')
```


```python
df_tennis.style.background_gradient()
```





![df6.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130588451/03zTDMwNk.png align="left")



```python
df_tennis.plot.pie(subplots=True, layout=(2,3), figsize=(10,6));
```


    

![output_82_0.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130597392/zABCZrrOp.png align="left")
    


### HTML Web Page




```python
pandas.read_html('https://www.skysports.com/la-liga-table/2021', index_col='Team')[0]
```


![df7.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130672939/S-4gjOt7L.png align="left")



```python
df_laliga = pandas.read_html('https://www.skysports.com/la-liga-table/2021', index_col='Team')[0]
```


```python
df_laliga.Pts.plot.barh();
```


    

![output_87_0.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130681450/McD07dhE7.png align="left")
    



```python
df_laliga.Pts.sort_values().plot.barh();
```


![output_88_0.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130689322/WHGyBFAER.png align="left")
    


### CSV


```python
url = 'https://raw.githubusercontent.com/jsulopzs/data/main/internet_usage_spain.csv'
pandas.read_csv(filepath_or_buffer=url)
```



![df8.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130732924/58UWzT1-e.png align="left")




```python
df_internet = pandas.read_csv(filepath_or_buffer=url)
```


```python
df_internet.hist();
```



![output_93_0.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130745186/fASXAVXUv.png align="left")
    



```python
df_internet.pivot_table(index='education', columns='internet_usage', aggfunc='size')
```


![df-pivot.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130772793/CZvqGR9Pq.png align="left")


```python
dfres = df_internet.pivot_table(index='education', columns='internet_usage', aggfunc='size')
```


```python
dfres.style.background_gradient('Greens', axis=1)
```



![dfpivot-color.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1658130804677/cVv6WNypP.png align="left")

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
