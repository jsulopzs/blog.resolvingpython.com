# #03 | Grouping & Pivot Tables

**© Jesús López 2022**

Ask him any doubt on **[Twitter](https://twitter.com/jsulopz)** or **[LinkedIn](https://linkedin.com/in/jsulopz)**

## Possibilities

Look at the following example as an aspiration you can achieve if you fully understand and replicate this whole tutorial with your data.

Let's load a dataset that contains information from transactions in tables (rows) at a restaurant considering socio-demographic and economic variables (columns).


```python
import seaborn as sns

df_tips = sns.load_dataset('tips')
df_tips
```

![df1.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658498755491/76phPG1hu.jpeg align="left")

Grouping data to summarise the information helps you identify conclusions. For example, the summary below shows that **Dinners on Sundays** come to the best customers because they:
1. Spend more on average (\$21.41)
2. Give more tips on average (\$3.25)
3. Come more people at the same table on average (\$2.84)


```python
df_tips.groupby(by=['day', 'time'])\
    .mean()\
    .fillna(0)\
    .style.format('{:.2f}').background_gradient(axis=0)
```

![df2.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658498763890/3c5WdW4t1c.jpeg align="left")


```python
df_tips.groupby(by=['day', 'time'])\
    .mean()\
    .fillna(0)\
    .style.format('{:.2f}').bar(axis=0, width=50, align='zero')
```

![df3.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658498768042/MpITQOirf.jpeg align="left")

Let's dig into the details of the `.groupby()` function from the basics in the following sections.

## Grouping by 1 Column

We use the `.groupby()` function to generate an object that contains as many `DataFrames` as categories are in the column.


```python
df_tips.groupby('sex')
```

As we have two groups in sex (Female and Male), the length of the `DataFrameGroupBy` object returned by the `groupby()` function is 2:


```python
len(df_tips.groupby('sex'))
```

How can we work with the object `DataFrameGroupBy`?

### Calculate the Average for All Columns

We use the `.mean()` function to get the average of the numerical columns for the two groups:


```python
df_tips.groupby('sex').mean()
```

![df4.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658498777322/ZZHPDrMhL.jpeg align="left")

A pretty and simple syntax to summarise the information, right?

- But what's going on inside the `DataFrameGroupBy` object?


```python
df_tips.groupby('sex')
```


```python
df_grouped = df_tips.groupby('sex')
```

The `DataFrameGroupBy` object contains 2 `DataFrames`. To see one of them `DataFrame` you need to use the function `.get_group()` and pass the group whose `DataFrame` you'd like to return:


```python
df_grouped.get_group('Male')
```

![df5.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658498791187/aTPOzlKJI.jpeg align="left")


```python
df_grouped.get_group('Female')
```

![df6.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658498794103/MWL6OD8ZJ.jpeg align="left")

As the `DataFrameGroupBy` distinguish the categories, at the moment we apply an aggregation function (click [here](https://sparkbyexamples.com/pandas/pandas-aggregate-functions-with-examples/) to see a list of them), we will get the mathematical operations for those groups separately:


```python
df_grouped.mean()
```

![df7.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658498796876/0xGgZAj4f.jpeg align="left")

We could apply the function to each `DataFrame` separately. Although *it is not the point of the `.groupby()` function*.


```python
df_grouped.get_group('Male').mean(numeric_only=True)
```


```python
df_grouped.get_group('Female').mean(numeric_only=True)
```

### Compute Functions to 1 Column

To get the results for just 1 column of interest, we access the column:


```python
df_grouped.total_bill
```

And use the aggregation function we wish, `.sum()` in this case:


```python
df_grouped.total_bill.sum()
```

We get the result for just 1 column (total_bill) because the `DataFrames` generated at `.groupby()` are accessed as if they were simple `DataFrames`:


```python
df_grouped.get_group('Female')
```

![df8.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658498800985/8h5m3MZh4.jpeg align="left")


```python
df_grouped.get_group('Female').total_bill
```


```python
df_grouped.get_group('Female').total_bill.sum()
```


```python
df_grouped.get_group('Male').total_bill.sum()
```


```python
df_grouped.total_bill.sum()
```

## Grouping by 2 Columns

So far, we have summarised the data based on the categories of just one column. But, what if we'd like to summarise the data **based on the combinations** of the categories within different categorical columns?

### Compute 1 Function


```python
df_tips.groupby(by=['day', 'smoker']).sum()
```

![df9.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658498804351/ZOX2POdko.jpeg align="left")

### Pivot Tables

We could have also used another function `.pivot_table()` to get the same numbers:


```python
df_tips.pivot_table(index='day', columns='smoker', aggfunc='sum')
```

![df10.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499257688/6CjUgXhoA.jpeg align="left")

Which one is best?

- I leave it up to your choice; I'd prefer to use the `.pivot_table()` because the syntax makes it more accessible.

### Compute More than 1 Function

The thing doesn't stop here; we can even compute different aggregation functions at the same time:

#### Groupby


```python
df_tips.groupby(by=['day', 'smoker'])\
    .total_bill\
    .agg(func=['sum', 'mean'])
```

![df11.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499260925/FUR6O3KeT.jpeg align="left")

#### Pivot Table


```python
df_tips.pivot_table(index='day', columns='smoker',
                    values='total_bill', aggfunc=['sum', 'mean'])
```

![df12.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499263208/E0CjHslzW.jpeg align="left")


```python
dfres = df_tips.pivot_table(index='day', columns='smoker',
                    values='total_bill', aggfunc=['sum', 'mean'])
```

You could even style the output `DataFrame`:


```python
dfres.style.background_gradient()
```

![df13.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499267314/eyHR4Lzo6.jpeg align="left")

For me, it's nicer than styling the `.groupby()` returned DataFrame.

As we say in Spain:

> Pa' gustos los colores!

```python
df_tips.groupby(by=['day', 'smoker']).total_bill.agg(func=['sum', 'mean'])
```

![df14.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499273013/ZwQvqRmd1.jpeg align="left")


```python
dfres = df_tips.groupby(by=['day', 'smoker']).total_bill.agg(func=['sum', 'mean'])
```


```python
dfres.style.background_gradient()
```

![df15.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499277996/y52E_874p.jpeg align="left")

## Pivot Tables in Depth

We can compute more than one mathematical operation:


```python
df_tips.pivot_table(index='sex', columns='time',
                    aggfunc=['sum', 'mean'], values='total_bill')
```

![df16.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499280706/ML5Dr8DGj.jpeg align="left")

And use more than one column in each of the parameters:


```python
df_tips.pivot_table(index='sex', columns='time',
                    aggfunc=['sum', 'mean'], values=['total_bill', 'tip'])
```

![df17.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499283427/xXX1KBjYa.jpeg align="left")


```python
df_tips.pivot_table(index=['day', 'smoker'], columns='time',
                    aggfunc=['sum', 'mean'], values=['total_bill', 'tip'])
```

![df18.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499287364/tvG8SrcSr.jpeg align="left")


```python
df_tips.pivot_table(index=['day', 'smoker'], columns=['time', 'sex'],
                    aggfunc=['sum', 'mean'], values=['total_bill', 'tip'])
```

![df19.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499290072/FPnOTr8Yk.jpeg align="left")

## The `.size()` Function

### `.groupby()`

#### 1 Variable to Group By

The `.size()` is a function used to count the number of rows (observations) in each of the `DataFrames` generated by `.groupby()`.


```python
df_grouped.size()
```

#### 2 Variables to Group By


```python
df_tips.groupby(by=['sex', 'time']).size()
```

### `.pivot_table()`

We can use `.pivot_table()` to represent the data clearer:


```python
df_tips.pivot_table(index='sex', columns='time', aggfunc='size')
```

![df20.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499293301/xIFXpG4SZ.jpeg align="left")

#### Other Example 1


```python
df_tips.pivot_table(index='smoker', columns=['day', 'sex'],aggfunc='size')
```

![df21.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499299318/SGejBvdNi.jpeg align="left")


```python
dfres = df_tips.pivot_table(index='smoker', columns=['day', 'sex'], aggfunc='size')
```


```python
dfres.style.background_gradient()
```

![df22.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499301554/S6stXsRw-.jpeg align="left")

#### Other Example 2


```python
df_tips.pivot_table(index=['day', 'time'], columns=['smoker', 'sex'], aggfunc='size')
```

![df23.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499303868/FwN8aTM2b.jpeg align="left")


```python
dfres = df_tips.pivot_table(index=['day', 'time'], columns=['smoker', 'sex'], aggfunc='size')
```


```python
dfres.style.background_gradient()
```

![df24.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499306249/hOefQXNI4.jpeg align="left")

We can even choose the way we'd like to gradient colour the cells:
- `axis=1`: the upper value between the columns of the same row 
- `axis=2`: the upper value between the rows of the same column


```python
dfres.style.background_gradient(axis=1)
```

![df25.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1658499308657/fwBm7IVeJ.jpeg align="left")

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
