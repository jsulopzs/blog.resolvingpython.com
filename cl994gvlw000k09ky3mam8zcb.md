# #06 | Locating & Filtering the pandas.DataFrame

## Possibilities

Sometimes, we want to select specific parts of the DataFrame to highlight some data points.

In this case, we refer to the topic as locating & filtering.

For example, let's load the dataset of cars:


```python
import seaborn as sns

df_mpg = sns.load_dataset('mpg', index_col='name').drop(columns=['cylinders', 'model_year', 'origin'])
df_mpg
```


![df1.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242858938/kVRSkcQTa.jpeg align="center")

To filter the best cars in each statistics/column.

First, we calculate the maximum values in each column:


```python
df_mpg.max()
```




    mpg               46.6
    displacement     455.0
    horsepower       230.0
    weight          5140.0
    acceleration      24.8
    dtype: float64



Then, we create a mask (array with True/False) to capture the rows where we have the cars with maximum values:


```python
mask_max = (df_mpg == df_mpg.max()).sum(axis=1) > 0
mask_max
```




    name
    chevrolet chevelle malibu    False
    buick skylark 320            False
                                 ...  
    ford ranger                  False
    chevy s-10                   False
    Length: 398, dtype: bool



Select the rows where the mask is True:


```python
df_mpg_max = df_mpg[mask_max].copy()
df_mpg_max
```


![df2.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242866415/S22qtaVCD.jpeg align="center")

And add some styling:


```python
df_mpg_max.style.format('{:.0f}').background_gradient()
```


![df3.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242872112/pvk4DYa33.jpeg align="center")

To understand the reasoning behind the previous example, read the rest of the article, where we explain the logic from the most basic example to locating data based on the index.

## Any Object

By now, we should know the difference between the brackets `[]` and the parenthesis `()`.

We use brackets to select parts of an object. For example, let's create a list of days:


```python
list_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
```

And select the second element:


```python
list_days[1]
```




    'Tuesday'



Or the last element:


```python
list_days[-1]
```




    'Sunday'



Until the third element (included):


```python
list_days[:3]
```




    ['Monday', 'Tuesday', 'Wednesday']



Nevertheless, the `list` is a simple element of Python. To get more functionalities, we use the `Series` object from `pandas` library.

## Series

Let's create a `Series` to store the **Apple Stock Return on Investment (ROI)** by quarters:


```python
import pandas as pd

sr_apple = pd.Series(
    data=[59.02, 63.57, 66.93, 69.05],
    index=['1Q', '2Q', '3Q', '4Q']
)

sr_apple
```




    1Q    59.02
    2Q    63.57
    3Q    66.93
    4Q    69.05
    dtype: float64



### `iloc` (integer-location) property

We use `.iloc[]` to select parts of the object based on the integer position of the element.

For example, let's select the first quarter ROI:


```python
sr_apple.iloc[0]
```




    59.02



Now, let's select the first and third quarters:

To select more than one object, we need to use double brackets `[[]]`:


```python
sr_apple.iloc[[0,2,3]]
```




    1Q    59.02
    3Q    66.93
    4Q    69.05
    dtype: float64



Could we have accessed with the name `1Q`?


```python
sr_apple.iloc['Q1']
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Input In [99], in <cell line: 1>()
    ----> 1 sr_apple.iloc['Q1']


    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexing.py:967, in _LocationIndexer.__getitem__(self, key)
        964 axis = self.axis or 0
        966 maybe_callable = com.apply_if_callable(key, self.obj)
    --> 967 return self._getitem_axis(maybe_callable, axis=axis)


    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexing.py:1517, in _iLocIndexer._getitem_axis(self, key, axis)
       1515 key = item_from_zerodim(key)
       1516 if not is_integer(key):
    -> 1517     raise TypeError("Cannot index by location index with a non-integer key")
       1519 # validate the location
       1520 self._validate_integer(key, axis)


    TypeError: Cannot index by location index with a non-integer key


The `iloc` property only works in `integers` (the position of the subelements we want).

To select the elements by their **label/name**, we need to use the `loc` property:

### `loc` (location) property

We select parts of an object with the `.loc[]` instance based on the **label/name** of the `index`:


```python
sr_apple.loc['1Q']
```




    59.02




```python
sr_apple.loc[['1Q', '3Q', '4Q']]
```




    1Q    59.02
    3Q    66.93
    4Q    69.05
    dtype: float64



If we would like to access by the position, we'd get an error:


```python
sr_apple.loc[0]
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexes/base.py:3621, in Index.get_loc(self, key, method, tolerance)
       3620 try:
    -> 3621     return self._engine.get_loc(casted_key)
       3622 except KeyError as err:


    File ~/miniforge3/lib/python3.9/site-packages/pandas/_libs/index.pyx:136, in pandas._libs.index.IndexEngine.get_loc()


    File ~/miniforge3/lib/python3.9/site-packages/pandas/_libs/index.pyx:163, in pandas._libs.index.IndexEngine.get_loc()


    File pandas/_libs/hashtable_class_helper.pxi:5198, in pandas._libs.hashtable.PyObjectHashTable.get_item()


    File pandas/_libs/hashtable_class_helper.pxi:5206, in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 0

    
    The above exception was the direct cause of the following exception:


    KeyError                                  Traceback (most recent call last)

    Input In [102], in <cell line: 1>()
    ----> 1 sr_apple.loc[0]


    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexing.py:967, in _LocationIndexer.__getitem__(self, key)
        964 axis = self.axis or 0
        966 maybe_callable = com.apply_if_callable(key, self.obj)
    --> 967 return self._getitem_axis(maybe_callable, axis=axis)


    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexing.py:1202, in _LocIndexer._getitem_axis(self, key, axis)
       1200 # fall thru to straight lookup
       1201 self._validate_key(key, axis)
    -> 1202 return self._get_label(key, axis=axis)


    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexing.py:1153, in _LocIndexer._get_label(self, label, axis)
       1151 def _get_label(self, label, axis: int):
       1152     # GH#5667 this will fail if the label is not present in the axis.
    -> 1153     return self.obj.xs(label, axis=axis)


    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/generic.py:3864, in NDFrame.xs(self, key, axis, level, drop_level)
       3862             new_index = index[loc]
       3863 else:
    -> 3864     loc = index.get_loc(key)
       3866     if isinstance(loc, np.ndarray):
       3867         if loc.dtype == np.bool_:


    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexes/base.py:3623, in Index.get_loc(self, key, method, tolerance)
       3621     return self._engine.get_loc(casted_key)
       3622 except KeyError as err:
    -> 3623     raise KeyError(key) from err
       3624 except TypeError:
       3625     # If we have a listlike key, _check_indexing_error will raise
       3626     #  InvalidIndexError. Otherwise we fall through and re-raise
       3627     #  the TypeError.
       3628     self._check_indexing_error(key)


    KeyError: 0


It results in `KeyError` because we don't have any `Key` in the `index` to be `0`:


```python
sr_apple
```




    1Q    59.02
    2Q    63.57
    3Q    66.93
    4Q    69.05
    dtype: float64



We have:


```python
sr_apple.keys()
```




    Index(['1Q', '2Q', '3Q', '4Q'], dtype='object')



The `loc` property only works **with the labels, not the position**.

### Masking with boolean objects

Now we'd like to select parts based on a condition. For example, let's show the quarters we had a Return on Investment (ROI) above 60.

First, we create a boolean object based on the stated condition:


```python
sr_apple
```




    1Q    59.02
    2Q    63.57
    3Q    66.93
    4Q    69.05
    dtype: float64




```python
sr_apple > 60
```




    1Q    False
    2Q     True
    3Q     True
    4Q     True
    dtype: bool




```python
mask_60 = sr_apple > 60
```

Now we pass the previous object to the `.loc` property:


```python
sr_apple.loc[mask_60]
```




    2Q    63.57
    3Q    66.93
    4Q    69.05
    dtype: float64



And here, we have the data for which the ROI is higher than 60.

### Just the brackets `[]`


```python
sr_apple
```




    1Q    59.02
    2Q    63.57
    3Q    66.93
    4Q    69.05
    dtype: float64



We could also access the data by only using the brackets, without the ~`.iloc`~ property:


```python
sr_apple['1Q']
```




    59.02



And also, the position:


```python
sr_apple[0]
```




    59.02



And the mask:


```python
sr_apple[mask_60]
```




    2Q    63.57
    3Q    66.93
    4Q    69.05
    dtype: float64



So far, we have played with **1-Dimensional** objects. Now it's time to level up and play with **2-Dimensional** objects, like the `DataFrame`.

## DataFrame

Let's play with a dataset of cars:


```python
import seaborn as sns

df_mpg = sns.load_dataset(name='mpg', index_col='name')
df_mpg
```


![df4.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242905084/lm1Zf6mUv.jpeg align="center")

### `iloc` (integer-location) property

We can select the second row:


```python
df_mpg.iloc[2]
```




    mpg              18.0
    cylinders           8
    displacement    318.0
    horsepower      150.0
    weight           3436
    acceleration     11.0
    model_year         70
    origin            usa
    Name: plymouth satellite, dtype: object



And keep the `DataFrame` style if we use double brackets `[[]]`:


```python
df_mpg.iloc[[2]]
```


![df5.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242915085/gWo-03aQu.jpeg align="center")

We can also **slice** (a term used for filtering as well) consecutive elements of the DataFrame with the **colon** `:`.

For example, let's select the first 4 rows:


```python
df_mpg.iloc[:4]
```


![df6.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242920726/Z2gRWb-nh.jpeg align="center")

Instead of:


```python
df_mpg.iloc[[0,1,2,3]]
```


![df7.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242927156/g_Wt1BsmT.jpeg align="center")

We can also select the columns we want.

For example, let's select the first 3 columns:


```python
df_mpg.iloc[:4, :3]
```


![df8.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242936031/UkQnsg2R5.jpeg align="center")

Or the rest of the columns from the 3rd position (not included):


```python
df_mpg.iloc[:4, 3:]
```


![df9.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242943359/ly2u0LhPj.jpeg align="center")

Or the last 3 columns by using the `-`:


```python
df_mpg.iloc[:4, -3:]
```


![df10.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242948596/BopZGok5_.jpeg align="center")

### `loc` (location) property

We can also select parts of the DataFrame based on the **index and column labels** (2-Dimensions):


```python
df_mpg.loc[['ford torino', 'fiat 124 sport coupe'], ['origin', 'model_year', 'cylinders']]
```


![df11.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242955429/Yo32-DD2W.jpeg align="center")


```python
df_mpg.loc[:'fiat 124 sport coupe', :'cylinders']
```


![df12.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242961338/VJh3IDCPF.jpeg align="center")

### Masking with boolean objects

#### Single Condition

Out of all the cars:


```python
df_mpg.index
```




    Index(['chevrolet chevelle malibu', 'buick skylark 320', 'plymouth satellite',
           'amc rebel sst', 'ford torino', 'ford galaxie 500', 'chevrolet impala',
           'plymouth fury iii', 'pontiac catalina', 'amc ambassador dpl',
           ...
           'chrysler lebaron medallion', 'ford granada l', 'toyota celica gt',
           'dodge charger 2.2', 'chevrolet camaro', 'ford mustang gl', 'vw pickup',
           'dodge rampage', 'ford ranger', 'chevy s-10'],
          dtype='object', name='name', length=398)



We could select all the **fiat** cars if we had a boolean array based on this condition:


```python
mask_fiat = df_mpg.index.str.contains('fiat')
mask_fiat
```




    array([False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False,  True, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False,  True, False, False,
            True, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False,  True,  True, False, False,  True, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False,  True, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False,  True, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False])



We can observe a few `True`s where we find some **Fiats**.

Let's filter them and show all the columns with the `:`:


```python
df_mpg.loc[mask_fiat, :]
```


![df13.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242972234/eG9NlboFS.jpeg align="center")

Although we could have omitted the `:`:


```python
df_mpg.loc[mask_fiat]
```


![df14.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242977974/eAQw89jxq.jpeg align="center")

#### Multiple Conditions

##### Both Conditions `&`

Just the fiats whose horsepower is above 80:


```python
mask_hp = df_mpg.horsepower > 80
mask_hp
```




    name
    chevrolet chevelle malibu     True
    buick skylark 320             True
                                 ...  
    ford ranger                  False
    chevy s-10                    True
    Name: horsepower, Length: 398, dtype: bool




```python
df_mpg.loc[mask_hp & mask_fiat, :]
```


![df15.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242984392/yx2tcbNya.jpeg align="center")

##### Any Condition `|`

We could also select all fiats **OR** cars whose horsepower is above 80:


```python
df_mpg.loc[mask_hp | mask_fiat, :]
```


![df16.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242990036/Pclz-NMHr.jpeg align="center")

### Just the brackets `[]`

We can select the columns by their labels:


```python
df_mpg['acceleration']
```




    name
    chevrolet chevelle malibu    12.0
    buick skylark 320            11.5
                                 ... 
    ford ranger                  18.6
    chevy s-10                   19.4
    Name: acceleration, Length: 398, dtype: float64




```python
df_mpg[['acceleration', 'origin', 'model_year']]
```


![df17.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663242995934/p40STxCRL.jpeg align="center")

But we can't select the rows by the index labels:


```python
df_mpg['amc rebel sst']
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexes/base.py:3621, in Index.get_loc(self, key, method, tolerance)
       3620 try:
    -> 3621     return self._engine.get_loc(casted_key)
       3622 except KeyError as err:

    ...

    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexes/base.py:3623, in Index.get_loc(self, key, method, tolerance)
       3621     return self._engine.get_loc(casted_key)
       3622 except KeyError as err:
    -> 3623     raise KeyError(key) from err
       3624 except TypeError:
       3625     # If we have a listlike key, _check_indexing_error will raise
       3626     #  InvalidIndexError. Otherwise we fall through and re-raise
       3627     #  the TypeError.
       3628     self._check_indexing_error(key)


    KeyError: 'amc rebel sst'


Unless we use the colon `:`:


```python
df_mpg[:'amc rebel sst']
```


![df18.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243006328/Mv2oTkLB9.jpeg align="center")


```python
df_mpg['buick skylark 320':'amc rebel sst']
```


![df19.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243012134/4UbCDYrg-.jpeg align="center")

We can also select the rows by position:


```python
df_mpg[:4]
```


![df20.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243017819/U0KFM4p00.jpeg align="center")

But we can't select both rows and columns (2-Dimensions):


```python
df_mpg[:4,:3]
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexes/base.py:3621, in Index.get_loc(self, key, method, tolerance)
       3620 try:
    -> 3621     return self._engine.get_loc(casted_key)
       3622 except KeyError as err:

    ...

    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexes/base.py:5637, in Index._check_indexing_error(self, key)
       5633 def _check_indexing_error(self, key):
       5634     if not is_scalar(key):
       5635         # if key is not a scalar, directly raise an error (the code below
       5636         # would convert to numpy arrays and raise later any way) - GH29926
    -> 5637         raise InvalidIndexError(key)


    InvalidIndexError: (slice(None, 4, None), slice(None, 3, None))


Unless we specify the columns we want in extra brackets:


```python
df_mpg[:4]['acceleration']
```




    name
    chevrolet chevelle malibu    12.0
    buick skylark 320            11.5
    plymouth satellite           11.0
    amc rebel sst                12.0
    Name: acceleration, dtype: float64




```python
df_mpg[:4][['acceleration']]
```


![df21.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243030001/UP16TlOrD.jpeg align="center")


```python
df_mpg[:4][['acceleration', 'origin']]
```


![df22.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243035640/rrVwweznp.jpeg align="center")

We can also select the rows given *boolean-arrays* (a.k.a. **masks**):


```python
df_mpg[mask_fiat]
```


![df23.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243040557/aawo41pFb.jpeg align="center")


```python
df_mpg[mask_fiat | mask_hp]
```


![df24.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243045251/tgIRwCj_P.jpeg align="center")


```python
df_mpg[mask_fiat & mask_hp]
```


![df25.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243050836/tp7-7szzS.jpeg align="center")

It doesn't mean that I cannot later select the columns that we want (programming is the art of everything, we just need to find a way):


```python
df_mpg[mask_fiat & mask_hp]['mpg']
```




    name
    fiat 124 sport coupe    26.0
    fiat 131                28.0
    Name: mpg, dtype: float64




```python
df_mpg[mask_fiat & mask_hp][['mpg', 'origin', 'model_year']]
```


![df26.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243057319/Unb2oOQJ8.jpeg align="center")

Everything may be a bit confusing, but we hope you get the main idea behind `locating` and `masking`:

1. Select the parts of an object with brackets `[]`
2. We can access it through
    1. The label/name `loc`
    2. The integer position `iloc`
    3. Masks: *boolean arrays* based on conditions
    4. Just the brackets `[]`*
3. If the object has:
    1. 1-Dimension `object[:]`
    2. 2-Dimension `object[:,:]`
    
*Carefully because it has many variations of use case, as we observed above

## DataFrame MultiIndex

Let's load a dataset with various categorical columns since we summarise data based on categories, not numbers.


```python
df_tips = sns.load_dataset(name='tips')
df_tips
```


![df27.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243064006/Kvn_X9F7C.jpeg align="center")

Let's make a pivot table to summarise the information to obtain a Hierarchical* DataFrame.


```python
dfres = df_tips.pivot_table(index=['smoker', 'time'], columns='sex', aggfunc='size')
dfres
```


![df28.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243070756/TJ4G6E0qM5.jpeg align="center")

*A Hierarchical DataFrame (MultiIndex) contains two "columns" as an index. As we may observe below:


```python
dfres.index
```




    MultiIndex([('Yes',  'Lunch'),
                ('Yes', 'Dinner'),
                ( 'No',  'Lunch'),
                ( 'No', 'Dinner')],
               names=['smoker', 'time'])



### First Index

Let's locate some parts of the Hierarchical DataFrame:


```python
dfres
```


![df29.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243077238/CikC3m7pU.jpeg align="center")

By using the `.loc` property:


```python
dfres.loc['Yes', :]
```


![df30.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243085445/xLH-S8Im5.jpeg align="center")


```python
dfres.loc['No', :]
```


![df31.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243092270/IHI5TZNts.jpeg align="center")

### Second Index

As we have multiple indexes `[index1, index2, columns]`, we can select a part of the second index:


```python
dfres.loc[:, 'Lunch', :]
```


![df32.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243099601/cEQWkQjBt.jpeg align="center")


```python
dfres.loc[:, 'Dinner', :]
```


![df33.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243105359/hruKA_Dbo.jpeg align="center")

## DataFrame MultiIndex & MultiColumns

Let's now play with a DataFrame that is both `MultiIndex` and `MultiColumns`:


```python
dfres = df_tips.pivot_table(index=['smoker', 'time'], columns=['sex', 'day'], aggfunc='size')
dfres
```


![df34.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243110096/IbFaezGfB.jpeg align="center")

We may observe two levels in the columns above.

### `loc` (location) property

#### First Index

We apply the same reasoning we used in the previous sections, `[index1, index2, column1, column2]`.


```python
dfres.loc['No', :, :, :]
```


![df35.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243115224/0cmK2379w.jpeg align="center")

Although, we can make it shorter.


```python
dfres.loc['No', :]
```


![df36.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243121663/8lSzfriU1.jpeg align="center")

#### Second Index

The same applies to the second index:


```python
dfres.loc[:,'Dinner', :, :]
```


![df37.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243127925/geXSrsQvG.jpeg align="center")


```python
dfres.loc[:,'Dinner', :]
```


![df38.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243132694/gsKIvc5nF.jpeg align="center")

#### Second Index & Second Column

Let's try to get Dinners on Sundays:


```python
dfres.loc[:, 'Dinner', :, 'Sun']
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Input In [158], in <cell line: 1>()
    ----> 1 dfres.loc[:, 'Dinner', :, 'Sun']


    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexing.py:961, in _LocationIndexer.__getitem__(self, key)
        959     if self._is_scalar_access(key):
        960         return self.obj._get_value(*key, takeable=self._takeable)
    --> 961     return self._getitem_tuple(key)
        962 else:
        963     # we by definition only have the 0th axis
        964     axis = self.axis or 0

    ...

    File ~/miniforge3/lib/python3.9/site-packages/pandas/core/indexes/frozen.py:70, in FrozenList.__getitem__(self, n)
         68 if isinstance(n, slice):
         69     return type(self)(super().__getitem__(n))
    ---> 70 return super().__getitem__(n)


    IndexError: list index out of range


To make it work, this time we need to create an intermediate object to separate rows and columns:


```python
idx = pd.IndexSlice

dfres.loc[idx[:, 'Dinner'], idx[:, 'Sun']]
```


![df39.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243145961/ja5Qyx1YY.jpeg align="center")

#### Second Index & First Column


```python
dfres.loc[idx[:, 'Dinner'], idx['Male', :]]
```


![df40.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243152144/HduuyPxiU.jpeg align="center")

#### Using the Slice

We can also use the `slice()` property:


```python
dfres.loc[('Yes', slice(None)), (slice(None), 'Sun')]
```


![df41.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243157274/bqMGCkbGj.jpeg align="center")


```python
dfres.loc['Yes', ('Female', slice(None))]
```


![df42.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243167379/7kbFBhJQ6.jpeg align="center")


```python
dfres.loc[(slice(None), 'Lunch'), 'Female']
```


![df43.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243172547/KoETv7uU3.jpeg align="center")


```python
dfres.loc[(slice(None), 'Lunch'), ('Female', slice(None))]
```


![df44.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243178071/txZs7kCLr.jpeg align="center")


```python
dfres.loc[idx[:, 'Dinner'], idx['Female', :]]
```


![df45.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243182485/F0bVhGryg.jpeg align="center")

### `iloc` (integer-location) property


```python
dfres
```


![df46.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243187339/T8_vECBKo.jpeg align="center")

As always, we can select by the position of the values with the `iloc` property:


```python
dfres.iloc[:2, :2]
```


![df47.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243193414/gUyYR9MD3.jpeg align="center")


```python
dfres.iloc[:2, 2:]
```


![df48.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243199727/wcthMToiB.jpeg align="center")

## DataFrame with DateTimeIndex

Now, we will use a DataFrame that has a `DateTimeIndex`:


```python
df_tsla = pd.read_excel('tsla_stock.xlsx', index_col=0)
df_tsla
```


![df49.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243213359/KE7urMGZu.jpeg align="center")

### `loc` (location) property

We can select parts of the DataFrame based on just one part of the `DateTimeIndex`. For example, we can select everything from the year 2020 and move forward:


```python
df_tsla.loc['2020':]
```


![df50.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243224375/fqFyu8IHd.jpeg align="center")

Until the last day of 2020:


```python
df_tsla.loc[:'2020']
```


![df51.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243237718/u-b-4SyjS.jpeg align="center")

Between two years:


```python
df_tsla.loc['2019':'2020']
```


![df52.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243244884/vE1eGEXBX.jpeg align="center")

One complete year:


```python
df_tsla.loc['2019']
```


![df53.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243251120/cn-WHzwMd.jpeg align="center")

We can even select a specific `year-month`:


```python
df_tsla.loc['2019-06']
```


![df54.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243256623/47yhq4QrO.jpeg align="center")

### `iloc` (integer-location) property

Of course, we can also select parts of the DataFrame based on the position of the values with `iloc`:


```python
df_tsla.iloc[:4, :3]
```


![df55.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243263761/FRYgrEi-v.jpeg align="center")


```python
df_tsla.iloc[-4:, :3]
```


![df56.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1663243269996/4bQltBarv.jpeg align="center")
