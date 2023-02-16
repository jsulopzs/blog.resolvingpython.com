# #04 | Data Visualization in Python

## Possibilities

Look at the following example as an aspiration you can achieve if you fully understand and replicate this whole tutorial with your data.

Let's load a dataset that contains information from countries (rows) considering socio-demographic and economic variables (columns).


```python
import plotly.express as px

df_countries = px.data.gapminder()
df_countries
```


![df1.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450030628/93RuebFN-.jpeg align="center")

Python contains 3 main libraries for Data Visualization:
    
1. **Matplotlib** (Mathematical Plotting)
2. **Seaborn** (High-Level based on Matplotlib)
3. **Plotly** (Animated Plots)

I love `plotly` because the Visualizations are interactive; you may hover the mouse over the points to get information from them:


```python
df_countries_2007 = df_countries.query('year == 2007')

px.scatter(data_frame=df_countries_2007, x='gdpPercap', y='lifeExp',
           color='continent', hover_name='country', size='pop')
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/1.embed"></iframe></center>

You can even animate the plots with a simple parameter. Click on play ↓

PS: The following example is taken from the [official plotly library website](https://plotly.com/python/animations/):


```python
px.scatter(df_countries, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/3.embed"></iframe></center>

In this article, we'll dig into the details of Data Visualization in Python to build up the required knowledge and develop awesome visualizations like the ones we saw before.

## Matplotlib

Matplotlib is a library used for Data Visualization.

We use the **sublibrary** (module) `pyplot` from `matplotlib` library to access the functions.


```python
import matplotlib.pyplot as plt
```

Let's make a bar plot:


```python
plt.bar(x=['Real Madrid', 'Barcelona', 'Bayern Munich'],
       height=[14, 5, 6]);
```


![plot3.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450042002/-BHmEW0R1.jpeg align="center")

We could have also done a point plot:


```python
plt.scatter(x=['Real Madrid', 'Barcelona', 'Bayern Munich'],
            y=[14, 5, 6]);
```


![plot4.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450049027/K9r0_sItv.jpeg align="center")

But it doesn't make sense with the data we have represented.

## Visualize DataFrame

Let's create a DataFrame:


```python
teams = ['Real Madrid', 'Barcelona', 'Bayern Munich']
uefa_champions = [14, 5, 6]

import pandas as pd

df_champions = pd.DataFrame(data={'Team': teams,
                   'UEFA Champions': uefa_champions})
df_champions
```


![df2.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450056486/ZZrHv6YS9.jpeg align="center")

And visualize it using:

### Matplotlib functions


```python
plt.bar(x=df_champions['Team'],
        height=df_champions['UEFA Champions']);
```


![plot5.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450062363/H8tIp55VX.jpeg align="center")

### DataFrame functions


```python
df_champions.plot.bar(x='Team', y='UEFA Champions');
```


![plot6.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450067540/yaWpEAiBo.jpeg align="center")

## Seaborn

Let's read another dataset: the Football Premier League classification for 2021/2022.


```python
df_premier = pd.read_excel(io='../data/premier_league.xlsx')
df_premier
```


![df3.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450074319/K13-WzKIn.jpeg align="center")

We will visualize a point plot, from now own **scatter plot** to check if there is a relationship between the number of goals scored `F` versus the Points `Pts`.


```python
import seaborn as sns

sns.scatterplot(x='F', y='Pts', data=df_premier);
```


![plot7.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450084414/6pEO0Tnz8.jpeg align="center")

Can we do the same plot with matplotlib `plt` library?


```python
plt.scatter(x='F', y='Pts', data=df_premier);
```


![plot8.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450090274/CTVwvSXeW.jpeg align="center")

Which are the differences between them?

1. The points: `matplotlib` points are bigger than `seaborn` ones
2. The axis labels: `matplotlib` axis labels are non-existent, whereas `seaborn` places the names of the columns

From which library do the previous functions return the objects?


```python
seaborn_plot = sns.scatterplot(x='F', y='Pts', data=df_premier);
```


![plot9.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450095630/8TK71W4WW.jpeg align="center")


```python
matplotlib_plot = plt.scatter(x='F', y='Pts', data=df_premier);
```


![plot10.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450100318/72r2ihwNA.jpeg align="center")


```python
type(seaborn_plot)
```




    matplotlib.axes._subplots.AxesSubplot




```python
type(matplotlib_plot)
```




    matplotlib.collections.PathCollection



Why does `seaborn` returns a `matplotlib` object?

Quoted from the [seaborn](https://seaborn.pydata.org/) official website:

> Seaborn is a Python data visualization library **based on matplotlib**. It provides a **high-level\* interface** for drawing attractive and informative statistical graphics.

\*High-level means the communication between humans and the computer is easier to understand than low-level communication, which goes through 0s and 1s.

Could you place the names of the teams in the points?


```python
plt.scatter(x='F', y='Pts', data=df_premier)

for idx, data in df_premier.iterrows():
    plt.text(x=data['F'], y=data['Pts'], s=data['Team'])
```


![plot11.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450109125/lhXGDo_gn.jpeg align="center")

It isn't straightforward.

Is there an easier way?

Yes, you may use an interactive plot with `plotly` library and display the name of the Team as you hover the mouse on a point.

## Plotly

We use the `express` module within `plotly` library to access the functions of the plots:


```python
import plotly.express as px

px.scatter(data_frame=df_premier, x='F', y='Pts', hover_name='Team')
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/6.embed"></iframe></center>

## Types of Plots

Let's read another dataset: the sociological data of clients in a restaurant.


```python
df_tips = sns.load_dataset(name='tips')
df_tips
```


![df4.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450118522/mOoblL0AU.jpeg align="center")

### One Column

#### Categorical Column


```python
df_tips.sex
```




    0      Female
    1        Male
            ...  
    242      Male
    243    Female
    Name: sex, Length: 244, dtype: category
    Categories (2, object): ['Male', 'Female']



We need to summarise the data first; we count how many `Female` and `Male` people are in the dataset.


```python
df_tips.sex.value_counts()
```




    Male      157
    Female     87
    Name: sex, dtype: int64




```python
sr_sex = df_tips.sex.value_counts()
```

##### Barplot

Let's place bars equal to the number of people from each gender:


```python
px.bar(x=sr_sex.index, y=sr_sex.values)
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/8.embed"></iframe></center>

We can also colour the bars based on the category:


```python
px.bar(x=sr_sex.index, y=sr_sex.values, color=sr_sex.index)
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/10.embed"></iframe></center>

##### Pie plot

Let's put the same data into a pie plot:


```python
px.pie(names=sr_sex.index, values=sr_sex.values, color=sr_sex.index)
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/12.embed"></iframe></center>

#### Numerical Column


```python
df_tips.total_bill
```




    0      16.99
    1      10.34
           ...  
    242    17.82
    243    18.78
    Name: total_bill, Length: 244, dtype: float64



##### Histogram

Instead of observing the numbers, we can visualize the distribution of the bills in a **histogram**.

We can observe that most people pay between 10 and 20 dollars. Whereas a few are between 40 and 50.


```python
px.histogram(x=df_tips.total_bill)
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/14.embed"></iframe></center>

We can also create a **boxplot** where the limits of the boxes indicate the 1st and 3rd quartiles.

The 1st quartile is 13.325, and the 3rd quartile is 24.175. Therefore, 50% of people were billed an amount between these limits.

##### Boxplot


```python
px.box(x=df_tips.total_bill)
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/16.embed"></iframe></center>

### Two Columns


```python
df_tips[['total_bill', 'tip']]
```


![df5.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450139175/3HOR3PCVa.jpeg align="center")

#### Numerical & Numerical

We use a scatter plot to see if a relationship exists between two numerical variables.

Do the points go up as you move the eyes from left to right?

As you may observe in the following plot: the higher the amount of the bill, the higher the tip the clients leave for the staff.


```python
px.scatter(x='total_bill', y='tip', data_frame=df_tips)
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/18.embed"></iframe></center>

Another type of visualization for 2 continuous variables:


```python
px.density_contour(x='total_bill', y='tip', data_frame=df_tips)
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/20.embed"></iframe></center>

#### Numerical & Categorical


```python
df_tips[['day', 'total_bill']]
```


![df6.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450147573/Jlfy9YY8T.jpeg align="center")

We can summarise the data around how much revenue was generated in each day of the week.


```python
df_tips.groupby('day').total_bill.sum()
```




    day
    Thur    1096.33
    Fri      325.88
    Sat     1778.40
    Sun     1627.16
    Name: total_bill, dtype: float64




```python
sr_days = df_tips.groupby('day').total_bill.sum()
```

We can observe that Saturday is the most profitable day as people have spent more money.


```python
px.bar(x=sr_days.index, y=sr_days.values)
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/22.embed"></iframe></center>


```python
px.bar(x=sr_days.index, y=sr_days.values, color=sr_days.index)
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/24.embed"></iframe></center>

#### Categorical & Categorical


```python
df_tips[['day', 'size']]
```


![df7.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1659450157108/3O2gIAOJf.jpeg align="center")

Which combination of day-size is the most frequent table you can observe in the restaurant?

The following plot shows that Saturdays with 2 people at the table is the most common phenomenon at the restaurant.

They could create an advertisement that targets couples to have dinner on Saturdays and make more money.


```python
px.density_heatmap(x='day', y='size', data_frame=df_tips)
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/26.embed"></iframe></center>

## Awesome Plots

The following examples are taken directly from [plotly](https://plotly.com/python/).


```python
df_gapminder = px.data.gapminder()
px.scatter_geo(df_gapminder, locations="iso_alpha", color="continent", #!
                     hover_name="country", size="pop",
                     animation_frame="year",
                     projection="natural earth")
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/28.embed"></iframe></center>


```python
import plotly.express as px

df = px.data.election()
geojson = px.data.election_geojson()

fig = px.choropleth_mapbox(df, geojson=geojson, color="Bergeron",
                           locations="district", featureidkey="properties.district",
                           center={"lat": 45.5517, "lon": -73.7073},
                           mapbox_style="carto-positron", zoom=9)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/31.embed"></iframe></center>


```python
import plotly.express as px

df = px.data.election()
geojson = px.data.election_geojson()

fig = px.choropleth_mapbox(df, geojson=geojson, color="winner",
                           locations="district", featureidkey="properties.district",
                           center={"lat": 45.5517, "lon": -73.7073},
                           mapbox_style="carto-positron", zoom=9)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
```

<center><iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~jsulopzs/33.embed"></iframe></center>
