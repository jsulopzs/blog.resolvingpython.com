# You Need to Have Flexible Thinking when Programming

If you are trying to solve a problem with programming, you may have several solutions to get the same result.

A basic idea that we don't get at the beginning because we look for that perfect solution.

It doesn’t exist.

It would be best to start thinking about choosing the "one" option, not "the" option.

Let’s say that we are facing the following problem: visualise two variables with a scatterplot.

```python
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
df.head()
```

![df.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1632906084410/PJZSKfhoph.png)

In Python, you’ve got 3 libraries that can make a `scatterplot`:

1. `matplotlib`
2. `seaborn`
3. `plotly`

Let’s observe the differences:

## Matplotlib

```python
import matplotlib.pyplot as plt

plt.scatter(x='total_bill', y='tip', data=df)
```

![plot_matplotlib.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1632906114826/j47rl0aQ_.jpeg)

## Seaborn

```python
import seaborn as sns

sns.scatterplot(x='total_bill', y='tip', data=df)
```

![plot_seaborn.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1632906130642/-W0_EwwJL.jpeg)
## Plotly

```python
import plotly.express as px

px.scatter(data_frame=df, x='total_bill', y='tip')
```

<iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~sotastica/1.embed"></iframe>

 
# Takeaways

1. `matplotlib` allows you to create custom plots, but you need to write more code.
2. `seaborn` automates the plot so that you don’t need to write more lines. For example, `seaborn` added the x & y axis labels by default. `matplotlib` didn’t.
3. `plotly` allows you to interact with the plot. Give it a try and hover the mouse over the points.

If you are to make a plot for an online post, you may like to use `plotly` due to its interactivity. Nevertheless, you wouldn’t use it if you were writing a paper article.

> I teach Python, R, Statistics & Data Science. I like to produce content that helps people to understand these topics better.
>
> Feel free and welcomed to give me feedback as I would like to make my tutorials clearer and generate content that interests you 🤗
>
> You can see my Tutor Profile [here](https://www.superprof.co.uk/online-rstudio-python-anaconda-jupyter-big-data-artificial-intelligence-sport-analytics-business-analytics.html) if you need Private Tutoring lessons.
