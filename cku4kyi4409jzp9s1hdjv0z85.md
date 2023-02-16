# Your First Lines of Code in Python

Programming is hard, especially at the beginning.

Don't make it yourself any harder!

Start with Data Visualization.

It's easier to understand programming with visual changes than abstract coding ("make a program that prints even numbers").

Get on Jupyter, a code editor. Here is the link to [download the program](https://www.anaconda.com/products/individual.

Your first lines of code should be as follows:

```python
import seaborn as sns

df = sns.load_dataset('tips')
sns.scatterplot(x='total_bill', y='tip', data=df)
```

You would get a plot that should look like this one ↓

![plot.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1632861779687/t23TrD07X.jpeg)

To configure the behaviour of the function, you should configure the code as follows:

```python
sns.scatterplot(x='total_bill', y='tip', data=df, color='red')
```

![plot.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1632862257093/_Su9Xi9Wi.jpeg)

This simple change helps you to understand a couple of core concepts in programming:

1. __Functions__ `sns.scatteplot()` are used to make things in programming (a plot in this case).

2. You use __parameters__ `color='red'` to configure the function's behaviour.

Feel free and welcome to ask me anything in the comments below, it will be my pleasure to help you out 🤗