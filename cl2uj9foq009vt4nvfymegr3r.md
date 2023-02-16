# #02 | The Decision Tree Classifier & Supervised Classification Models

**© Jesús López 2022**

Don't miss out on his posts on [**LinkedIn**](https://linkedin.com/in/jsulopzs) to become a more efficient Python developer.

## Introduction to Supervised Classification Models

Machine Learning is a field that focuses on [**getting a mathematical equation**](https://twitter.com/sotastica/status/1449735653328031745) to make predictions. Although not all Machine Learning models work the same way.

Which types of Machine Learning models can we distinguish so far?

* **Classifiers** to predict **Categorical Variables**
    
* **Regressors** to predict **Numerical Variables**
    

The previous chapter covered the explanation of a Regressor model: Linear Regression.

This chapter covers the explanation of a Classification model: the Decision Tree.

Why do they belong to Machine Learning?

* The Machine wants to get the best numbers of a mathematical equation such that **the difference between reality and predictions is minimum**:
    
    * **Classifier** evaluates the model based on **prediction success rate** y=?y^
        
    * **Regressor** evaluates the model based on the **distance between real data and predictions** (residuals) y−y^
        

There are many Machine Learning Models of each type.

You don't need to know the process behind each model because they all work the same way (see article). In the end, you will choose the one that makes better predictions.

This tutorial will show you how to develop a Decision Tree to calculate the probability of a person surviving the Titanic and the different evaluation metrics we can calculate on Classification Models.

**Table of Important Content**

1. 🛀 [How to preprocess/clean the data to fit a Machine Learning model?](#heading-data-preprocessing)
    
    * Dummy Variables
        
    * Missing Data
        
2. 🤩 [How to **visualize** a Decision Tree model in Python step by step?](#heading-model-visualization)
    
3. 🤔 [How to **interpret** the nodes and leaf's values of a Decision Tree plot?](#heading-model-interpretation)
    
4. ⚠️ How to **evaluate** Classification models?
    
    * [Accuracy](#heading-models-score)
        
    * [Confussion Matrix](#heading-the-confusion-matrix-to-compute-other-classification-metrics)
        
        * Sensitivity
            
        * Specificity
            
        * ROC Curve
            
5. 🏁 [How to compare Classification models to choose the best one?](#heading-which-one-is-the-best-model-why?)
    

## Load the Data

* This dataset represents **people** (rows) aboard the Titanic
    
* And their **sociological characteristics** (columns)
    

```python
import seaborn as sns #!
import pandas as pd

df_titanic = sns.load_dataset(name='titanic')[['survived', 'sex', 'age', 'embarked', 'class']]
df_titanic
```

![df1.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660170797225/mnR5xocNk.jpeg align="center")

## How do we compute a Decision Tree Model in Python?

We should know from the previous chapter that we need a function accessible from a Class in the library `sklearn`.

### Import the Class

```python
from sklearn.tree import DecisionTreeClassifier
```

### Instantiante the Class

To create a copy of the original's code blueprint to not "modify" the source code.

```python
model_dt = DecisionTreeClassifier()
```

### Access the Function

The theoretical action we'd like to perform is the same as we executed in the previous chapter. Therefore, the function should be called the same way:

```python
model_dt.fit()
```

\---------------------------------------------------------------------------

TypeError Traceback (most recent call last)

/var/folders/24/tg28vxls25l9mjvqrnh0plc80000gn/T/ipykernel\_3553/3699705032.py in ----&gt; 1 model\_dt.fit()

TypeError: fit() missing 2 required positional arguments: 'X' and 'y'

Why is it asking for two parameters: `y` and `X`?

* `y`: target ~ independent ~ label ~ class variable
    
* `X`: explanatory ~ dependent ~ feature variables
    

### Separate the Variables

```python
target = df_titanic['survived']
explanatory = df_titanic.drop(columns='survived')
```

### Fit the Model

```python
model_dt.fit(X=explanatory, y=target)
```

\---------------------------------------------------------------------------

ValueError: could not convert string to float: 'male'

Most of the time, the data isn't prepared to fit the model. So let's dig into why we got the previous error in the following sections.

## Data Preprocessing

The error says:

```python
ValueError: could not convert string to float: 'male'
```

From which we can interpret that the function `.fit()` does **not accept values of** `string` type like the ones in `sex` column:

```python
df_titanic
```

![df2.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660170816894/CamtYcRxD.jpeg align="center")

### Dummy Variables

Therefore, we need to convert the categorical columns to **dummies** (0s & 1s):

```python
pd.get_dummies(df_titanic, drop_first=True)
```

![df3.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660170823884/p5Y1U9DUN.jpeg align="center")

```python
df_titanic = pd.get_dummies(df_titanic, drop_first=True)
```

We separate the variables again to take into account the latest modification:

```python
explanatory = df_titanic.drop(columns='survived')
target = df_titanic[['survived']]
```

### Fit the Model Again

Now we should be able to fit the model:

```python
model_dt.fit(X=explanatory, y=target)
```

\---------------------------------------------------------------------------

ValueError: Input contains NaN, infinity or a value too large for dtype('float32').

### Missing Data

The data passed to the function contains **missing data** (`NaN`). Precisely 177 people from which we don't have the age:

```python
df_titanic.isna()
```

![df4.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660170833201/OKv7hlWrp.jpeg align="center")

```python
df_titanic.isna().sum()
```

survived 0 age 177 sex\_male 0 embarked\_Q 0 embarked\_S 0 class\_Second 0 class\_Third 0 dtype: int64

Who are the people who lack the information?

```python
mask_na = df_titanic.isna().sum(axis=1) > 0
```

```python
df_titanic[mask_na]
```

![df5.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660170841198/pjKqV7g3V.jpeg align="center")

What could we do with them?

1. Drop the people (rows) who miss the age from the dataset.
    
2. Fill the age by the average age of other combinations (like males who survived)
    
3. Apply an algorithm to fill them.
    

We'll choose **option 1 to simplify the tutorial**.

Therefore, we go from 891 people:

```python
df_titanic
```

![df6.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660170847355/3owbqm7FO.jpeg align="center")

To 714 people:

```python
df_titanic.dropna()
```

![df7.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660170853597/QL3lheWC0.jpeg align="center")

```python
df_titanic = df_titanic.dropna()
```

We separate the variables again to take into account the latest modification:

```python
explanatory = df_titanic.drop(columns='survived')
target = df_titanic['survived']
```

Now we shouldn't have any more trouble with the data to fit the model.

### Fit the Model Again

We don't get any errors because we correctly preprocess the data for the model.

Once the model is fitted, we may observe that the object contains more attributes because it has calculated the best numbers for the mathematical equation.

```python
model_dt.fit(X=explanatory, y=target)
model_dt.__dict__
```

{'criterion': 'gini', 'splitter': 'best', 'max\_depth': None, 'min\_samples\_split': 2, 'min\_samples\_leaf': 1, 'min\_weight\_fraction\_leaf': 0.0, 'max\_features': None, 'max\_leaf\_nodes': None, 'random\_state': None, 'min\_impurity\_decrease': 0.0, 'class\_weight': None, 'ccp\_alpha': 0.0, 'feature\_names\_in\_': array(\['age', 'sex\_male', 'embarked\_Q', 'embarked\_S', 'class\_Second', 'class\_Third'\], dtype=object), 'n\_features\_in\_': 6, 'n\_outputs\_': 1, 'classes\_': array(\[0, 1\]), 'n\_classes\_': 2, 'max\_features\_': 6, 'tree\_': &lt;sklearn.tree.\_tree.Tree at 0x16612cce0&gt;}

### Predictions

#### Calculate Predictions

We have a fitted `DecisionTreeClassifier`. Therefore, we should be able to apply the mathematical equation to the original data to get the predictions:

```python
model_dt.predict_proba(X=explanatory)[:5]
```

array(\[\[0.82051282, 0.17948718\], \[0.05660377, 0.94339623\], \[0.53921569, 0.46078431\], \[0.05660377, 0.94339623\], \[0.82051282, 0.17948718\]\])

#### Add a New Column with the Predictions

Let's create a new `DataFrame` to keep the information of the target and predictions to understand the topic better:

```python
df_pred = df_titanic[['survived']].copy()
```

And add the predictions:

```python
df_pred['pred_proba_dt'] = model_dt.predict_proba(X=explanatory)[:,1]
df_pred
```

![df_pred1.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660171919701/LhaXXpgQg.jpg align="center")

How have we calculated those predictions?

### Model Visualization

The **Decision Tree** model doesn't specifically have a mathematical equation. But instead, a set of conditions is represented in a tree:

```python
from sklearn.tree import plot_tree

plot_tree(decision_tree=model_dt);
```

![plot1.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660170890682/MaHbkMPfG.jpeg align="center")

There are many conditions; let's recreate a shorter tree to explain the Mathematical Equation of the Decision Tree:

```python
model_dt = DecisionTreeClassifier(max_depth=2)
model_dt.fit(X=explanatory, y=target)

plot_tree(decision_tree=model_dt);
```

![small_tree.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660172038089/E6lvEdoHn.jpg align="center")

Let's make the image bigger:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plot_tree(decision_tree=model_dt);
```

![plot3.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660170915015/s32Oo3CmT.jpeg align="center")

The conditions are `X[2]<=0.5`. The `X[2]` means the 3rd variable (Python starts counting at 0) of the explanatory ones. If we'd like to see the names of the columns, we need to add the `feature_names` parameter:

```python
explanatory.columns
```

Index(\['age', 'sex\_male', 'embarked\_Q', 'embarked\_S', 'class\_Second', 'class\_Third'\], dtype='object')

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plot_tree(decision_tree=model_dt, feature_names=explanatory.columns);
```

![plot4.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660170922327/Cbnqnp2Dh.jpeg align="center")

Let's add some colours to see how the predictions will go based on the fulfilled conditions:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plot_tree(decision_tree=model_dt, feature_names=explanatory.columns, filled=True);
```

![plot5.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660170928175/UV_CWp7b_.jpeg align="center")

### How does the Decision Tree Algorithm computes the Mathematical Equation?

The Decision Tree and the Linear Regression algorithms look for the best numbers in a mathematical equation. The following video explains how the Decision Tree configures the equation:

%[https://www.youtube.com/watch?v=_L39rN6gz7Y] 

### Model Interpretation

Let's take a person from the data to explain how the model makes a prediction. For storytelling, let's say the person's name is John.

John is a 22-year-old man who took the titanic on 3rd class but didn't survive:

```python
df_titanic[:1]
```

![df10.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660171151064/leJQxgNgG.jpeg align="center")

To calculate the chances of survival in a person like John, we pass the explanatory variables of John:

```python
explanatory[:1]
```

![df11.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660171173555/1lWpXXrSA.jpeg align="center")

To the function `.predict_proba()` and get a probability of 17.94%:

```python
model_dt.predict_proba(X=explanatory[:1])
```

array(\[\[0.82051282, 0.17948718\]\])

But wait, how did we get to the probability of survival of 17.94%?

Let's explain it step-by-step with the Decision Tree visualization:

```python
plt.figure(figsize=(10,6))
plot_tree(decision_tree=model_dt, feature_names=explanatory.columns, filled=True);
```

![plot6.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660171180961/45A72rDZH.jpeg align="center")

Based on the tree, the conditions are:

#### 1st condition

* sex\_male (John=1) &lt;= 0.5 ~ False
    

John doesn't fulfil the condition; we move to the right side of the tree.

#### 2nd condition

* age (John=22.0) &lt;= 6.5 ~ False
    

John doesn't fulfil the condition; we move to the right side of the tree.

#### Leaf

The ultimate node, the leaf, tells us that the training dataset contained 429 males older than 6.5 years old.

Out of the 429, 77 survived, but 352 didn't make it.

Therefore, the chances of John surviving according to our model are 77 divided by 429:

```python
77/429
```

0.1794871794871795

We get the same probability; John had a 17.94% chance of surviving the Titanic accident.

### Model's Score

#### Calculate the Score

As always, we should have a function to calculate the goodness of the model:

```python
model_dt.score(X=explanatory, y=target)
```

0.8025210084033614

The model can correctly predict 80.25% of the people in the dataset.

What's the reasoning behind the model's evaluation?

#### The Score Step-by-step

As we saw [earlier](#Leaf), the classification model calculates the probability for an event to occur. The function `.predict_proba()` gives us two probabilities in the columns: people who didn't survive (0) and people who survived (1).

```python
model_dt.predict_proba(X=explanatory)[:5]
```

array(\[\[0.82051282, 0.17948718\], \[0.05660377, 0.94339623\], \[0.53921569, 0.46078431\], \[0.05660377, 0.94339623\], \[0.82051282, 0.17948718\]\])

We take the positive probabilities in the second column:

```python
df_pred['pred_proba_dt'] = model_dt.predict_proba(X=explanatory)[:, 1]
```

At the time to compare reality (0s and 1s) with the predictions (probabilities), we need to turn probabilities higher than 0.5 into 1, and 0 otherwise.

```python
import numpy as np

df_pred['pred_dt'] = np.where(df_pred.pred_proba_dt > 0.5, 1, 0)
df_pred
```

![df_pred2.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660171966417/mhrVTnMsQ.jpg align="center")

The simple idea of the accuracy is to get the success rate on the classification: how many people do we get right?

We compare if the reality is equal to the prediction:

```python
comp = df_pred.survived == df_pred.pred_dt
comp
```

0 True 1 True ...  
889 False 890 True Length: 714, dtype: bool

If we sum the boolean Series, Python will take True as 1 and 0 as False to compute the number of correct classifications:

```python
comp.sum()
```

573

We get the score by dividing the successes by all possibilities (the total number of people):

```python
comp.sum()/len(comp)
```

0.8025210084033614

It is also correct to do the mean on the comparisons because it's the sum divided by the total. Observe how you get the same number:

```python
comp.mean()
```

0.8025210084033614

But it's more efficient to calculate this metric with the function `.score()`:

```python
model_dt.score(X=explanatory, y=target)
```

0.8025210084033614

### The Confusion Matrix to Compute Other Classification Metrics

Can we think that our model is 80.25% of good and be happy with it?

* We should not because we might be interested in the accuracy of each class (survived or not) separately. But first, we need to compute the confusion matrix:
    

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(
    y_true=df_pred.survived,
    y_pred=df_pred.pred_dt
)

CM = ConfusionMatrixDisplay(cm)
CM.plot();
```

![plot7.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660171221714/MwYHvaBVg.jpeg align="center")

1. Looking at the first number of the confusion matrix, we have 407 people who didn't survive the Titanic in reality and the predictions.
    
2. It is not the case with the number 17. Our model classified 17 people as survivors when they didn't.
    
3. The success rate of the negative class, people who didn't survive, is called the **specificity**: $407/(407+17)$.
    
4. Whereas the success rate of the positive class, people who did survive, is called the **sensitivity**: $166/(166+124)$.
    

#### Specificity (Recall=0)

```python
cm[0,0]
```

407

```python
cm[0,:]
```

array(\[407, 17\])

```python
cm[0,0]/cm[0,:].sum()
```

0.9599056603773585

```python
sensitivity = cm[0,0]/cm[0,:].sum()
```

#### Sensitivity (Recall=1)

```python
cm[1,1]
```

166

```python
cm[1,:]
```

array(\[124, 166\])

```python
cm[1,1]/cm[1,:].sum()
```

0.5724137931034483

```python
sensitivity = cm[1,1]/cm[1,:].sum()
```

#### Classification Report

We could have gotten the same metrics using the function `classification_report()`. Look a the recall (column) of rows 0 and 1, specificity and sensitivity, respectively:

```python
from sklearn.metrics import classification_report

report = classification_report(
    y_true=df_pred.survived,
    y_pred=df_pred.pred_dt
)

print(report)
```

precision recall f1-score support

0 0.77 0.96 0.85 424 1 0.91 0.57 0.70 290

accuracy 0.80 714 macro avg 0.84 0.77 0.78 714 weighted avg 0.82 0.80 0.79 714

We can also create a nice `DataFrame` to later use the data for simulations:

```python
report = classification_report(
    y_true=df_pred.survived,
    y_pred=df_pred.pred_dt,
    output_dict=True
)

pd.DataFrame(report)
```

![df13.jpeg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660171236074/O6grRSYws.jpeg align="center")

Our model is not as good as we thought if we predict the people who survived; we get 57.24% of survivors right.

How can we then assess a reasonable rate for our model?

#### ROC Curve

Watch the following video to understand how the Area Under the Curve (AUC) is a good metric because it sort of combines accuracy, specificity and sensitivity:

%[https://www.youtube.com/watch?v=4jRBRDbJemM] 

We compute this metric in Python as follows:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

y = df_pred.survived
pred = model_dt.predict_proba(X=explanatory)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y, pred)
roc_auc = metrics.auc(fpr, tpr)

display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='example estimator')
display.plot()
plt.show()
```

![roc_curve.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660172000155/2v_ElfbBE.jpg align="center")

```python
roc_auc
```

0.8205066688353937

## Other Classification Models

Let's build other classification models by applying the same functions. In the end, computing [Machine Learning models is the same thing all the time](https://blog.resolvingpython.com/why-all-machine-learning-models-are-the-same).

### `RandomForestClassifier()` in Python

#### Fit the Model

```python
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X=explanatory, y=target)
```

RandomForestClassifier()

#### Calculate Predictions

```python
df_pred['pred_rf'] = model_rf.predict(X=explanatory)
df_pred
```

![df_pred3.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660172073022/n0FOnicgV.jpg align="center")

#### Model's Score

```python
model_rf.score(X=explanatory, y=target)
```

0.9117647058823529

### `SVC()` in Python

#### Fit the Model

```python
from sklearn.svm import SVC

model_sv = SVC()
model_sv.fit(X=explanatory, y=target)
```

SVC()

#### Calculate Predictions

```python
df_pred['pred_sv'] = model_sv.predict(X=explanatory)
df_pred
```

![df_pred4.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660172081260/Ox186zPuL.jpg align="center")

#### Model's Score

```python
model_sv.score(X=explanatory, y=target)
```

0.6190476190476191

## Which One Is the Best Model? Why?

To simplify the explanation, we use accuracy as the metric to compare the models. We have the Random Forest as the best model with an accuracy of 91.17%.

```python
model_dt.score(X=explanatory, y=target)
```

0.8025210084033614

```python
model_rf.score(X=explanatory, y=target)
```

0.9117647058823529

```python
model_sv.score(X=explanatory, y=target)
```

0.6190476190476191

```python
df_pred.head(10)
```

![df_pred5.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1660172092852/sYw1QCnTf.jpg align="center")

[![Creative Commons License](https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png align="center")](http://creativecommons.org/licenses/by-nc-nd/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](http://creativecommons.org/licenses/by-nc-nd/4.0/).