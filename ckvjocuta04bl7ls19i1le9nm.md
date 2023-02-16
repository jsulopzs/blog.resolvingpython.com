# Tutorial | Machine Learning Model Deployment

We already know that a Machine Learning Model is a **mathematical formula** to calculate something ↓

%[https://twitter.com/sotastica/status/1449735653328031745]

Machine Learning Models are deployed to, for example:

- Predict objects within an image (**Tesla**) so that the car can take actions
- **Spotify** recommends songs to a user so that you'd fall in love with the service
- Most likely to interact posts in **Facebook or Twitter** so that you will spend more time on the app

If you just care about getting the code to make this happen, you can forget the storytelling and get right into those lines in [GitHub ↗︎](https://github.com/py-thocrates/tutorial-machine-learning-deployment)

If you want to follow the tutorial and understand the topic in depth, let's get started ↓

Let's say that **we are a car sales company** and we want to make things easier for clients when they decide which car to buy.

They usually don't want to have a car that **consumes lots of fuel** `mpg`.

Nevertheless, *they won't know this until they use the car*.

Is there a way to **`predict` the consumption** based on other characteristics of the car?

- Yes, with a mathematical formula, for example:

```
consumption = 2 + 3 * acceleration * 2.1  horsepower
```

We have **historical data** from all cars models we have sold over the past few years.

We could use this **data to calculate the BEST mathematical formula**.

And `deploy it to a website` with a form to solve the consumption question by themselves.

To make this happen, we will follow the structure:

1. Create ML Model Object in Python
2. Create an HTML Form
3. Create Flask App
4. Deploy to Heroku
5. Visit Website and Make a Prediction

## Create ML Model Object in Python

### Import Data to Python

- This dataset contains information about **car models** (rows)
- For which we have some **characteristics** (columns)


```python
import seaborn as sns

df = sns.load_dataset(name='mpg', index_col='name')[['acceleration', 'weight', 'mpg']]
df.sample(5)
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
      <th>acceleration</th>
      <th>weight</th>
      <th>mpg</th>
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
      <th>subaru</th>
      <td>17.8</td>
      <td>2065</td>
      <td>32.3</td>
    </tr>
    <tr>
      <th>bmw 2002</th>
      <td>12.5</td>
      <td>2234</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>audi 5000</th>
      <td>15.9</td>
      <td>2830</td>
      <td>20.3</td>
    </tr>
    <tr>
      <th>toyota corolla 1200</th>
      <td>21.0</td>
      <td>1836</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>ford gran torino (sw)</th>
      <td>16.0</td>
      <td>4638</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>



### Linear Regression Model from Historical Data


```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X=df[['acceleration', 'weight']], y=df['mpg'])
model.__dict__
```




    {'fit_intercept': True,
     'normalize': False,
     'copy_X': True,
     'n_jobs': None,
     'positive': False,
     'n_features_in_': 2,
     'coef_': array([ 0.25081589, -0.00733564]),
     '_residues': 7317.984100916719,
     'rank_': 2,
     'singular_': array([16873.21840634,    49.92970477]),
     'intercept_': 41.39982830200016}



And the BEST mathematical formula is:
    
```
consumption = 41.39 + 0.25 * acceleration - 0.0073 * weight
```

### Save `LinearRegression()` into a File

- The object `LinearRegression()` contains the Mathematical Formula
- That we will use in the website to make the `prediction`


```python
import pickle

with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

Now a file called `linear_regression_model.pkl` should appear in the **same folder that your script**.

## Create HTML form

All websites that you see online are displayed through an HTML file.

Therefore, we need to create an HTML file that contains a `form` for the user to **input the data**.

And **calculate the `prediction` for the fuel consumption**.

Website example [here ↗︎](https://ml-model-deployment-car-mpg.herokuapp.com/)

- Let's head over a Code Editor (VSCode in my case) and create a new file called `index.html`

> You may download Visual Studio Code (VSCode) [here ↗︎](https://code.visualstudio.com/download)

That should contain the following lines:

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Document</title>
  </head>
  <body>
    <form>
      <label for="acceleration">Acceleration (m/s^2):</label><br />
      <input
        type="number"
        id="acceleration"
        name="acceleration"
        value="34"
      /><br />
      <label for="weight">Weight (kg):</label><br />
      <input type="number" id="weight" name="weight" value="12" /><br /><br />
      <input type="submit" value="Submit" />
    </form>
  </body>
</html>
```

1. If you open the file `index.html` in a browser, you will see the form.

2. And the `submit` button that is supposed to calculate the prediction.

3. Nevertheless, if you click, nothing will happen.

- Because we need to develop the `Flask` application **to send the user input to a mathematical formula to calculate the prediction** and return that into the website.

## Create Flask App

As we are going to develop a whole application to a web server (Heroku), we need to create a **dedicated environment** with just the necessary packages.

- Let's head over the terminal and type the following commands:

```zsh
python -m venv car_consumption_prediction
source car_consumption_prediction/bin/activate
```

- Now let's install the required packages:

```zsh
pip install flask
pip install scikit-learn
```

- Now you should open the folder `car_consumption_prediction` in a Code Editor

- And create a new folder `app` with two other folders inside:

```
- app
    - model
    - templates
```

- Then move the files we created before to its corresponding folders:

```
- app
    - model
        - linear_regression_model.pkl
    - templates
        - index.html
```

Now that we have the project structure, let's continue with the core functionality

We will build a **Python script that handles the user input** and make the prediction for fuel consumption

- So, create a new file within `app` folder called `app.py`

> **PS:** This is the most important file in a `Flask` app because it **manages everything**.

```
- app
    - model
        - linear_regression_model.pkl
    - templates
        - index.html
    - app.py
```

- And add the following lines of code:

```python
import flask
import pickle

with open(f'model/linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    elif flask.request.method == 'POST':
        acceleration = flask.request.form['acceleration']
        weight = flask.request.form['weight']

        input_variables = [[acceleration, weight]]

        prediction = model.predict(input_variables)[0]

        return flask.render_template('index.html',
                                     original_input={'Acceleration': acceleration,
                                                     'Weight': weight},
                                     result=prediction,
                                     )


if __name__ == '__main__':
    app.run()
```

1. We need to pay attention to what's going on in the last `return ...`:

2. The function `render_template()` is passing the objects from parameters `original_input` and `result` to `index.html`

3. Then, how can we use this variables in the file `index.html`?

- Copy-paste the following lines of code into `index.html`:

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Document</title>
  </head>
  <body>
    <form action="{{ url_for('main') }}" method="POST">
      <label for="acceleration">Acceleration (m/s^2):</label><br />
      <input type="number" id="acceleration" name="acceleration" required /><br />
      <label for="weight">Weight (kg):</label><br />
      <input type="number" id="weight" name="weight" required /><br /><br />
      <input type="submit" value="Submit" />
    </form>
    <br />
    {% if result %}
    <p>
      The calculated fuel consumption is
      <span style="color: orange">{{result}}</span>
    </p>
    {% endif %}
  </body>
</html>
```

We made two changes to the file:
    
1. Specify the action to take when `form` is submitted:

    ```html
    <form action="{{ url_for('main') }}" method="POST">
    ```

2. Show the prediction below the form

    ```html
    {% if result %}
    <p>
      The calculated fuel consumption is
      <span style="color: orange">{{result}}</span>
    </p>
    {% endif %}
    ```

In this case, we had to use the conditional `if` to display `result` if existed, as `result` won't exist until the form is submitted and the `server` computes the prediction in `app.py`.

### Add the Procfile

I did some research about an error in which Heroku wasn't working the way I expected

And found that I needed to add a `Procfile` ↓

1. Create a file in the folder `app` called `procfile`

2. Write the following line and save the file:

    ```procfile
    web: gunicorn app:app
    ```
    
    The folder structure will now be:

    ```
    - app
        - model
            - linear_regression_model.pkl
        - templates
            - index.html
        - app.py
        - procfile
    ```
    
3. Install the `gunicorn` package in the virtual environment. In terminal:

    ```zsh
    pip install gunicorn
    ```

### Deploy to Heroku

Now it's the time to upload the application to Heroku so that anyone can get its prediction on fuel comsumption given a car's `acceleration` and `weight`.

1. Create an Account in [Heroku](https://signup.heroku.com).
2. Download [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli#download-and-install)
3. Create the Heroku App within the Terminal:

```zsh
heroku create ml-model-deployment-car-mpg
```

This will be traduced into a website called https://ml-model-deployment-car-mpg.herokuapp.com/

> - **PS:** You should use a different name instead of `ml-model-deployment-car-mpg` heroku will turn your repository into an `url`.

**Commit the `app` files to your heroku hosting**.

1. `git init` within `car_consumption_prediction` folder
2. Create a `requirements.txt` file with the instruction for required packages. You could automatically create this by:

    ```zsh
    pip freeze > requirements.txt
    ```

    The folder structure will now be:

    ```
    - app
        - model
            - linear_regression_model.pkl
        - templates
            - index.html
        - app.py
        - procfile
        - requirements.txt
    ```

3. Add the files for commit.

    ```zsh
    git add .
    ```

4. Commit the files to the remote

    ```zsh
    git commit -m 'some random message'
    git push heroku master
    ```

That's all the technical aspect.

Now if some user would like to use the app...

## Visit Website and Make a Prediction

1. Visit https://ml-model-deployment-car-mpg.herokuapp.com/
2. Introduce some numbers in the form
3. Submit and watch the prediction

## References

- https://blog.cambridgespark.com/deploying-a-machine-learning-model-to-the-web-725688b851c7