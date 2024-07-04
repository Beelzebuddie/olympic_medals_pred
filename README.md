 READ ME

 Overview

This script analyzes a dataset of Olympic teams and their performance over the years, predicting the number of medals they might win based on historical data. It includes data cleaning, visualization, and a simple linear regression model to make predictions.

 Requirements

- Python 3.x
- pandas
- seaborn
- matplotlib
- scikit-learn
- numpy

 Usage

 Data Loading and Cleaning

1. Load Data:
   ```python
   import pandas as pd
   teams = pd.read_csv('teams.csv')
   ```

2. Display Data:
   ```python
   teams
   ```

3. Select Relevant Columns:
   ```python
   teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
   ```

4. Identify Missing Values:
   ```python
   teams[teams.isnull().any(axis=1)]
   ```

5. Drop Missing Values:
   ```python
   teams = teams.dropna()
   ```

 Data Visualization

1. Regression Plot - Athletes vs. Medals:
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)
   ```

2. Regression Plot - Age vs. Medals:
   ```python
   sns.lmplot(x="age", y="medals", data=teams, fit_reg=True, ci=None)
   ```

3. Histogram of Medals:
   ```python
   teams.plot.hist(y="medals")
   ```

 Model Training and Prediction

1. Split Data into Training and Testing Sets:
   ```python
   train = teams[teams["year"] < 2012].copy()
   test = teams[teams["year"] >= 2012].copy()
   ```

2. Train Linear Regression Model:
   ```python
   from sklearn.linear_model import LinearRegression
   reg = LinearRegression()
   predictors = ["athletes", "prev_medals"]
   reg.fit(train[predictors], train["medals"])
   ```

3. Make Predictions:
   ```python
   predictions = reg.predict(test[predictors])
   ```

4. Adjust Predictions:
   ```python
   test["predictions"] = predictions
   test.loc[test["predictions"] < 0, "predictions"] = 0
   test["predictions"] = test["predictions"].round()
   ```

 Error Analysis

1. Calculate Mean Absolute Error:
   ```python
   from sklearn.metrics import mean_absolute_error
   error = mean_absolute_error(test["medals"], test["predictions"])
   ```

2. Error Analysis by Team:
   ```python
   errors = (test["medals"] - test["predictions"]).abs()
   error_by_team = errors.groupby(test["team"]).mean()
   medals_by_team = test["medals"].groupby(test["team"]).mean()
   error_ratio = error_by_team / medals_by_team
   error_ratio = error_ratio[np.isfinite(error_ratio)]
   ```

3. Visualize Error Ratio:
   ```python
   error_ratio.plot.hist()
   ```

4. Sort Error Ratio:
   ```python
   error_ratio.sort_values()
   ```

 Next Steps

- Add more predictor variables to improve the model.
- Experiment with different types of models.
- Use more robust error metrics.
- Consider training separate models for different countries.
