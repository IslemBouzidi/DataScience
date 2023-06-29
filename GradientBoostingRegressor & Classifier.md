# Gradient Boosting Regressor & Classifier:

`Gradient boosting` is a popular machine learning technique for tabular data that can capture complex patterns in your data, even if they are nonlinear. 
It can handle missing values, outliers, and categorical features with ease, without requiring special treatment. 
However, it requires careful `tuning`(the process of adjusting the hyperparameters of a model to optimize its performance.) and preprocessing to achieve the best results.

### Gradient Boosting Regressor:

#### In Depth:

https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502

#### Python Code:
```python
from sklearn.ensemble import GradientBoostingRegressor

# Create the regressor object
regressor = GradientBoostingRegressor()

# Fit the regressor to your training data
regressor.fit(X_train, y_train)

# Make predictions on new data
predictions = regressor.predict(X_test)
```
### Gradient Boosting Classifier:

#### In Depth:

[https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502](https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-2-classification-d3ed8f56541e)

#### Python Code:
```python
from sklearn.ensemble import GradientBoostingClassifier

# Create the classifier object
classifier = GradientBoostingClassifier()

# Fit the classifier to your training data
classifier.fit(X_train, y_train)

# Make predictions on new data
predictions = classifier.predict(X_test)

```


RESOURCES:
100 pages of ML

ML cheatchat from stanford 

https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502

https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-2-classification-d3ed8f56541e
