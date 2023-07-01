# Logistic Linear Regression
 Logistic regression is a statistical technique used to model the relationship between a binary dependent variable and one or more independent variables. 
 Unlike linear regression, which assumes a linear relationship between variables, logistic regression is specifically designed for binary classification problems.
## The math:

For this case, we're gonna use the logistic function $`S(X)`$ & Cross-entropy cost function $`Cost`$ 
$$S(x)= L/(1+exp(-β(x-x_0))$$

_$`L`$	=	the curve's maximum value_ 

_$`k`$	=	logistic growth rate or steepness of the curve_

_$`x_0`$	=	the x value of the sigmoid midpoint_

_$`x`$	=	real number_ 

![image](https://github.com/IslemBouzidi/DataScience/assets/87117961/fea42303-665c-4482-880a-50e471d32857)

$$Cost =  1/n \sum_{k=1}^n [Y_klog(h_λ(X_k)) +(1-Y_k)log(1-h_λ(X_k))]$$
_$`h_λ(X_k)= β₀ + β₁X₁`$ used in linear regression_

![image](https://github.com/IslemBouzidi/DataScience/assets/87117961/b4385285-406d-45f7-897f-639e21ca6f08)

## The code: 
``` python 

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
        
 ```
 
 ## Hands on Real Data:( sooon)
