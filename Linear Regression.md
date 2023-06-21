# Linear Regression :
Linear regression is a statistical technique used to model the relationship between a dependent variable and one or more independent variables.
It assumes a linear relationship between the variables, meaning that the dependent variable can be expressed as a linear combination of the independent variables.

  - ### Estimation : 

$$Y = β₀ + β₁X₁$$
 It looks something like this.
 
 We try to draw that line using the data we have in ordre to predict our target. 
  ![image](https://github.com/IslemBouzidi/DataScience/assets/87117961/75e563a6-91f5-46bd-a9da-2926809e8207)

  - ### Calculating the Error : 
Mean sequared Error : 
$$MSE = 1/n \sum_{k=1}^n ((Y_k- (wX_k +β))^2$$

While the mean sequared error helps Evaluate the model's performance, its derivatives `(the Jacobian matrix)` provide crucial information about the direction in which we should adjust these coefficients to minimize the error.
The derivatives indicate the slope or rate of change of the error with respect to each coefficient.

![linear Reg](https://github.com/IslemBouzidi/DataScience/assets/87117961/d4f89590-3d94-4122-ba39-18f525eec6e6)


Such that: 

$$w= w - αdw$$

$$β= β - αdβ$$

`α` is the learning rate which we can adjust according to the data that we're dealing with.
![image](https://github.com/IslemBouzidi/DataScience/assets/87117961/6344a01d-4e58-425f-9af1-7fb52bac8bc2)

  - ### The code: 

``` python 
import numpy as np

class LinearRegression:

    def __init__(self, lr=0.01, n_iters=1000):
        # Initialize the linear regression model with learning rate and number of iterations
        self.learning_rate = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Fit the linear regression model to the training data
        n_samples, n_features = X.shape
        
        # Initialize weights and bias to 0
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            # Predict the target variable
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute the gradients or derivatives of the cost function:
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # Update the weights and bias using the gradients and learning rate
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        # Predict the target variable for new input data
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
   ```
   
  - ### Hands on real data: (sooooon)


