# Lasso & Ridge Regression

### 1. Why Lasso or Ridge Regression:

By now, many of you are likely familiar with the term "overfitting," which occurs when a model performs exceptionally well on the training data but fails to generalize to new, unseen test data. In other words, the model becomes too closely tuned to the specific characteristics of the training dataset, compromising its ability to handle unfamiliar data.

![image](https://github.com/IslemBouzidi/DataScience/assets/87117961/0064dc84-0aa7-4006-8050-04b5c750d450)

To address this issue, we have two powerful techniques: Lasso and Ridge Regression. Lasso, named after the L1 norm it employs, and Ridge Regression, named after the L2 norm it utilizes. These regularization methods help mitigate overfitting by imposing constraints on the model's coefficients, preventing them from becoming overly sensitive to the training data's idiosyncrasies.

In Lasso Regression, the L1 norm penalty is applied, which can be represented as:
$$||w||^1 = |w_₁| + |w_₂| + ... + |w_n|$$

In Ridge Regression, the L2 norm penalty is used, which can be represented as:
$$||w||^2 = √(w_₁² + w_₂² + ... + w_n²)$$

These regularization terms encourage the model to find a balance between fitting the training data and maintaining simplicity, thus helping to prevent overfitting.

In Lasso Regression, the L1 norm penalty is applied to the cost function, which can be represented as:

```scss
Cost(Lasso) = MSE + λ * ||w||₁
```
where MSE (Mean Squared Error) measures the average squared difference between the predicted and actual values, and λ (lambda) is the regularization parameter that controls the strength of the regularization effect. The L1 norm penalty term, ||w||₁, encourages sparsity in the coefficient values. It shrinks some coefficients to zero, effectively selecting a subset of the most relevant features and eliminating less important ones.



In Ridge Regression, the L2 norm penalty is applied to the cost function, which can be represented as:

```scss
Cost(Ridge) = MSE + λ * ||w||₂²
```
Where ||w||₂² represents the square of the L2 norm of the coefficient vector w. Similar to Lasso Regression, λ is the regularization parameter controlling the strength of regularization. The L2 norm penalty term, ||w||₂², encourages small and balanced coefficient values without forcing them to zero. It helps prevent extreme coefficient values and reduces the impact of less influential features.

To obtain the best coefficients in Lasso or Ridge Regression, we aim to minimize the cost function by adjusting the coefficient values. This is typically done using optimization algorithms such as `gradient descent` (for Ridge, lasso usese  L1 which is not continuous, and hence not differentiable) or `closed-form solutions`(for ridge)

![image](https://github.com/IslemBouzidi/DataScience/assets/87117961/0ca59c78-f1c5-4808-bf4a-8410ddc001bb)


credits: `Sreenath S`

### 2. Code:
``` python
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
```
#### Adjusting Parameters:

For Ridge Regression:
```python
ridge = Ridge(alpha=0.5)  # Adjust the alpha parameter for regularization strength
```
By modifying the value of `alpha`, you can control the strength of regularization in Ridge Regression. Higher values of `alpha` will increase the amount of regularization, which may help prevent overfitting. Experiment with different values of `alpha` to find the optimal balance between bias and variance in your model.

For Lasso Regression:
```python
lasso = Lasso(alpha=0.1)  # Adjust the alpha parameter for regularization strength
```
Similar to Ridge Regression, you can adjust the value of `alpha` in Lasso Regression to control the strength of regularization. Higher values of `alpha` will increase the level of sparsity in the model by shrinking more coefficients towards zero. Test different values of `alpha` to find the optimal level of regularization for your dataset.
