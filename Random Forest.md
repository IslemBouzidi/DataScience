# Random Forest :


### - The idea:
Random Forest is a popular machine learning algorithm used for both classification and regression tasks. 
It is an ensemble method that combines multiple decision trees to make predictions.
So instead of creating one tree, we create multiples ones in a random way. like the shown pic.

![image](https://github.com/IslemBouzidi/DataScience/assets/87117961/9355299e-5e76-4b8b-9995-6b37243551f8)


### - The code :
``` python
from DecisionTree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions 

```

### - Python code:
```python
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(
    n_estimators=300,   # Number of decision trees in the forest
    max_features='log2',   # Maximum number of features to consider at each split, log2 == sqrt 2
    criterion='entropy',   # Splitting criterion based on entropy
    class_weight='balanced',   # Class weights for addressing class imbalance
    min_samples_split=2,   # Minimum number of samples required to split an internal node
    min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
    max_depth=100,   # Maximum depth of the decision trees
    random_state=2   # Random seed for reproducibility
)

# Fit the model to the training data
rf_classifier.fit(X_train, y_train)

# Predict using the trained model
y_pred = rf_classifier.predict(X_test)
```

`n_estimators`: It determines the number of decision trees in the Random Forest. In this case, we have set it to 300, meaning the Random Forest will consist of 300 decision trees. You can adjust this value based on the trade-off between model performance and computational complexity.

`max_features`: This parameter determines the maximum number of features to consider when looking for the best split at each node of a decision tree. In this example, we set it to 'log2', which means the square root of the total number of features will be considered. You can also use other values like 'sqrt' or an integer value to specify a fixed number of features.

`criterion`: It defines the criterion used to measure the quality of a split in the decision trees. Here, we set it to 'entropy', which uses information gain based on the entropy of the target variable. Alternatively, you can use 'gini' to use the Gini impurity as the splitting criterion.

`class_weight`: This parameter addresses class imbalance by assigning weights to the classes. In this example, we set it to 'balanced', which automatically adjusts the class weights inversely proportional to their frequencies. This helps give more importance to minority classes. You can also provide a custom dictionary of class weights if needed.

`min_samples_split`: It specifies the minimum number of samples required to split an internal node. If the number of samples at a node is less than this value, the node will not be split further, and it will become a leaf node. The default value is 2, meaning the split will stop if there is only one sample at a node.

`min_samples_leaf`: It determines the minimum number of samples required to be at a leaf node. If the number of samples at a leaf node is less than this value, the splitting process will not continue, and the node will become a leaf. The default value is 1, meaning even individual samples can become leaf nodes.

`max_depth`: This parameter sets the maximum depth of the decision trees in the Random Forest. If None (default), the trees are grown until all leaves are pure or until all leaves contain less than min_samples_split samples.

`random_state`: It is used to set a random seed for reproducibility. Providing a specific value ensures that the Random Forest algorithm produces the same results each time it is run. You can change this value to get different random splits of data during the training process.

For refrence: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
