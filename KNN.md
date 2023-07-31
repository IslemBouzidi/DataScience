# KNN

One popular method used for prediction tasks is the K-Nearest Neighbors (KNN) algorithm. KNN is a non-parametric algorithm that can be used for both classification and regression tasks. It operates on the principle that similar data points tend to have similar outcomes.

<p align="center">
  <img width="460" height="300" src="https://media.springernature.com/lw685/springer-static/image/chp%3A10.1007%2F978-981-15-9938-5_35/MediaObjects/488434_1_En_35_Fig1_HTML.png">
</p>

From: https://link.springer.com/chapter/10.1007/978-981-15-9938-5_35
### 1.1 KNN from scretch simple implementation :

In this implementation, we will set ***K=3*** ,and for the distance we will use ***the euclidean formula*** thought in school. 
$$d(\mathbf {p,q})= \sqrt{\sum \limits_{i=1}^n (q_i-p_i)^2}$$


``` python

import numpy as np
import matplotlib.pyplot as plt

# Define the dataset
X = np.array([[1, 2], [1.5, 1], [1.25, 3],[1.75, 2], [2, 4], [2, 2.5], [3, 2], [4, 2], [3, 3],[2.75, 2], [2.5, 4], [4, 2.5]])
y = np.array([2, 2, 2,2,2,2, 1, 1, 1,1,1,1]) # categories

# Define the Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Define the predict function for KNN classification
def predict(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])  # Sort distances in ascending order
    neighbors = distances[:k]  # Get the K nearest neighbors

    class_labels = [neighbor[1] for neighbor in neighbors]  # Extract the class labels

    # Count the occurrences of each class label
    counts = np.bincount(class_labels)

    # Return the class label with the highest count
    return np.argmax(counts)

# Scatter plot of the dataset
plt.scatter(X[:, 0], X[:, 1], c=y,cmap='jet')
plt.xlabel('X1')
plt.ylabel('X2')

new_point = np.array([2.5, 2.5])
plt.scatter(new_point[0], new_point[1], color='green', marker='x', label='Test Point')

# Plot lines connecting the test point to its nearest neighbors
k = 3
distances = []
for i in range(len(X)):
    dist = euclidean_distance(X[i], new_point)
    distances.append((dist, X[i], y[i]))

distances.sort(key=lambda x: x[0])  # Sort distances in ascending order
neighbors = distances[:k]  # Get the K nearest neighbors

for neighbor in neighbors:
    neighbor_point = neighbor[1]
    plt.plot([new_point[0], neighbor_point[0]], [new_point[1], neighbor_point[1]], 'k--', linewidth=0.5)

    
plt.legend()
plt.show()
# Predict the label for a new data point
new_point = np.array([2.5, 2.5])
predicted_label = predict(X, y, new_point, k=3)
print("Predicted label: blue" if predicted_label == 1 else "Predicted label: red")

```
![image](https://github.com/IslemBouzidi/DataScience/assets/87117961/46370214-8cd7-48b7-8542-975b3c601538)

``` python
# if we set newpoint to:
new_point = np.array([2, 2])
```
![image](https://github.com/IslemBouzidi/DataScience/assets/87117961/fbfb0c75-3dad-40ef-a56c-95d9303b95ad)

### Let's put it in a class :

``` python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):

        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]# euclidean_distance is already defined in the code above
    

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]


        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
```

### Sklearn:
``` python
import numpy as np
from sklearn.neighbors import KNeighborsRegressor #KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42
)

# Initializing the kNN regression model
k = 3  # Set the value of k (number of neighbors)
knn_regression = KNeighborsRegressor(n_neighbors=k)# knn_classification = KNeighborsClassifier(n_neighbors=k)

# Fitting the model to the training data
knn_regression.fit(X_train_regression, y_train_regression)

# Making predictions on the test data
y_pred_regression = knn_regression.predict(X_test_regression)

# Calculate the mean squared error to evaluate the model
mse_regression = mean_squared_error(y_test_regression, y_pred_regression)
print("Mean Squared Error:", mse_regression)

```
### 1.2 Hands on a Real Life Project: 
1.2.1 Small Dataset : ***Classification of Iris Species***

The dataset was clean and small.

I would like to just showcase the K-Nearest Neighbors (KNN) code for your reference.

[The NoteBook :SeabornVisualisationProject & KNN Classification](https://github.com/IslemBouzidi/DataScience/blob/main/Seaborn%20Visualization%20%26%20KNN%20classification.ipynb)

1.2.2 BIG Data: ***Classification of (GALAXY, STAR QSO)***

A more complicated problem with Knn classification implementation.

[Seaborn Visualisation & knn Classification (Star Data)](https://github.com/IslemBouzidi/DataScience/blob/main/Seaborn%20Visualization%20%26%20knn%20Classification%20(Star%20Data).ipynb)

The notebook contains some french waiting to be translated, plus working on visualization more (Soon)


