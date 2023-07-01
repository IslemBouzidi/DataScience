# 1- Decision Tree

### 1.1 The idea: 
![image](https://github.com/IslemBouzidi/DataScience/assets/87117961/5a563384-2e15-4986-bf38-52b72bca37ca)

### 1.3 The math:

  - Information gain $`IG`$:

$$IG = Ent(parent) - [Weighted Average].Ent(children)$$

  - Entropy $`Ent`$:
  
 $$Ent = -\sum_{k=1}^n p(X_K)log_2(p(X_K))$$
 
   - Such that: 
   
     $`p(X)`$= _the total number the class $`X`$ has occured in this node / total nodes_

   - Stopping Criteria: _max depth, min number of samples, min impurity decrease_
### 1.3 The code:

``` python 
###########                                    Decision Tree Implementation in python                                    ###########

import numpy as np
from collections import Counter 

class Node :
    def __init__(self, feature=None, threshold= None, left=None, right= None,*,value=None):
        ''' 
        feature: the feature index that the node uses for splitting 
        threshold: the threshold value used for the split (it's like the condition that we're splitting the data according to)
        left: the left child node
        right: the righ child node 
        value: the predicted value at the node this is only set if the node is a leaf node
        '''
        self.feature= feature
        self.threshold= threshold
        self.right= right
        self.left= left 
        self.value= value
        
    def is_leaf_node(self):
        # function to be called to check if the node a a leaf node or not 
        # a leaf node is when we stop, just as the leaves of a given tree
        return self.value is not None 
    
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth= 100, n_features= None):
        '''
        min_samples_split: this argument represents the minimum number of samples required to split an internal node.By default, it is set to 2
        because it's a common sense that we can split two or more things 

        max_depth= how far the tree can go ( the levels )

        n_features=  This argument determines the number of features to consider when looking for the best split at each node. 
        By default, it is set to None, which means that all features will be considered.

        it's important though to know that if f a specific value is provided,
        the decision tree will randomly select n_features features at each node to find the best split,
        which can help introduce randomness and reduce overfitting.
         
        '''
        self.min_samples_split= min_samples_split
        self.max_depth= max_depth
        self.n_features= n_features
        self_root= None


    def fit(self, X,y):
        # we're not gonna use all the features at once, so we should check if the columns of x equals nmbr of feaatures
        self.n_features =  X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X,y)
        

    def _grow_tree(self, X,y, depth= 0):
        n_samples, n_feats = X.shape
        n_labels= len(np.unique(y))


        # checking for the stopping criteria 
        if(depth >= self.max_depth or n_labels==1 or n_samples< self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node( value=leaf_value )
        
        
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    ```
