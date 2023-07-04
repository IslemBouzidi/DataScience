# Introduction to Machine Learning:
1. **What is machine learning?**
   Machine learning is the field of study that enables computers to learn from data without being explicitly programmed.

2. **Four types of problems where ML shines:**
   - Problems with hand-tuning or long rule lists (e.g., spam emails).
   - Complex problems with no traditional solutions (e.g., speech recognition).
   - Adapting to fluctuating environments.
   - Gaining insights from complex data (e.g., big data).

3. **Two most common supervised tasks:**
   Classification and regression.

4. **Four common unsupervised tasks:**
   - Clustering: Grouping data into clusters of similar items (e.g., website visitors with similar behavior).
   - Anomaly detection: Identifying unusual patterns or outliers (e.g., detecting fraudulent credit card transactions).
   - Visualization and dimensionality reduction: Representing high-dimensional data in lower dimensions for analysis and visualization.
   - Association rule learning: Discovering interesting relationships between attributes in large datasets.

5. **Machine learning for a robot to walk in unknown terrains:**
   Reinforcement learning.

6. **Labeled data:**
   Data with known and assigned classes or labels.

7. **ML algorithm to segment customers into multiple groups:**
   Clustering.

8. **Spam detection:**
   Supervised ML problem (classification).

9. **Online learning:**
   Updating the model continuously as new data becomes available, allowing the model to adapt to changing patterns.

10. **Out-of-core learning:**
    Training the model in small manageable chunks when the dataset is too large to fit into memory.

11. **Learning algorithm relying on similarity measures:**
    Instance-based learning.

12. **Difference between model parameters and learning algorithm hyperparameters:**
    Model parameters are learned from the data during training (e.g., $`θ`$ values in linear regression), while hyperparameters are set by the user before training (e.g., $`learning rate`$).

13. **Model-based ML searches for:**
    Patterns in the data and captures relationships.

14. **Strategy used by model-based ML:**
    Searching for the best parameters based on performance measures.

15. **How does the model make predictions?**
    Using the best parameters found during training, the model takes input values (X) and predicts the corresponding output (Y).

16. **Main challenges of ML:**
    - Insufficient quantity of training data.
    - The unreasonable effectiveness of some algorithms, leading to overfitting.
    - Non-representative training data.
    - Poor data quality.

17. **Overfitting:**
    When the model performs well on the training data but generalizes poorly to new, unseen instances.

18. **Possible solutions for overfitting:**
    - Simplify the model by selecting one with fewer parameters.
    - Gather more training data.
    - Reduce noise in the training data or remove outliers.

19. **Test data:**
    Data used to evaluate the performance of the model on unseen instances.

20. **Purpose of a validation set:**
    Selecting the best model or hyperparameters based on their performance during training.

21. **Test set vs validation set:**
    The test set provides an unbiased evaluation of the final model's performance on unseen data, while the validation set is used for model selection and hyperparameter tuning during development.

22. **Issues with tuning hyperparameters using the test set:**
    Tuning hyperparameters using the test set can lead to overfitting, data leakage, and unreliable model evaluation. It can result in difficulty in selecting the best model or hyperparameters.

23. **Solution to the problem of tuning hyperparameters using the test set:**
    To avoid these issues, it is recommended to use a separate validation set or adopt techniques like cross-validation. The validation set should be used for hyperparameter tuning and model selection, while the test set should be kept strictly for final evaluation after the hyperparameter tuning process. This approach helps ensure better generalization of the model and provides a more accurate assessment of its performance on unseen data.

24. **Repeated cross-validation and single validation set:**
    Repeated cross-validation involves performing cross-validation multiple times, using different random partitions of the data each time. This approach provides a more robust evaluation of the model's performance. However, it increases the training time as the model is trained multiple times. In contrast, using a single validation set reduces training time but may result in a slightly less precise estimation of model performance.

     
