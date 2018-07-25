
## HOMEWORK 2: Decision Trees

### Programming

The goal of this assignment is to implement a binary classifier from scratch - a Decision Tree Learner.

### Datasets

The datasets are included in the handout directory.

1. politician: train data and test data to carry out the task to predict whether a US politician is a member of the Democrat or Republican party

2. education: train data and test data to carry out the task to predic t he final grade (A, not A) for high school students

3. small: small dataset for initial stage checking convenience

### Program#1: Inspecting the Data

Write a program inspect.{py|java|cpp|m} to calculate the label entropy at the root (i.e. the entropy of the labels before any splits) and the error rate (the percent of incorrectly classified instances) of classifying using a majority vote (picking the label with the most examples). You do not need to look at the values of any of the attributes to do these calculations, knowing the labels of each example is sufficient. Entropy should be calculated in bits using log base 2.

### Program#2: Decision Tree Learner

In decisionTree.{py | java | cpp | m}, implement a Decision Tree learner. This file should learn a decision tree with a specified maximum depth, print the decision tree in a specified format, predict the labels of the training and testing examples, and calculate training and testing errors.




