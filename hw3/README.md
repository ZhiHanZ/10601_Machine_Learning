## HOMEWORK 4: Logistic Regression

### Programming

The goal of this assignment is to implement a working Natural Language Processing (NLP) system, i.e., a mini Siri, using multinomial logistic regression. You will then use your algorithm to extract flight informa- tion from natural text. You will do some very basic feature engineering, though which you will be able to im- prove the learner’s performance on this task.implement a binary classifier from scratch - a Decision Tree Learner.

### Datasets

The handout contains data from the Airline Travel Information System (ATIS) data set (for more details, see <http://deeplearning.net/tutorial/rnnslu.html>). Each data set consists of attributes (words) and labels (airline flight information tags). The attributes and tags are separated into sequences (i.e., phrases) with a blank line between each sequence.

The tags are in Begin-Inside-Outside (BIO) format. Tags starting with B indicating the beginning of a piece of information, tags beginning with I indicating a continuation of a previous type of information, and O tags indicating words outside of any information chunk. For example, in the sentence below, Newark is the departure city (fromloc.city_name), Los Angeles is the destination (toloc.city_name), and the user is requesting flights for Wednesday (depart_date.day_name). In this homework, you will treat the BIO tags as arbitrary labels for each word.

```
what    O
flights O
from    O
newark  B-fromloc.city_name
to      O
los     B-toloc.city_name
angeles I-toloc.city_name
on      O
wed     B-depart_date.day_name
```

### Program: Tagger

Write a program, tagger.{py|java|cpp|m}, that implements a text analyzer using multinomial logis- tic regression. The file should learn the parameters of a multinomial logistic regression model that predicts a tag (i.e. label) for each word and its corresponding feature vector. The program should output the labels of the training and test examples and calculate training and test error (percentage of incorrectly labeled words).

