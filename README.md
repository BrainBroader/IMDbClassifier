# IMDb Classifier
Classifing IMDb reviews as positive or negative, using multinomial Naive Bayes and Random Forest with ID3 algorithm.

**University:** Athens University of Economics and Business  
**Department:** Informatics  
**Subject:** Artificial Intelligence

**Team:** Lampros Lountzis (@lamproslntz), Andreas Gouletas (@BrainBroader)

## Table of Contents
* [Problem Description](#problem-description)
* [Dataset](#dataset)
* [Technologies](#technologies)
* [Prerequisites](#prerequisites)
* [Execution Instructions](#execution-instructions)

### Problem Description
Implement two of the following Machine Learning algorithms, so that they can be used to classify texts into two (disjoined) categories (eg. positive/negative opinion):
* **Naive Bayes** (multinomial Naive Bayes or Bernoulli Naive Bayes)
* **ID3** (optionally with pruning or premature termination of tree extension)
* **Random Forest** (optionally with pruning or premature termination of tree extension, although it's not common)
* **AdaBoost** (with decision trees of depth equal to one as the base classifier)
* **Logistic Regression** (with SGD and by adding by adding a normalization term to the objective function)

Each text should be represented by an attribute vector with values of 0 or 1, which will indicate which words of the vocabulary the text contains. 
The vocabulary should include the m most common words in the training data (or the entire data set), possibly omitting the n most common words first, 
where m and n will be hyper-parameters. Optionally, you can also add a selection of properties by calculating information gain (or otherwise) to the 
Naive Bayes classifier and Logistic Regression. The rest of the algorithms already incorporate property selection methods.

Demonstrate the learning capabilities of your algorithms. You should include in your report the results of the experiments you will perform, including (at least):
* **learning curves** showing the percentage of **accuracy** in the **training data** and **test data** depending on the number of training examples used in each repetition
of the experiment,
* curves with results of **precision**, **recall** and **F1 score** depending on the number of training examples.
You should also mention in your report the values of the hyperparameters you used (eg value of the normalization term in the Logistic Regression algorithm, number 
of trees in the Random Forest) and how you selected them (eg by testing in separate development data).

### Dataset 
To train and test our Machine Learning models, we use the collection "Large Movie Review Dataset" which is also known as "IMDb Dataset". You can download it from
the link below:

https://ai.stanford.edu/~amaas/data/sentiment/

### Technologies
The technologies used that are worth mentioning, are:
* Python,
* Numpy,
* NLTK,
* Matplotlib

### Prerequisites
Before you execute the given program, you need to:
1. download and unzip the "IMDb Dataset" (see Section "Dataset"), 
2. check if you have installed the libraries mention in Section "Technologies". 

If you haven't previously installed the libraries mentioned above, you can use the provided requirements.txt file, by running the following command:
```
cd path-to-project
pip install -r requirements.txt
```

### Execution Instructions
To execute the program the following command is used:
```
python main_xxx.py arg1
```
where 
* main_xxx is one of the two main running scripts (main_naive_bayes.py, main_random_forest.py),
* arg2 is the path to the "IMDb Dataset".


Running Naive Bayes, 
```
python main_naive_bayes.py C:\Users\lampr\Downloads\aclImdb
```

Running Random Forest, 
```
python main_random_forest.py C:\Users\lampr\Downloads\aclImdb
```
