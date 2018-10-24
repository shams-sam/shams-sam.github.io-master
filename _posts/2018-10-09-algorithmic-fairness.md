---
layout: post
title: "Social Bias in Machine Learning"
categories: []
tags: [machine-learning, algorithmic-fairness]
description: Algorithmic bias is machines making unfair decisions that have been observed in the history and recorded in to form of data that mirror the prevailing social, ethnic or gender inequalities of the time.
cover: "/assets/images/bias.jpeg"
cover_source: "https://cdn-images-1.medium.com/max/1280/1*l--yko0wJLMxahiXeoZ3kg.jpeg"
comments: true
mathjax: true
---

### Introduction

Discrimination, injustice, oppression are some of the dark words that have been an integral part of human history. While there is a active effort to make world a fair place in every sphere of life, it is almost impossible to make the data that has been recorded over the years fair to all the caste, creeds, races and religions because the history is written in ink. Since the world was far more biased as we age backwards in time, it would not be incorrect to say that a historical record of data would often reflect these biases in terms of minority and majority classes. These are the very same data that is continuously used in training most of our machine learning models without actually giving a conscious thought to the fairness of the algorithm i.e. whether or not the algorithm reflects the biases that prevailed back then. Recently machine learning has seen its utilitization in a lot of important decision making pipelines such as predicting time of recidivism, college acceptance, loan approvals etc., and hence it becomes increasingly important to question the machine learning models being developed in terms of implicit bias that they might be inheriting from the data that they train on. In order to do away with such biases in a machine learning algorithm one needs to understand how exactly does bias creep in, what are the various metrics through which it can be measured and what are the methods through which one can remove such unfairness. This post is an attempt to summarize such issues and possible remedies.

### Background

Since machine learning is now being used to make a lot policy decisions that affect the life of people on an everyday basis, it should be made sure that unfairness is not a part of such decision making. It is found that training machine learning algorithms with the standard utility maximization and loss minimization objectives sometimes result in algorithms that behave in a way that a fair human observer would deem biased. A very recent example of such a case was [cited](https://www.ml.cmu.edu/news/news-archive/2018/october/amazon-scraps-secret-artificial-intelligence-recruiting-engine-that-showed-biases-against-women.html){:target="\_blank"} by Amazon which notices a gender bias in its recruiting engine algorithms.

### It's all in the Data

One of the potential reasons for such biases in these algorithms can be attributed to the training data itself. Since the algorithms are big numerical puzzles that are trained to recognize and mimic the statistical patterns over the history, it is only natural for such a trained system to display biased characteristics. Even some of the state of the art solutions in the field of NLP and Machine Learning are not free from biases and unfairness. For example, it has been shown that word2vec embeddings learnt from huge corpuses of text often show gender bias as the euclidean distance between words that signifies correlation between words, suggests strong correlation between words like homemaker, nanny with she and maestro, boss with he. Any system built on top of such a word embedding is very likely to propagate this bias on a daily basis at some level.

One of the contested ways of dealing with this issue is to retrain the models continuously with new data, which relies on the assumption that historical bias is on a process of correcting itself.

Another major question that continuously arise is based on the fact the these machine learning algorithms work well when the amount of data they train on is huge. While this is true in an overall sense, if we break down the number of data points one has for minority class it becomes more apparent that the algorithms does not have enough supporting instances to learn as good a representation about minority classes as it would about the majority and hence could lead in unfair judgements because of lack of data.

> There is general tendency for automated decisions to favor those who belong to statistically dominant groups.

Statistical patterns that apply to majority population might be invalid for the the minority group. It can also happen that a variable that is positively correlated with target in general population maybe negatively correlated with target in the minority group. For example, a real name might be a short common name in one culture and a long unique name in another. Hence same rules for detecting fake names would not work across such groups.

![Fig-1: Survival Distribution](/assets/2018-10-09-algorithmic-fairness/fig-2-survival-distribution.png?raw=true)

Consider a very simple dataset from Kaggle called [titanic](https://www.kaggle.com/c/titanic){:target="\_blank"}. This is a basic dataset where based on a bunch of features given one has to **predict the survival probability of an individual who was on titanic**. The survival distribution on the training data shows that in past **during the titanic incident a female candidate had much higher chances of surviving than a male candidate**. It would be rather obvious **for an algorithm trained on this data that being female is a strong indicator of survival**. If the same algorithm was used to predict survival on an impending sinking incident where candidates who have higher survival probability would be boarded on rescue boats first, it is bound to make biased decisions.

Also it can be seen that being male is negatively correlated to surviving while being a female is positively correlated, because graph 2 in fig-1 shows that more males died than survived and by contrast,  more females survived than died. So **if the algorithm was to learn only from majority of the data belonging to males, it would predict badly for the female population**.


### Undeniable Complexities

One way to counter the sample size disparity might be to learn different classifiers for different sub-groups. But it is not as simple as it sounds because of the reason that learning and testing for individual sub-group might require acting on the protected attributes which might in itself be objectionable. Also the definition of minority is fuzzy as there could be many different overlapping minorities and no straightforward way of determining group membership.

### Noise vs Modeling Error

Say a classifier achieves 95 percent accuracy. In the real world scenario this 5 percent error rate would point to a really well trained classifier. But what is often overlooked is that there might be two different kinds of underlying reasons behind the error rate. One could be the general case of noise that the classifier was not able to model and hence was not able to predict and account for. Other possible reason could be that while the model is 100 percent accurate on majority class, it is only 50 percent accurate on minority class. This systematic error in the minority class would be a clear case of algorithmic unfairness.

The bigger issue of the matter here is that there is no principled or book methodology for distinguishing noise from the modeling errors. Such questions can only be answered by great deal of domain knowledge and experience.

### Edge Cases always exist

It is also true to assume that in a very unexpected way it is possible for bias to creep into the algorithms even if the training data is labelled correctly and is free of any issues that could be pointed out as unbiased. A recent [example](https://www.theverge.com/2015/7/1/8880363/google-apologizes-photos-app-tags-two-black-people-gorillas){:target="\_blank"} of this is when google photos by mistake labeled two black people as gorillas. Obviously, the machine was never trained with any training data that should lead to such inferences, but because the number of trained parameters are so high, it often becomes intractable and unimaginably hard to understand why a system behaves haphazardly in certain conditions. This uncertainty of outcomes can also be a cause of bias in situations that could not be predicted in advance.

### What is Fairness?

Fairness in classification involves studying algorithms not only from a perspective of accuracy, but also from a perspective of fairness. 

> The most difficult part of this is to define what is fairness.

Consideration for fairness often leads to compromise on accuracy but it's a necessary evil that is not going anywhere in the near future. What if often more surprising is that many of these metrics have a trade off among themselves.

### Fairness of Process vs Fairness of Outcome

- An **aware** algorithm is one that uses the information regarding the protected attribute (such as gender, ethnicity etc.) in the process of learning. An **unaware** algorithm will not.

- While the motivation regarding unaware algorithm is that being fair means disregarding the protected attribute, it often does not work just by removing the protected attribute. Sometimes there is a strong correlation between protected attribute and some other feature. So in order to train a truly unaware algorithm, one needs to remove the correlated feature group as well.

- This process of manually engineering a feature list that conveys no information about the protected attribute can also be automated using machine learning techniques discussed in following sections.

### Are Unaware Algorithms the Solution

- There could be inherent differences between the populations defined by these masked protected attributes, which would only render this process undesirable.

- The aware approaches use these proctected attributes and have a better chance of understanding depence of outcome on them.

- This can be seen as a distinction between **fairness of process** vs **fairness of outcomes**. The unaware algorithms ensure a fairness of process, because under such a scheme the algorithm does not use any of the protected attributes for decision making. However, such fairness in process does not guarantee a fair outcome towards the protected and un-protected sub-groups.

- The aware approaches on the contrary use these protected attributes and hence not a fair process, but it can reach an outcome that is more fair towards the minorities.

### Mathematical Fairness: Statistical Parity

A mathematical version of absolute fairness can be a statistical condition where the chances of success or failure is same for both the majority and minority classes (or more classes in case of multi-class scenarios). This can be written as,

$$Pr[h(x) = 1 \vert x \in P^C] = Pr[h(x) = 1 \vert x \in P] \tag{1} \label{1}$$

The main drawback of such models is given by the argument that is that **does one really want to equalize the outcomes across all sub-groups?**. For example, predicting the success chances of a basketball player irrespective of his height is not really a very strong model, because the discrimination in various domains do not really fall in a black or white region but may lie in the gray region somewhere in between. Another example might be predicting the chances of child birth without using the features such as gender and age would be a really poor algorithm. So, **enforcing the statistical parity is not always the solution**.

### Cross-group Calibration

- Instead of equalizing the outcomes themselves, one can look to equalize some other statistics of the algorithm's performance, for example **error rates across groups**.

> A fair algorithm would make as many mistakes on a minority group as it does on the majority group.

A useful tool for such an analysis is the confusion matrix as shown below

![Fig-2: Confusion Matrix](/assets/2018-10-09-algorithmic-fairness/fig-1-confusion-matrix.png?raw=true)

Some of the metrics based on the confusion matrix are:

* **Treatment equality** is achieved by a classifier that yields a ratio of false negatives and false positives (in table, c/b or b/c) that is same for both protected group categories.

* **Conditional procedure accuracy equality** is achieved when conditioning on the known outcome, the classifier is equally accurate across protected group categories. This is equivalent to the false negative rate and false positive rate being same for all protected categories.

Since all the columns and rows of a confusion matrix should add up to the total number of observations, many of these fainess metrics have a trade-off relationship. This basically means **zero-sum game**, one increases at the cost of the other and there is no win-win situation. Based on the use-case one has to decide which metrics should be optimized for as there is no blanket solution to the group.


### Example: Titanic

[**Kaggle Notebook**](https://www.kaggle.com/shamssam/algorithmic-fairness-in-ml){:target="\_blank"}

```python

### libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

df_train = pd.read_csv('train.csv', index_col='PassengerId')

df_train.Sex = df_train.Sex == 'female'
df_train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

X_train, X_valid = train_test_split(df_train, test_size=0.3, stratify=df_train.Survived.tolist())

# aware classification

clf = XGBClassifier()
clf.fit(X_train.drop(['Survived'], axis=1), X_train.Survived.tolist())

print("="*40)
print("OVERALL")
print("="*40)
y_valid_hat = clf.predict(X_valid.drop(['Survived'], axis=1))
print(classification_report(X_valid.Survived.tolist(), y_valid_hat))
print("Accuracy: {}".format(accuracy_score(X_valid.Survived.tolist(), y_valid_hat)))
print("="*40)

print("FEMALE")
print("="*40)
y_valid_hat_female = clf.predict(X_valid[X_valid.Sex == True].drop(['Survived'], axis=1))
print(classification_report(X_valid[X_valid.Sex == True].Survived.tolist(), y_valid_hat_female))
print("Accuracy: {}".format(accuracy_score(X_valid[X_valid.Sex == True].Survived.tolist(), y_valid_hat_female)))

print("="*40)
print("MALE")
print("="*40)
y_valid_hat_male = clf.predict(X_valid[X_valid.Sex == False].drop(['Survived'], axis=1))
print(classification_report(X_valid[X_valid.Sex == False].Survived.tolist(), y_valid_hat_male))
print("Accuracy: {}".format(accuracy_score(X_valid[X_valid.Sex == False].Survived.tolist(), y_valid_hat_male)))


# output

# ========================================
# OVERALL
# ========================================
#              precision    recall  f1-score   support

#           0       0.85      0.90      0.87       165
#           1       0.82      0.75      0.78       103

# avg / total       0.84      0.84      0.84       268

# Accuracy: 0.8395522388059702
# ========================================
# FEMALE
# ========================================
#              precision    recall  f1-score   support

#           0       0.45      0.41      0.43        22
#           1       0.83      0.86      0.84        76

# avg / total       0.75      0.76      0.75        98

# Accuracy: 0.7551020408163265
# ========================================
# MALE
# ========================================
#              precision    recall  f1-score   support

#           0       0.90      0.97      0.94       143
#           1       0.75      0.44      0.56        27

# avg / total       0.88      0.89      0.88       170

# Accuracy: 0.888235294117647


# unaware classification

clf = XGBClassifier()
clf.fit(X_train.drop(['Survived', 'Sex'], axis=1), X_train.Survived.tolist())

print("="*40)
print("OVERALL")
print("="*40)
y_valid_hat = clf.predict(X_valid.drop(['Survived', 'Sex'], axis=1))
print(classification_report(X_valid.Survived.tolist(), y_valid_hat))
print("Accuracy: {}".format(accuracy_score(X_valid.Survived.tolist(), y_valid_hat)))
print("="*40)

print("FEMALE")
print("="*40)
y_valid_hat_female = clf.predict(X_valid[X_valid.Sex == True].drop(['Survived', 'Sex'], axis=1))
print(classification_report(X_valid[X_valid.Sex == True].Survived.tolist(), y_valid_hat_female))
print("Accuracy: {}".format(accuracy_score(X_valid[X_valid.Sex == True].Survived.tolist(), y_valid_hat_female)))

print("="*40)
print("MALE")
print("="*40)
y_valid_hat_male = clf.predict(X_valid[X_valid.Sex == False].drop(['Survived', 'Sex'], axis=1))
print(classification_report(X_valid[X_valid.Sex == False].Survived.tolist(), y_valid_hat_male))
print("Accuracy: {}".format(accuracy_score(X_valid[X_valid.Sex == False].Survived.tolist(), y_valid_hat_male)))

# output

# ========================================
# OVERALL
# ========================================
#              precision    recall  f1-score   support

#           0       0.73      0.84      0.78       165
#           1       0.66      0.51      0.58       103

# avg / total       0.71      0.71      0.70       268

# Accuracy: 0.7126865671641791
# ========================================
# FEMALE
# ========================================
#              precision    recall  f1-score   support

#           0       0.32      0.82      0.46        22
#           1       0.90      0.50      0.64        76

# avg / total       0.77      0.57      0.60        98

# Accuracy: 0.5714285714285714
# ========================================
# MALE
# ========================================
#              precision    recall  f1-score   support

#           0       0.91      0.84      0.87       143
#           1       0.39      0.56      0.46        27

# avg / total       0.83      0.79      0.81       170

# Accuracy: 0.7941176470588235
```

Confusion matrices for the cases using awareness and without awareness of protected attribute (sex in this case) is shown below.

![Fig-3: Aware Confusion Matrix](/assets/2018-10-09-algorithmic-fairness/fig-3-aware-confusion-matrix.png?raw=true)

![Fig-4: Unaware Confusion Matrix](/assets/2018-10-09-algorithmic-fairness/fig-4-unaware-confusion-matrix.png?raw=true)

Note:

- Conditional accuracy in the code output shows that the system is very biased both in aware in unaware scenarios.
- treatment equality is more divergent in aware case than in unaware case.

## REFERENCES:

<small>[A Gentle Introduction to the Discussion on Algorithmic Fairness
](https://towardsdatascience.com/a-gentle-introduction-to-the-discussion-on-algorithmic-fairness-740bbb469b6){:target="_blank"}</small><br> 
<small>[How big data is unfair](https://medium.com/@mrtz/how-big-data-is-unfair-9aa544d739de){:target="\_blank"}</small><br> 
<small>[Fairness Measures](http://fairness-measures.org/){:target="\_blank"}</small><br> 
