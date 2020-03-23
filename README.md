# Device Failure

## Introduction

The following project illustrates one approach into finding a model that predicts if a device will fail depending on daily aggregate metrics. This type of problem is quite common in the industry since it can guide non-reactive maintenance and help companies save considerable costs associated to failing devices.

For more detail on the problem definition, see the document in **/data/Device Failure.pdf**. For data exploration, data is available in **data/device_failure.csv**

All content and detailed code for this project can be found in: 
- **[https://github.com/joseromero/Device_Failure_AWS](https://github.com/joseromero/Device_Failure_AWS)**

I strongly invite you to visit it.

## Challenges

As you will see along the process, this is not a trivial problem. The main challenge emerges from the nature of the problem itself. As you can imagine, the data you have available is extremely unbalanced. This means that you are going to have few cases of failing devices compared with non-failing cases. Is literally like finding a needle in a haystack. More on this later.

## Process

As you can see in this repository, the process followed to build the model is divided in six mayor stages:

- **Target and Missing Values Analysis:** We will make a deeper understanding of how unbalanced the data really is. Additionally, we will assess the effect of missing values in overall data set.
- **Exploratory Analysis:** We will explore each feature on original data. For categorical features (device and date), we will explore possible derived features. For numerical features (metrics) we will explore their respective distribution. Each analysis will separate target values (failing and not failing) to understand how they differ (if they do) in original data as is.
- **Derived Features:** We will define derived features based on original numeric data (metrics). Such new features will consider time changes and metrics rank (per metric and accumulated along metrics).
- **Data Preparation:** We will experiment on different transformations on all numeric features. These transformations aim to reduce information redundancy (multicollinearity) and to standardize features to meet requirements of some model candidates. For categorical data One Hot Encoding will be reviewed as well.
- **Model Design:** We will ensemble six different model candidates. These candidates will evaluate under sampling strategies (more of this on target / missing values analysis), classification algorithms and ensemble structures. Each model will be built without any hyperparameter tuning, and as output, two candidates will be selected for further tuning.
- **Model Tuning:** For the two selected candidates. Hyperparameter tuning will be performed with Random Search technique. Best candidate based on scoring (more on this on target / missing values analysis) with best configuration will be selected for final Train / Test.
- **Final Results:** We will perform the best model on train / test data and analyze results. Final comments and further analysis will be commented.

## Toolkit

As you may notice the project development is supported by **toolkit.py** which consist of three main classes:

```python
class Viz():
    ...

class Toolkit():
    ...

class PF():
    ...
```

- Class `Viz` includes a set of functions that support visualization requirements that emerged while performing the process described above. This class was built focused on general purpose usage.

- Class `Toolkit` includes a set of functions that support preparation and analysis requirements that emerged while performing the process described above. This class was built focused on general purpose usage.

- Class `PF` which stands for **Problem Functions** is the accumulation of all problem specific functions that where designed along the process described above. The idea for accumulating such functions is for easier usage on further steps of the process.

All functions and elements in **toolkit.py** are fully documented for easy understanding.

##  Concepts

The following project aims to experiment different concepts in the pursuit of an effective model. This concepts will be explained along as they are introduced in the process described above.

## Output Files

In the **/output** folder, you are going to find the results for a Random Search made for hyperparameter tuning of two models and the results for a Feature Importance analysis performed throughout the process.

The purpose of this files is to avoid the execution of both analysis since they are resource intensive and time consuming. In each step where these results are needed, you will find a function that easily loads them for use.

## Some Resources

Many of the ideas implemented in this project are sourced from the following post.

 - **[Silicon Data Science Post](https://www.svds.com/learning-imbalanced-classes/)**

## Final Comment

### Hope you like the work done. 

### **Please provide any feedback that can help in my learning process.**