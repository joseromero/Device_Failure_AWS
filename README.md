# Device Failure

## Introduction

The following project illustrates one approach into finding a model that predicts if a device will fail depending on daily aggregate metrics. This type of problem is quite common in the industry since it can guide non reactive mainteinance and help companies save considerable costs associated to failing devices.

For more detail on the problem definition, see the document in **/data/Device Failure.pdf** folder. For data exploration, data is available in **data/device_failure.csv**

## Challenges

As you will see along the process, this is not a trivial problem. The main challenge emerges from the nature of the problem itself. As you can imagine, the data you have available is extremelly unbalanced. This means that you are going to have few cases failing devices compared with non failing cases. Is literally like finding a needle in a haystack. More on this later.

## Process

As you can see in this repository, the process followed to build the model is divided in six mayor stages:

- **Target and Missing Values Analysis:** We will make a deeper understanding of how unbalanced the data really is. Additionally we will assess the effect of missing values in overall data set.
- **Exploratory Analysis:** We will explore each feature on original data. For categorical features (device and date), we will explore possible derived features. For numrical features (metrics) we will explore their respective distribution. Each analysis will separate target values (failing and not failing) to understand how they differ (if they do) in original data as is.
- **Derived Features:** We will define derived features based on original numeric data (metrics). Such new features will consider time changes and metrics rank (per metric and accumulated along metrics).
- **Data Preparation:** We will experiment on different transformations on all numeric features. This transformations aims to reduce information redundancy (multicolinearity) and to standardize features to meet requirements of some model candidates. For categorical data One Hot Encoding will be reviewed as well.
- **Model Design:** We will ensemble six different model candidates. This candidates will evaluate undersampling strategies (more of this on target / missing values analysis), classification algorithms and ensemble structures. Each model will be built without any hyperparamether tuning, and as output, two candidates will be selcted for further tuning.
- **Model Tunning:** For the two selected candidates. Hyperparametter tunning will be performed with Random Search technique. Best candidate based on scoring (more on this on target / missing values analysis) with best configuration will be selected for final Train / Test.
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

- Class `Toolkit` includes a set of function that support preparation and analysis requirements that emerged while performing the process described above. This class was built focused on general purpose usage.

- Class `PF` which stand for **Problem Functions** is the accumulation of all problem specific functions that where designed along the process described above. The idea for accumulating such funtions is for easier usage on further steps of the process.

All functions and elements in **toolkit.py** are fully documented for easy understanding.

##  Concepts

The following project aims to experiment different concepts in the persuit of an effective model. This concept will be explained along as they are introduced in the process described above.

## Executive Summary

In order to follow the process, you can either follow each file for each step, or follow the **device_failure.ipynb** file. I highly encourage you to check all files for a deeper understanding of the entire process.

## Output Files

In the **/output** folder, you are going to find the results for a Random Search made for hyperparamether tunning of two models and the results for a Feature Importance analysis performed throughout the process. 

The purpose of this files is to avoy the execution of both analysis since they are resource intensive and time consuming. In each step where theis results are needed, you will find a function that easilly loads the resukt data.

## Final Comment

### Hope you like the work done. 

### **Please provide any feedback that can help in my learning process.**


