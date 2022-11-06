#!/usr/bin/env python
# coding: utf-8

# # Group 8 term project

# This project uses a support vector machine to predict whether the patient has type 1 or type 2 diabetes.

# ## Import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# ## Import Dataset

# In[2]:


diabetes_data = pd.read_csv('C:/Users/Pranav/Documents/MS Program/MSDS 530/Term_Project/diabetes.csv')


# ## Feature Split
#  Splitting features to train machine learning model that calculates the likeliness of patient having Type 1 diabetes

# In[3]:


X_T1 = diabetes_data.drop(columns = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Outcome'],axis=1)
Y_T1 = diabetes_data['Outcome']


# ## Data Split
#  Splitting data for training and testing in 80- 20 split

# In[4]:


xTrain_T1, xTest_T1, yTrain_T1, yTest_T1 = train_test_split(X_T1,Y_T1, test_size=0.2, random_state=2)


# ## Machine learning model for Type 1 diabetes
# SVM model with Polynimial kernel type has given accuraccy of 77.273% for given dataset.

# In[5]:


#kernel_fn = input("Enter the kernal function SVM algorithm. :")
s_v_m_T1 = SVC(kernel = 'poly')
s_v_m_T1.fit(xTrain_T1, yTrain_T1)
yPredict_T1 = s_v_m_T1.predict(xTest_T1)
acc_T1 = accuracy_score(yTest_T1, yPredict_T1)
accuracy_T1 = acc_T1 * 100
save_svm_model_T1 = pickle.dumps(s_v_m_T1)
print(f'Accuracy for training data is: {accuracy_T1} \n')


# ## Feature Split
#  Splitting features to train machine learning model that calculates the likeliness of patient having Type 2 diabetes

# In[6]:


X_T2 = diabetes_data.drop(columns = ['Outcome'],axis=1)
Y_T2 = diabetes_data['Outcome']


# ## Data Split
#  Splitting data for training and testing in 80- 20 split

# In[7]:


xTrain_T2, xTest_T2, yTrain_T2, yTest_T2 = train_test_split(X_T2,Y_T2, test_size=0.2, random_state=2)


# ## Machine learning model for Type 1 diabetes
# SVM model with Polynimial kernel type has given accuraccy of 79.221% for given dataset.

# In[8]:


#kernel_fn = input("Enter the kernal function SVM algorithm. :")
s_v_m_T2 = SVC(kernel = 'poly')
s_v_m_T2.fit(xTrain_T2, yTrain_T2)
yPredict_T2 = s_v_m_T2.predict(xTest_T2)
acc_T2 = accuracy_score(yTest_T2, yPredict_T2)
accuracy_T2 = acc_T2*100
save_svm_model_T2 = pickle.dumps(s_v_m_T2)
print(f'Accuracy for training data is: {accuracy_T2} \n')


# ## Testing
# Here, based on the given data we test whether patient has type 1 or type 2 diabetes

# In[9]:


input_test = (1,93,70,31,0,30.4,0.315,23)
input_arr_T1 = (input_test[1], input_test[6], input_test[7])
input_np_arr_T1 = np.asarray(input_arr_T1)
input_data_T1 = input_np_arr_T1.reshape(1,-1)

input_np_arr_T2 = np.asarray(input_test)
input_data_T2 = input_np_arr_T2.reshape(1,-1)

predict_T1 = s_v_m_T1.predict(input_data_T1)
predict_T2 = s_v_m_T2.predict(input_data_T2)

if (predict_T1 == 1 and predict_T2 == 1) or (predict_T1 == 0 and predict_T2 == 1):
    print('Patient may have Type 2 diabetes')
elif predict_T1 == 1 and predict_T2 == 0:
    print('Patient may have Type 1 diabetes')
elif predict_T1 == 0 and predict_T2 == 0:
    print('Patient does not have diabetes')

