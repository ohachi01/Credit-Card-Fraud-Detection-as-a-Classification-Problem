#!/usr/bin/env python
# coding: utf-8

# # CREDIT CARD FRAUD DETECTION AS A CLASSIFICATION PROBLEM 
# 

# By Chimbudike Ohakim

# Project Objective: The objective of my project is to develop and implement an effective online credit card fraud detection system using machine learning algorithms. Our aim was to leverage advanced analytics and predictive modeling techniques to accurately identify fraudulent transactions in real-time, thereby minimizing financial losses and enhancing security for both cardholders and financial institutions. By integrating cutting-edge technologies with comprehensive data analysis, our goal was to create a robust and scalable solution capable of adapting to evolving fraud patterns and ensuring the integrity of online transactions
# 
# Project background: This project involves building a model to detect fraudulent financial transactions, the internet also comes with pros and cons. All of us enjoy the pros as the internet has changed our lifestyle by enhancing our communication. But, at the same time, we are witnessing digital frauds, which include fraudulent transactions through stolen credit cards. 
# This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced; the positive class (frauds) account for 0.172% of all transactions. The dataset has been collected and analyzed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Universite Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on http://mlg.ulb.ac.be/BuFence and http://mlg.ulb.ac.be/ARTML.

# In[7]:


# Importing modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from zipfile import ZipFile
import urllib.request
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


# In[3]:


folder = urllib.request.urlopen('https://s3.amazonaws.com/hackerday.datascience/68/creditcard.csv.zip')
zipfile = ZipFile(BytesIO(folder.read()))
zipfile.namelist()


# In[5]:


#Loading Dataset
data = pd.read_csv(zipfile.open("creditcard.csv"))
pd.concat([data.head(),data.tail()])


# In[6]:


#Finding proportion of fraudulent vs non fraudulent credit card transactions
print("Fraudulent Cases: " + str(len(data[data["Class"] == 1])))
print("Valid Transactions: " + str(len(data[data["Class"] == 0])))
print("Proportion of Fraudulent Cases: " + str(len(data[data["Class"] == 1]) / data.shape[0]))

# To see how small are the number of Fraud transactions
data_pi = data.copy()
data_pi[" "] = np.where(data_pi["Class"] == 1, "Fraud", "Genuine")

get_ipython().run_line_magic('matplotlib', 'inline')
data_pi[" "].value_counts().plot(kind="pie")
plt.title('Distribution of Transaction Types')
plt.ylabel('')
plt.show()


# Now we look at the distribution of the two named features in the dataset. For Time, it is clear that there were certain duration in the day when most of the transactions took place.

# In[8]:


f, axes = plt.subplots(1, 2, figsize=(18,4), sharex = True)

amount_val = data['Amount'].values
time_val = data['Time'].values

sns.distplot(amount_val, hist=False, color="m", kde_kws={"shade": True}, ax=axes[0]).set_title('Distribution of Transaction Amount')
sns.distplot(time_val, hist=False, color="m", kde_kws={"shade": True}, ax=axes[1]).set_title('Distribution of Transaction Time')

plt.show()


# In[10]:


print("Average Amount in a Fraudulent Transaction: " + str(data[data["Class"] == 1]["Amount"].mean()))
print("Average Amount in a Valid Transaction: " + str(data[data["Class"] == 0]["Amount"].mean()))


# As we can clearly notice from this, the average Money transaction for the fraudulent ones are more. This makes this problem crucial to deal with. Now let us try to understand the distribution of values in each of the features. Let's start with the Amount.
# 

# In[11]:


print("Summary of the feature - Amount" + "\n-------------------------------")
print(data["Amount"].describe())


# The rest of the features don't have any physical interpretation and will be seen through histograms. Here the values are subgrouped according to Class (valid or fraud).
# 

# In[12]:


# Reorder the columns Amount, Time then the rest
data_plot = data.copy()
amount = data_plot['Amount']
data_plot.drop(labels=['Amount'], axis=1, inplace = True)
data_plot.insert(0, 'Amount', amount)

# Plot the distributions of the features
columns = data_plot.iloc[:,0:30].columns
plt.figure(figsize=(12,30*4))
grids = gridspec.GridSpec(30, 1)
for grid, index in enumerate(data_plot[columns]):
 ax = plt.subplot(grids[grid])
 sns.distplot(data_plot[index][data_plot.Class == 1], hist=False, kde_kws={"shade": True}, bins=50)
 sns.distplot(data_plot[index][data_plot.Class == 0], hist=False, kde_kws={"shade": True}, bins=50)
 ax.set_xlabel("")
 ax.set_title("Distribution of Column: "  + str(index))
plt.show()


# # Data Preparation

# The features are created using PCA and so the feature selection is not necessary as the number of features is small as well. Now we turn to the treatment of Missing values in the dataframe.

# In[13]:


data.isnull().shape[0]
print("Number of cases with non-missing values: " + str(data.isnull().shape[0]))
print("Number of cases with missing values: " + str(data.shape[0] - data.isnull().shape[0]))


# As there are no missing data, we turn to standardization. We standardize only Time and Amount using RobustScaler.

# In[14]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler().fit(data[["Time", "Amount"]])
data[["Time", "Amount"]] = scaler.transform(data[["Time", "Amount"]])

pd.concat([data.head(),data.tail()])


# # Data Modeling for Financial Transactions Data

# Now we start modelling.
# First we divide the data into response and features. And also make the train-test split of the data for further modelling and validation.

# In[16]:


# Separate response and features
y = data["Class"]
X = data.iloc[:,0:30]

# Use SKLEARN for the split
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size = 0.2, random_state = 42)


# Now we describe the flow of the modelling section first and then dive into the sea. As we identified earlier, the dataset is highly imbalanced. Fitting a model on this dataset will result in overfitting towards the majority class. For illustration let's run one model (Random Forest) on the imbalanced data and see the performance.

# In[20]:


# Using SKLEARN module for random forest
from sklearn.ensemble import RandomForestClassifier 

# Fit and predict
naive_rfc = RandomForestClassifier() 
naive_rfc.fit(X_train, y_train) 
naive_test_preds = naive_rfc.predict(X_test)


# For the performance let's use some metrics from SKLEARN module
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
  
print("The accuracy is {}".format(accuracy_score(y_test, naive_test_preds) )) 
print("The precision is {}".format(precision_score(y_test, naive_test_preds)))
print("The recall is {}".format(recall_score(y_test, naive_test_preds) )) 


# In[21]:


print("The accuracy is {}".format(f1_score(y_test, naive_test_preds) )) 


# One thing to notice here is, we had only 0.17% cases with fraud transactions and a model predicting all trasactions to be valid would have similar accuracy. So we need to train our model in a way that is not overfitted to either of the classes. for this, we introduce Oversampling and Undersampling methods. 
# Oversampling resamples from the minority class to balance the class proportions. And undersampling merges or removes similar observations from the majority to achive the same.

# # Undersampling
# 

# Now we first describe the structure of the modelling and validations. One trivial point to note is, we will not undersample the test data as we want our model to perform well with skewed class distributions eventually.
# The steps are as follows (The whole set-up will be structured using the imbalance-learn module): 
# * Use a 5-fold cross validation on the training set
# * On each of the folds use undersampling 
# * Fit the model on the training folds and validate on the validation fold

# In[22]:


# Create the cross validation framework 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV

kf = StratifiedKFold(n_splits=5, random_state = 42, shuffle = True)


# In[23]:


# Import the imbalance Learn module
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Now that we have all the libraries and module imported, we start with the classifiers.

# # Undersampling - Logistic Regression

# In[24]:


# Logistic Regression 
imba_pipeline = make_pipeline(NearMiss(), 
                              LogisticRegression())

log_reg_params = {"penalty": ['l1', 'l2'], 
                  'C': [ 0.01, 0.1, 1, 100], 
                  'solver' : ['liblinear']}

new_params = {'logisticregression__' + key: log_reg_params[key] for key in log_reg_params}
#grid_imba_log_reg = GridSearchCV(imba_pipeline, param_grid=new_params, cv=kf, scoring='recall',
#                        return_train_score=True)


grid_imba_log_reg = GridSearchCV(imba_pipeline, param_grid=new_params, cv=kf, 
                                  return_train_score=True)

grid_imba_log_reg.fit(X_train, y_train);
#logistic_cv_score_us = cross_val_score(grid_imba_log_reg, X_train, y_train, scoring = 'recall', cv = kf)
logistic_cv_score_us = cross_val_score(grid_imba_log_reg, X_train, y_train, scoring = 'recall', cv = kf)


y_test_predict = grid_imba_log_reg.best_estimator_.named_steps['logisticregression'].predict(X_test)
logistic_recall_us = recall_score(y_test, y_test_predict)
logistic_accuracy_us = accuracy_score(y_test, y_test_predict)

# print("Cross Validation Score for Logistic Regression: " + str(ulogistic_cv_score.mean()))
# print("Recall Score for Logistic Regression: " + str(ulogistic_recall))
log_reg_us = grid_imba_log_reg.best_estimator_


# In[25]:


log_reg_us, logistic_cv_score_us


# In[26]:


log_reg_us, logistic_cv_score_us, logistic_recall_us, logistic_accuracy_us


# In[27]:


f1_socre_log = f1_score(y_test, y_test_predict)

recall_log = recall_score(y_test, y_test_predict)

precision_log = precision_score(y_test, y_test_predict)

print(f1_socre_log, recall_log, precision_log)


# In[28]:


f1_socre_log = f1_score(y_test, y_test_predict, average = 'weighted')

recall_log = recall_score(y_test, y_test_predict) 

precision_log = precision_score(y_test, y_test_predict)

print(f1_socre_log, recall_log, precision_log)


# In[29]:


# Cumulatively create a table for the ROC curve
from sklearn.metrics import roc_curve, roc_auc_score

result_list = []
yproba = grid_imba_log_reg.best_estimator_.named_steps['logisticregression'].predict_proba(X_test)[::, 1]
fpr, tpr, _ = roc_curve(y_test, yproba)
auc = roc_auc_score(y_test, yproba)

new_entry = {'classifiers': "Logistic Regression", 'fpr': fpr, 'tpr': tpr, 'auc': auc}
result_list.append(new_entry)

# Create DataFrame outside the loop
result_table = pd.DataFrame(result_list, columns=['classifiers', 'fpr', 'tpr', 'auc'])

result_table


# # Undersampling - Random Forest 

# In[30]:


# Define the pipeline
imba_pipeline = make_pipeline(NearMiss(), 
                              RandomForestClassifier())
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 10, 12],
    'random_state': [13]
}

new_params = {'randomforestclassifier__' + key: params[key] for key in params}
#grid_imba_rf = GridSearchCV(imba_pipeline, param_grid=new_params, cv=kf, scoring='recall',
#                        return_train_score=True)
grid_imba_rf = GridSearchCV(imba_pipeline, param_grid=new_params, cv=kf,
                        return_train_score=True)

grid_imba_rf.fit(X_train, y_train);

rfc_cv_score_us = cross_val_score(grid_imba_rf, X_train, y_train, scoring='recall', cv=kf)

y_test_predict = grid_imba_rf.best_estimator_.named_steps['randomforestclassifier'].predict(X_test)
rfc_recall_us = recall_score(y_test, y_test_predict)
rfc_accuracy_us = accuracy_score(y_test, y_test_predict)

# print("Cross Validation Score for Random Forest: " + str(urfc_cv_score.mean()))
# print("Recall Score for Random Forest: " + str(urfc_recall))
rfc = grid_imba_rf.best_estimator_


# In[31]:


rfc,rfc_recall_us, rfc_accuracy_us, rfc_cv_score_us


# In[32]:


# Cumulatively create a table for the ROC curve
yproba = grid_imba_rf.best_estimator_.named_steps['randomforestclassifier'].predict_proba(X_test)[::,1]
    
fpr, tpr, _ = roc_curve(y_test,  yproba)
auc = roc_auc_score(y_test, yproba)

new_entry = {'classifiers': "Random Forest",
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}
result_list.append(new_entry)

# Create DataFrame outside the loop
result_table = pd.DataFrame(result_list, columns=['classifiers', 'fpr', 'tpr', 'auc'])

result_table


# # Undersampling - Support Vector Classifier

# In[ ]:


# Define the pipeline
imba_pipeline = make_pipeline(NearMiss(), 
                              SVC(probability = True))
svc_params = {'C': [0.5, 0.7, 0.9, 1], 
              'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

new_params = {'svc__' + key: svc_params[key] for key in svc_params}
#grid_imba_svc = GridSearchCV(imba_pipeline, param_grid=new_params, cv=kf, scoring='recall',
#                        return_train_score=True)
grid_imba_svc = GridSearchCV(imba_pipeline, param_grid=new_params, cv=kf,
                        return_train_score=True)

grid_imba_svc.fit(X_train, y_train);

svc_cv_score_us = cross_val_score(grid_imba_svc, X_train, y_train, scoring='recall', cv=kf) 

y_test_predict = grid_imba_svc.best_estimator_.named_steps['svc'].predict(X_test)
svc_recall_us = recall_score(y_test, y_test_predict)
svc_accuracy_us = accuracy_score(y_test, y_test_predict)

# print("Cross Validation Score for Support Vector Classifier: " + str(usvc_cv_score.mean()))
# print("Recall Score for Support Vector Classifier: " + str(usvc_recall))
svc = grid_imba_svc.best_estimator_


# In[ ]:


from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
from sklearn.metrics import confusion_matrix


# In[ ]:


svc, svc_recall_us, svc_accuracy_us, svc_cv_score_us


# In[ ]:


f1_socre_svc = f1_score(y_test, y_test_predict)

recall_svc = recall_score(y_test, y_test_predict)

precision_svc = precision_score(y_test, y_test_predict)

print(f1_socre_svc, recall_svc, precision_svc)


# In[ ]:


conf = confusion_matrix(y_test, y_test_predict)
conf 


# In[ ]:


# Cumulatively create a table for the ROC curve

yproba = grid_imba_svc.best_estimator_.named_steps['svc'].predict_proba(X_test)[::,1]
    
fpr, tpr, _ = roc_curve(y_test,  yproba)
auc = roc_auc_score(y_test, yproba)

new_entry = {'classifiers': "Support Vector Classifier",
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}
result_list.append(new_entry)

# Create DataFrame outside the loop
result_table = pd.DataFrame(result_list, columns=['classifiers', 'fpr', 'tpr', 'auc'])

result_table


# # Undersampling - Decision Tree Classifier

# In[ ]:


# DecisionTree Classifier
imba_pipeline = make_pipeline(NearMiss(), 
                              DecisionTreeClassifier())

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
new_params = {'decisiontreeclassifier__' + key: tree_params[key] for key in tree_params}
#grid_imba_tree = GridSearchCV(imba_pipeline, param_grid=new_params, cv=kf, scoring='recall',
#                        return_train_score=True)
grid_imba_tree = GridSearchCV(imba_pipeline, param_grid=new_params, cv=kf, 
                        return_train_score=True)


grid_imba_tree.fit(X_train, y_train);
dtree_cv_score_us = cross_val_score(grid_imba_tree, X_train, y_train, scoring='recall', cv=kf)


y_test_predict = grid_imba_tree.best_estimator_.named_steps['decisiontreeclassifier'].predict(X_test)
dtree_recall_us = recall_score(y_test, y_test_predict)
dtree_accuracy_us = accuracy_score(y_test, y_test_predict)

# print("Cross Validation Score for Decision Tree Classifier: " + str(udtree_cv_score.mean()))
# print("Recall Score for Decision Tree Classifier: " + str(udtree_recall))
tree_clf = grid_imba_tree.best_estimator_


# In[ ]:


tree_clf, dtree_accuracy_us, dtree_recall_us, dtree_cv_score_us


# Project Conclusion/Lessons Learned: An important learning point from the online credit card fraud detection project is the realization that while machine learning algorithms offer powerful tools for identifying fraudulent activities, their effectiveness heavily relies on the quality and relevance of the data used for training. This underscores the significance of data preprocessing, feature engineering, and continuous evaluation of model performance to ensure accuracy and adaptability in real-world scenarios. Furthermore, the project emphasized the necessity of striking a balance between model complexity and interpretability, as transparent models not only facilitate trust among stakeholders but also offer insights into underlying financial fraud patterns. 
# Overall, this project highlighted the iterative nature of machine learning application in financial fraud detection, underscoring the importance of agility, collaboration, and domain expertise in developing robust solutions to combat financial fraud in online transactions. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:





# In[ ]:





# In[ ]:




