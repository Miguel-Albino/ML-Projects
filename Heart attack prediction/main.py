#!/usr/bin/env python
# coding: utf-8

# # Heart Attack Prediction

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier


# In[2]:


## Import dataset
df = pd.read_csv('heart.csv')

print('Info:\n', df.head())

print('\nData Shape: \n', df.shape)

print('Description:\n', df.describe())

print('\n Sum of label events: \n', df['output'].value_counts())

print('We have an balanced dataset')


# In[3]:


#Check for null values
print('Null values: \n', df.isna().sum())

# list of categorical values
cat_list = ['sex','cp','fbs', 'restecg', 'exng','slp','caa','thall']

# check unique values of features
print('\n', df[cat_list].nunique())


# In[4]:


# Change type of categorical variables to 'category'

print('Before:\n',df.dtypes)

df[cat_list] = df[cat_list].astype('category')

print('After:\n',df.dtypes)


# In[5]:


#isolate the target - y

y = df.output
df = df.drop(['output'], axis = 1)

print(df.shape,y.shape)


# In[6]:


# Check description of numerical values
df_num = df[df.columns[~df.columns.isin(cat_list)]]
print('Description of the numerical features')
print(df_num.describe())

ax = sns.boxplot(data=df_num)
plt.show()

print('As we can see, the range of values between features are very different, so we should standardize them.')


# In[7]:


# But first let's see if it exists any correlation between them
corr_mtrx = df_num.corr()

sns.heatmap(corr_mtrx, vmin=-1, vmax=1)
plt.title('Correlation matrix')
plt.show()

print("As we can see from the corr matrx above, theres is no evident correlation between numerical features.       In this case we can not select features to drop. ")


# In[8]:


#We can now standardize our features

#list of columns that are numerical

num_list = df.select_dtypes(exclude=['category']).columns
print(df[num_list])

df[num_list] = StandardScaler().fit_transform(df[num_list])
print(df)


# Convert categorical fatures into One-Hot encoding


df = pd.get_dummies(df, columns = cat_list, drop_first=True)


# In[12]:


print(df.head())


# In[13]:


#split data into train and test
X = np.array(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size= 0.33, random_state = 42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ### *Note: article about feature selection: https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/

# ## Adjustment of hyperparameters with Cross Validation

# In[14]:


# We are using classification models: LogReg, SVM and DecisionTree
# For each model we have wide range of hyperparameters. Let's define them

param_logreg = {'penalty': ('l1','l2'), 'C' : [0.05, 0.15, 0.25, 0.35, 0.45,0.55, 0.65, 0.75,0.85, 1] }

param_svm = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10,]}

param_dt = {'max_depth': [3, 4, 5, 6, 8], 'min_samples_leaf':[0.04, 0.06, 0.08],
           'max_features': [0.2, 0.4, 0.6, 0.8]}


# In[15]:


#Now let's perform a gridsearch cv for each model and get best parameters and score

logreg = LogisticRegression(solver='liblinear')

grid_logreg = GridSearchCV(estimator = logreg, param_grid = param_logreg, scoring='accuracy', cv=5)
grid_logreg.fit(X_train, y_train)
best_logreg = grid_logreg.best_params_
print('Best Hyperparam for LogReg: \n', best_logreg)
best_logreg_score = grid_logreg.best_score_
print('With accuracy of: \n', best_logreg_score)

#extract the model w/ best hyper
best_model_logreg = grid_logreg.best_estimator_
print('Test set accuracy is: \n', best_model_logreg.score(X_test, y_test))

y_pred = grid_logreg.predict(X_test)
c_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(c_matrix)

print()

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print('Precision: ',precision)
print('Recall: ', recall)
print('F1 score: ',f1)


# In[16]:


svm = SVC(gamma='auto')
grid_svm = GridSearchCV(estimator = svm, param_grid = param_svm, cv=5, scoring='recall')
grid_svm.fit(X_train, y_train)
best_svm = grid_svm.best_params_
print('Best Hyperparam for SVM: \n', best_svm)
best_svm_score = grid_svm.best_score_
print('With accuracy of: \n', best_svm_score)

#extract the model w/ best hyper
best_model_svm = grid_svm.best_estimator_
print('Test set accuracy is: \n', best_model_svm.score(X_test, y_test))
y_pred_svm = grid_svm.predict(X_test)
c_matrix_svm = pd.crosstab(y_test, y_pred_svm, rownames=['Actual'], colnames=['Predicted'])
print(c_matrix_svm)
print()

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_svm).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print('Precision: ',precision)
print('Recall: ', recall)
print('F1 score: ',f1)


# In[17]:


dt = DecisionTreeClassifier(random_state=1)
grid_dt = GridSearchCV(estimator = dt, param_grid = param_dt, cv=5, scoring='recall')
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_params_
print('Best Hyperparam for Decision Tree: \n', best_dt)
best_dt_score = grid_dt.best_score_
print('With accuracy of: \n', best_dt_score)

#extract the model w/ best hyper
best_model_dt = grid_dt.best_estimator_
print('Test set accuracy is: \n', best_model_dt.score(X_test, y_test))
y_pred_dt = grid_dt.predict(X_test)
c_matrix_dt = pd.crosstab(y_test, y_pred_dt, rownames=['Actual'], colnames=['Predicted'])
print(c_matrix_dt)
print()
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_dt).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print('Precision: ',precision)
print('Recall: ', recall)
print('F1 score: ',f1)


# In[18]:


## Random Forest Classifier

rf = RandomForestClassifier(random_state=1)

params_rf = {'n_estimators': [100,200,300], 'max_depth': [2,3,4,6,8],
            'min_samples_leaf':[0.02, 0.04, 0.06, 0.08], 'max_features': ['log2']}

grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='accuracy', cv=5)

grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_params_
print('Best hyperparameters: \n', best_rf)
best_rf_score = grid_rf.best_score_
print('With accuracy of: \n', best_rf_score)

y_pred_rf = grid_rf.predict(X_test)
c_matrix_rf = pd.crosstab(y_test, y_pred_rf, rownames=['Actual'], colnames=['Predicted'])
print(c_matrix_rf)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
print()
print('Precision: ',precision)
print('Recall: ', recall)
print('F1 score: ',f1)


# ## Let's try another ensemble method - Voting Classifier

# In[19]:


esb_model = VotingClassifier(estimators=[('lr', best_model_logreg), ('svm', best_model_svm), ('dt', best_model_dt)], voting='hard')
esb_model.fit(X_train, y_train)
y_pred_esb = esb_model.predict(X_test)

c_matrix_esb = pd.crosstab(y_test, y_pred_esb, rownames=['Actual'], colnames=['Predicted'])
print(c_matrix_esb)

print()

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_esb).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print('Precision: ',precision)
print('Recall: ', recall)
print('F1 score: ',f1)


# ## **another artile, this time about ensemble methods: https://towardsdatascience.com/ensemble-models-5a62d4f4cb0c 
