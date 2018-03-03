
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import random
import cv2
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[4]:


train.info()


# In[5]:


test.head()


# In[6]:


PassengerId = test['PassengerId']


# In[7]:


full_data = [train, test]


# In[8]:


#Length of the person's name 
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)


# In[9]:


#Find if passenger has a cabin 
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)


# In[10]:


#Creating a new column Family Size (sum of siblings and parents present )
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


# In[11]:


#Categorise into whether Alone or NotAlone
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


# In[12]:


dataset.head()


# In[13]:


#Fill the non-filled embarkations as S(South Hampton)
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[14]:


#Find missing values for Fare and fill them and then categorise into 4 groups

for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)    


# In[15]:


#Generate random numbers to fill NaN values and separate into different age groups 

for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)


# In[16]:


# train['Name_Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
# train['Name_Title'].value_counts()
#train['Survived'].groupby(train['Name_Title']).mean()


# In[17]:


#More men population
print(train['Sex'].value_counts(normalize=True))

#But more women survived than men 
print(train['Survived'].groupby(train['Sex']).mean())


# In[18]:


# To split the names using the titles in names 
import re

# Function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(train['Title'], train['Sex']))



# In[19]:


# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
 
print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# In[20]:


for dataset in full_data:
    # Mapping Sex into male and female as 0 and 1
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles 
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[21]:


for dataset in full_data:
    # Mapping Fare from range of 0 to 3 
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age into 5 categories 
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 
    
    dataset.loc[dataset['Embarked']=='Q', 'Embarked'] = 0
    dataset.loc[dataset['Embarked']=='S', 'Embarked'] = 1
    dataset.loc[dataset['Embarked']=='C', 'Embarked'] = 2
    #dataset['Embarked'] = dataset['Embarked'].astype(int)


# In[22]:


# colormap = plt.cm.RdBu
# plt.figure(figsize=(14,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sb.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
#             square=True, cmap=colormap, linecolor='white', annot=True)


# In[23]:


drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin']

train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)

print (train.head(10))

train = train.values
test  = test.values


# In[24]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LogisticRegression()]

log_cols = ["Classifier", "Accuracy"]
log  = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = train[0::, 1::]
y = train[0::, 0]

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sb.set_color_codes("muted")
sb.barplot(x='Accuracy', y='Classifier', data=log, color="b")


# In[25]:


classifier_af = GradientBoostingClassifier(n_estimators=150,random_state=42,learning_rate=0.01)

classifier_af.fit(train[0::, 1::], train[0::, 0])

pred = classifier_af.predict(test)




# In[27]:


pred


# In[29]:


submission = pd.DataFrame({
        "Survived": pred
        })
submission.to_csv('s1.csv', index=False)

