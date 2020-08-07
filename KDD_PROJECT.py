#!/usr/bin/env python
# coding: utf-8

# # KDD FINAL PROJECT
# 
#     CHRAN SURESH - 10450732     SHREYA VHADADI - 10453495    PRITIHIV DEV - 10453922

# ## Online Shopper's Intention
# The main idea is to find the customers intention of whether he/she would buy the particular product given the features.

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings  
warnings.filterwarnings('ignore')
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


# In[36]:


data = pd.read_csv("online_shoppers_intention.csv")


# In[37]:


data.Revenue.unique().tolist()


# #### The dataset is a binary class with imbalanced data with Revenue- True with lesser sample which depicts the customers who have bought products from the ecommerce website

# In[57]:


plt.pie([len(data[data["Revenue"]==True]),len(data[data["Revenue"]==False])],labels=["True","False"])


# #### The dataset is found to have 14 values in each column and found out that all the 14 values like in the same column, so we decided to remove the corresponding 14 records

# In[4]:


data.isnull().sum()


# In[5]:


data = data.dropna()


# In[6]:


data.isnull().sum()


# #### The dataset is divided for features(x) and labels(y) and the values are label encoder(distinct string values changed to numeric integers)

# In[7]:


x = data.iloc[:,0:17]
y = data.iloc[:,17:]


# In[8]:


le = LabelEncoder()


# In[9]:


x['Month'] = le.fit_transform(x['Month'])
x['VisitorType'] = le.fit_transform(x['VisitorType'])
x['Weekend'] = le.fit_transform(x['Weekend'])


# In[10]:


y['Revenue'] = le.fit_transform(y['Revenue'])


# #### The correlation matrix is drawn even though the correlation was high for two of the values there literal meaning had been different wherein we decided not to remove the data

# In[13]:


f,ax = plt.subplots(figsize=(13,13))
corr = data.corr()
fig = sns.heatmap(corr,mask = np.zeros_like(corr,dtype=np.bool),cmap = sns.diverging_palette(220,10,as_cmap=True),square = True,ax = ax)
fig.figure.savefig("corr.png")


# #### The train and test data is split to with a 80-20 ratio

# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test = train_test_split(x, y,test_size = 0.2, random_state =42 )


# #### The baseline model is defined with random prediction 

# In[15]:


random_prediction = np.random.randint(2, size = len(y_test))


# In[16]:


accuracy = accuracy_score(random_prediction, y_test)
print("The Accuracy for Random Prediction: ",accuracy)


# #### The test data is compared with all true

# In[17]:


ones = np.ones(len(y_test))


# In[18]:


accuracy = accuracy_score(ones, y_test)
print("The Accuracy for all Ones Prediction: ",accuracy)


# #### The test data is compared with all false

# In[19]:


zeros = np.zeros(len(y_test))


# In[20]:


accuracy = accuracy_score(zeros, y_test)
print("The Accuracy for all Zeros Prediction: ",accuracy)


# # KNN

# In[21]:


prec = []
values = []
for i in range(1,11):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train, y_train)
    pred_knn = neigh.predict(x_test)
    print("\nThe number of neighbors :",i)
    print(classification_report(pred_knn,y_test))
    precision_knn,_,_,_ = precision_recall_fscore_support(pred_knn,y_test)
    prec.append(precision_knn[1])
    values.append(i)


# In[22]:


fig = plt.figure()
plt.plot(values,prec)
plt.xlabel("No.of K's")
plt.ylabel("Precision of Class 1")
fig.savefig('k values vs precision.png')


# In[23]:


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)
pred_knn = neigh.predict(x_test)
print(classification_report(pred_knn,y_test))


# # Decision Tree

# In[24]:


trees = DecisionTreeClassifier()
trees.fit(x_train, y_train)
pred_dt = trees.predict(x_test)
print(classification_report(pred_dt,y_test))


# In[32]:


a = x.columns
plt.bar(a,trees.feature_importances_, align='center')
plt.xticks(rotation=90)
plt.show()


# # Random Forest

# In[17]:


prec = []
values = []
for i in range(100,1100,100):
    rf = RandomForestClassifier(n_estimators = i)
    rf.fit(x_train,y_train)
    pred_rf = rf.predict(x_test)
    precision_rf,_,_,_ = precision_recall_fscore_support(pred_rf,y_test)
    prec.append(precision_rf[1])
    values.append(i)
    print(classification_report(pred_rf,y_test))


# In[18]:


fig = plt.figure()
plt.plot(values,prec)
plt.xlabel("No.of Tree's(in 100's)")
plt.ylabel("Precision of Class 1")
fig.savefig('No. of trees vs precision.png')


# In[19]:


rf = RandomForestClassifier(n_estimators = 300)
rf.fit(x_train,y_train)
pred_rf = rf.predict(x_test)
print(classification_report(pred_rf,y_test))


# In[23]:


a = x.columns
plt.bar(a,rf.feature_importances_, align='center')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# from sklearn.externals.six import StringIO  
# from IPython.display import Image  
# from sklearn.tree import export_graphviz
# import pydotplus
# dot_data = StringIO()
# export_graphviz(trees, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Image(graph.create_png())


# # ANN

# In[126]:


ann = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(8))
ann.fit(x_train, y_train)
nn_prediction = ann.predict(x_test)
print(classification_report(nn_prediction,y_test))


# # K MEANS

# In[91]:


wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10 , random_state = 0)
    kmeans.fit(x_train)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.show()


# In[92]:


kmeans = KMeans(n_clusters = 4,init = 'k-means++',max_iter = 300, n_init = 10, random_state =0)
y_kmeans = kmeans.fit_predict(x)


# # SVM

# In[145]:


from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


# In[152]:


svm = LinearSVC()
svm.fit(x_train, y_train)
pred_svm = svm.predict(x_test)
print(classification_report(pred_svm,y_test))


# # Logistic Regression

# In[153]:


clf = LogisticRegression()
clf.fit(x_train, y_train)
predicts = clf.predict(x_test)
print(classification_report(predicts,y_test))


# # Random Final

# #### The final model is calculated with the most important features that shows a similar precision to the all feature model.

# In[188]:


x_trains = x_train.iloc[:,[7,8]]


# In[189]:


x_tests = x_test.iloc[:,[7,8]]


# In[190]:


rf = RandomForestClassifier(n_estimators = 300)
rf.fit(x_trains,y_train)
pred_rf = rf.predict(x_tests)
print(classification_report(pred_rf,y_test))


# # Naive Bayes

# In[192]:


from sklearn.naive_bayes import GaussianNB


# In[193]:


gb = GaussianNB()
gb.fit(x_train,y_train)
pred_gb = gb.predict(x_test)
print(classification_report(pred_gb,y_test))

