#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("train_data1.csv", engine= "python")
data.head(10)


# In[4]:


test_data = pd.read_csv('test_data.csv')
test_data.head()


# In[5]:


age_lst = data["Age"].unique()

age_lst.sort()
age_dict = dict(zip(age_lst, range(len(age_lst))))
data["Age"]=data["Age"].replace(age_dict)
print(age_dict)

stay_list = data["Stay"].unique()
stay_list.sort()
dept_Stay = dict(zip(stay_list, range(len(stay_list))))
data["Stay"]= data["Stay"].replace(dept_Stay)
print(dept_Stay)

data.head()


# In[7]:


new1 = data.drop(['case_id','patientid']
           , axis = 1)


# In[15]:


import numpy as np

new_dept = new1["Department"].unique()
new_dept.sort()
new_dept = dict(zip(new_dept, range(len(new_dept))))
new1.Department.replace(new_dept, inplace= True)
print(new_dept)
    
new_hosp_code = new1["Hospital_region_code"].unique()
new_hosp_code.sort()
new_hosp_code= dict(zip(new_hosp_code, range(len(new_hosp_code))))
new1.Hospital_region_code.replace(new_hosp_code, inplace = True)
print(new_hosp_code)

new_ward_type = new1["Ward_Type"].unique()
new_ward_type.sort()
new_ward_type = dict(zip(new_ward_type, range(len(new_ward_type))))
new1.replace(new_ward_type, inplace=True)
print(new_ward_type)

new_type_admiss = new1["Type of Admission"].unique()
new_type_admiss.sort()
new_type_admiss = dict(zip(new_type_admiss, range(len(new_type_admiss))))
new1["Type of Admission"].replace(new_type_admiss, inplace=True)
print(new_type_admiss)
   
new_severity = new1["Severity of Illness"].unique()
new_severity .sort()
new_severity  = dict(zip(new_severity, range(len(new_severity ))))
new1["Severity of Illness"].replace(new_severity , inplace=True)
print(new_severity )

new_Hospital_type_code = new1["Hospital_type_code"].unique()
new_Hospital_type_code .sort()
new_Hospital_type_code  = dict(zip(new_Hospital_type_code, range(len(new_Hospital_type_code ))))
new1["Hospital_type_code"].replace(new_Hospital_type_code , inplace=True)
print(new_Hospital_type_code )

new_Ward_Facility_Code = new1["Ward_Facility_Code"].unique()
new_Ward_Facility_Code .sort()
new_Ward_Facility_Code  = dict(zip(new_Ward_Facility_Code, range(len(new_Ward_Facility_Code))))
new1["Ward_Facility_Code"].replace(new_Ward_Facility_Code , inplace=True)
print(new_Ward_Facility_Code )


# In[16]:


new1.head()


# In[94]:


import plotly.express as px
extra_room=new1.groupby('Hospital_region_code')['Available Extra Rooms in Hospital'].sum().reset_index()
fig4=px.pie(extra_room,values='Available Extra Rooms in Hospital',names='Hospital_region_code',hole=0.4)
fig4.update_layout(title='Number of extra rooms in each region code',title_x=0.5)
fig4.update_traces(textinfo='percent+label')


# In[106]:


age_plot=px.sunburst(data, path=['Age','Severity of Illness'])
age_plot.update_layout(title='Age and Severity of Illness',title_x=0.5)
age_plot.show()


# In[17]:


new2 = test_data.drop(['case_id','patientid']
           , axis = 1)


# In[20]:



new_dept = new2["Department"].unique()
new_dept.sort()
new_dept = dict(zip(new_dept, range(len(new_dept))))
new2.Department.replace(new_dept, inplace= True)
print(new_dept)
    
new_hosp_code = new2["Hospital_region_code"].unique()
new_hosp_code.sort()
new_hosp_code= dict(zip(new_hosp_code, range(len(new_hosp_code))))
new2.Hospital_region_code.replace(new_hosp_code, inplace = True)
print(new_hosp_code)

new_ward_type = new2["Ward_Type"].unique()
new_ward_type.sort()
new_ward_type = dict(zip(new_ward_type, range(len(new_ward_type))))
new2.replace(new_ward_type, inplace=True)
print(new_ward_type)

new_type_admiss = new2["Type of Admission"].unique()
new_type_admiss.sort()
new_type_admiss = dict(zip(new_type_admiss, range(len(new_type_admiss))))
new2["Type of Admission"].replace(new_type_admiss, inplace=True)
print(new_type_admiss)
   
new_severity = new2["Severity of Illness"].unique()
new_severity .sort()
new_severity  = dict(zip(new_severity, range(len(new_severity ))))
new2["Severity of Illness"].replace(new_severity , inplace=True)
print(new_severity )

new_Hospital_type_code = new2["Hospital_type_code"].unique()
new_Hospital_type_code .sort()
new_Hospital_type_code  = dict(zip(new_Hospital_type_code, range(len(new_Hospital_type_code ))))
new2["Hospital_type_code"].replace(new_Hospital_type_code , inplace=True)
print(new_Hospital_type_code )

new_Ward_Facility_Code = new2["Ward_Facility_Code"].unique()
new_Ward_Facility_Code .sort()
new_Ward_Facility_Code  = dict(zip(new_Ward_Facility_Code, range(len(new_Ward_Facility_Code))))
new2["Ward_Facility_Code"].replace(new_Ward_Facility_Code , inplace=True)
print(new_Ward_Facility_Code )

new_age = new2["Age"].unique()
new_age .sort()
new_age = dict(zip(new_age, range(len(new_age))))
new2["Age"].replace(new_age , inplace=True)
print(new_age )


# In[21]:


new2.head()


# In[25]:


pip install catboost


# # CATBOOST MODEL

# In[30]:



from catboost import CatBoostClassifier, Pool
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split


# In[23]:



X = new1.drop(columns=['Stay'])
Y = new1['Stay']


# selecting features for test data 

test_X = new2


# In[31]:


X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 0)


# In[32]:


train_dataset = Pool(data=X_train, label=Y_train)

eval_dataset = Pool(data=X_test, label=Y_test)


# In[33]:


model = CatBoostClassifier(iterations=750,
                           learning_rate=0.08,
                           depth=7,
                           loss_function='MultiClass',
                           eval_metric='Accuracy')


# In[34]:


model.fit(train_dataset)

# validation

eval_pred = model.predict(eval_dataset)


# In[35]:



model.get_best_score()


# In[43]:


from catboost.utils import get_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[44]:


cm = confusion_matrix(Y_test, eval_pred)
cm


# In[51]:



new_dataset = Pool(test_X)

y_pred = model.predict(new_dataset)


# In[52]:



output = pd.DataFrame(test_data['case_id'].values,columns=['case_id'])
output['Stay'] = y_pred
swap_dict_stay = dict([(value, key) for key, value in dept_Stay.items()])
output['Stay'].replace(swap_dict_stay, inplace=True)


# In[112]:


output.head(50)


# In[72]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='whitegrid')

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, f1_score, recall_score, precision_score


from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


# In[73]:


new1.dropna()


# In[80]:


new1.isna().sum()


# # KNN MODEL

# In[85]:



x = new1.drop(["Stay",'Bed Grade','City_Code_Patient','Hospital_code', 'Hospital_type_code',
       'Hospital_region_code' ], axis=1).to_numpy()
y = new1['Stay'].values


# In[86]:


X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = 0.2, random_state=0)


# In[107]:


neighbors = KNeighborsClassifier(n_neighbors=11) # 11 different values of Stay
neighbors.fit(X_train, Y_train)
new_Y_pred= neighbors.predict(X_val)
# get the accuracy score
acc_neigh = accuracy_score(new_Y_pred, Y_val)
print(acc_neigh)


# # RANDOM FOREST MODEL

# In[108]:


randfor = RandomForestClassifier(n_estimators=200, max_depth=15)

randfor.fit(X_train, Y_train)

pred_randfor = randfor.predict(X_val)
# get the accuracy score
accuracy = accuracy_score(Y_pred_rf, Y_val)
print(accuracy)


# # LINEAR REGRESSION

# In[109]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
linear_reg = LinearRegression()

linear_reg.fit(X_train, Y_train)

pred_linear_reg = linear_reg.predict(X_val)

accuracy = linear_reg.score(X_val,Y_val)
print(accuracy)


# In[ ]:




