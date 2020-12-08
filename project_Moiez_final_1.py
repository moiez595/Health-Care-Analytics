#!/usr/bin/env python
# coding: utf-8

# In[186]:


import pandas as pd


# In[187]:


data = pd.read_csv("train_data1.csv", engine= "python")


# In[188]:


data.head(10)


# In[189]:


data.info()


# In[190]:


pip install plotly_express==0.4.1


# In[191]:


data.isnull().sum()


# In[192]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[193]:



age_lst = data["Age"].unique()

age_lst.sort()
age_dict = dict(zip(age_lst, range(len(age_lst))))
data["new_age"]=data["Age"].replace(age_dict)
print(age_dict)
    


# In[194]:


data.head()


# In[195]:


stay_list = data["Stay"].unique()
stay_list.sort()
dept_Stay = dict(zip(stay_list, range(len(stay_list))))
data["new_stay"]= data["Stay"].replace(dept_Stay)
print(dept_Stay)


# In[196]:


data.head()


# In[197]:


import plotly.express as px
fig = px.histogram(data, x="Department").update_xaxes(categoryorder="total descending")
fig.show()


# In[198]:


plt.figure(figsize=(15, 6))
sns.countplot(data.Stay)


# In[199]:



import plotly.express as px
fig = px.histogram(data, x="Department",).update_xaxes(categoryorder="total descending")

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()


# In[200]:


scat1 = data.groupby(['Age','Stay']).count().reset_index()
scat1['count']=y_val['Hospital_code']

import plotly.express as px
df = px.data.gapminder()

fig = px.scatter(scat1, x="Age", y="Stay",
      size="count")
fig.show()


# In[201]:


fig2 = px.histogram(data, x="Hospital_type_code").update_xaxes(categoryorder="total descending")
fig2.show()


# In[202]:


fig3 = px.histogram(data, x="Type of Admission").update_xaxes(categoryorder="total descending").update_xaxes(categoryorder="total descending")
fig3.show()


# In[206]:


beds = data.groupby('Hospital_region_code')['Available Extra Rooms in Hospital'].sum().reset_index()
fig4=px.pie(beds,values='Available Extra Rooms in Hospital',names='Hospital_region_code',hole=0.4)
fig4.update_layout(title='Number of extra rooms in each region code',title_x=0.5)
fig4.update_traces(textinfo='percent+label')


# In[207]:


data['Stay'].value_counts()


# In[208]:


import seaborn as sns
sns.set(style="darkgrid")

g=sns.barplot(y="Stay", x="Admission_Deposit", data=data,order=['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','More than 100 Days'])


# In[209]:


import plotly.express as px
df = px.data.gapminder()
 
fig4 = px.scatter(data, x="Admission_Deposit", y="Stay")
fig4.show()


# In[210]:


heatmapdata = data[['Available Extra Rooms in Hospital', 'Bed Grade', 'Visitors with Patient'
             , 'Admission_Deposit','City_Code_Patient','new_age','new_stay']]

correlation_graph = heatmapdata.corr()

sns.set(rc={'figure.figsize':(10,10)})
sns.heatmap(data = correlation_graph,annot=True)
plt.show()


# In[211]:


cormat = heatmapdata.corr()
cormat


# In[212]:


import numpy as np
import scipy.stats
x = data['new_age']
y = data['new_stay']
m = scipy.stats.pearsonr(x, y)    # Pearson's r
print( 'pearsons value is {} '.format(m) )
n = scipy.stats.spearmanr(x, y)   # Spearman's rho
print(n)
k = scipy.stats.kendalltau(x, y)
k


# In[213]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
fig = go.Figure() 
fig.add_trace(go.Box(x=data['Admission_Deposit'],
                     marker_color="silver",
                     name="Train"))


# In[214]:


fig = go.Figure() 
fig.add_trace(go.Box(x=data['new_age'],
                     marker_color="green",
                     name="Train"))
fig.update_layout(title="Distributions of Age")
fig.show()


# In[215]:


fig = go.Figure() 
fig.add_trace(go.Box(x=data['new_stay'],
                     marker_color="green",
                     name="Train"))
fig.update_layout(title="duration of stay")
fig.show()


# In[216]:


p = data.groupby('Department')['Available Extra Rooms in Hospital'].agg('count')
q = data.groupby('Type of Admission')['Available Extra Rooms in Hospital'].agg('count')
p,q


# In[217]:


data.groupby('Severity of Illness')['Available Extra Rooms in Hospital'].agg('count')


# In[218]:


data.groupby('Severity of Illness')['Bed Grade'].agg('mean')


# In[219]:


px.pie(data,values='Available Extra Rooms in Hospital',names='Department',title='Distribution of Extra Rooms in Departments')


# In[220]:


px.pie(data,values='Available Extra Rooms in Hospital',names='Bed Grade',title='Distribution of Bed in extra rooms')


# In[221]:


px.pie(data,values='patientid',names='Age',title='Distribution of Age in Patients')


# In[222]:


px.pie(data,values='patientid',names='Stay',title='Distribution of Stay Length of Patients')


# In[223]:


data.to_csv('new_mz.csv')


# In[224]:


numerical_data = data[['Available Extra Rooms in Hospital', 'Bed Grade', 'Visitors with Patient'
             , 'Admission_Deposit', 'new_age','new_stay']]


# In[225]:


fig, new_plot =plt.subplots(3,2, figsize=(14,10))
fig.tight_layout(pad=5.0)

for new_plot, n in zip(new_plot.flatten(), numerical_data.columns.tolist()):
    sns.distplot(ax=new_plot, a=numerical_data[n], label="Skewness = %.2f"%(numerical_data[n].skew()))
    new_plot.set_title(n, fontsize = 14)
    new_plot.legend(loc = 'best')
    


# In[226]:


fig = go.Figure() 
fig.add_trace(go.Box(x=numerical_data['new_age'],
                     marker_color="green",
                     name="Train"))
fig.update_layout(title="age")
fig.show()


# In[227]:


from scipy import stats
import numpy as np
z = np.abs(stats.zscore(numerical_data))
print(z)


# In[228]:


threshold = 3
print (np.where(z > 3))


# In[ ]:





# In[229]:


print('Train columns :\n',data.columns)
print('Train shape : ', data.shape)
print('\n')


# In[230]:


new1 = data.drop(['case_id', 'Hospital_code','Age', 'City_Code_Hospital', 'City_Code_Patient'
            , 'Hospital_type_code', 'Hospital_region_code','Stay', 'Ward_Type', 'Ward_Facility_Code']
           , axis = 1)


# In[231]:



    TOA_lst = new1["Type of Admission"].unique()
    TOA_lst.sort()
    TOA_dict = dict(zip(TOA_lst, range(len(TOA_lst))))
    new1["Type of Admission"].replace(TOA_dict, inplace=True)
    print(TOA_dict)


# In[232]:


new1


# In[233]:


new2= pd.get_dummies(new1,columns=['Department','Severity of Illness'],drop_first= True)


# In[234]:


new3 = new2.dropna()


# In[235]:


new3.head()


# In[ ]:





# In[237]:



column_names = ['new_stay','new_age','Available Extra Rooms in Hospital','Bed Grade',
                'patientid','Type of Admission','Visitors with Patient','Admission_Deposit',
                'Department_anesthesia','Department_gynecology',
                'Department_radiotherapy','Department_surgery','Severity of Illness_Minor','Severity of Illness_Moderate'
               ]

new3 = new3.reindex(columns=column_names)


# In[238]:


new3.head()


# In[239]:


x_train = new3.iloc[:, 1:].values
y_train = new3.iloc[:, 0].values


# In[240]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train, test_size = 0.3, random_state = 0)
clf = RandomForestClassifier(n_estimators=300, max_depth = 20, min_samples_leaf= 10, max_features=0.5)
clf.fit(x_train_split, y_train_split)
y_pred = clf.predict(x_val_split)
accuracy = accuracy_score(y_pred, y_val_split)
print('Accuracy :',accuracy)


# In[241]:


# Fit the model into the whole data train
clf.fit(x_train, y_train)

