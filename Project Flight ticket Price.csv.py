#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[3]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
df_train=pd.read_excel('flight_data_Train.xlsx')
df_test=pd.read_excel('flight_test.xlsx')


# In[4]:


df_train.head()


# In[5]:


df_train.describe()


# In[6]:


df_train.info()


# In[7]:


df_train.isnull().sum()


# In[8]:


df_train.dropna(inplace=True)


# In[9]:


df_train.isnull().sum()


# In[10]:


df_train['Date_of_Journey'].value_counts()


# In[11]:


df_train['Journey_day']=pd.to_datetime(df_train.Date_of_Journey,format='%d/%m/%Y').dt.day
df_train['Journey_month']=pd.to_datetime(df_train.Date_of_Journey,format='%d/%m/%Y').dt.month


# In[12]:


df_train.head()


# In[13]:


df_train.drop(['Date_of_Journey'],axis=1,inplace=True)


# In[14]:


df_train.head()


# In[15]:


df_train['Dep_hour']=pd.to_datetime(df_train.Dep_Time).dt.hour
df_train['Dep_min']=pd.to_datetime(df_train.Dep_Time).dt.minute


# In[16]:


df_train.head()


# In[17]:


df_train.drop(['Dep_Time'],axis=1,inplace=True)
df_train.head()


# In[18]:


df_train['Arrival_hour']=pd.to_datetime(df_train.Arrival_Time).dt.hour
df_train['Arrival_min']=pd.to_datetime(df_train.Arrival_Time).dt.minute


# In[19]:


df_train.drop(['Arrival_Time'],axis=1,inplace=True)


# In[20]:


df_train.head()


# In[21]:



    
    
duration = list(df_train["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   
        else:
            duration[i] = "0h " + duration[i]           

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   
    


# In[22]:


df_train['duration_hours']=duration_hours
df_train['duration_mins']=duration_mins


# In[23]:


df_train.drop(['Duration'],axis=1,inplace=True)


# In[24]:


df_train['Airline'].value_counts()


# In[25]:



sns.boxplot(x='Airline',y='Price',data=df_train.sort_values('Price',ascending=False))


# In[26]:


airline=df_train[['Airline']]
airline=pd.get_dummies(airline,drop_first=True)
airline.head()


# In[27]:



sns.boxplot(x='Source',y='Price',data=df_train.sort_values('Price',ascending=False))


# In[28]:


source=df_train[['Source']]
source=pd.get_dummies(source,drop_first=True)
source.head()


# In[29]:


df_train.head()


# In[30]:


df_train['Route'].value_counts()


# In[31]:


df_train.drop(['Route','Additional_Info'],axis=1,inplace=True)


# In[32]:


df_train.head()


# In[33]:


df_train['Total_Stops'].value_counts()


# In[34]:


df_train.replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4},inplace=True)


# In[35]:


df_train['Destination'].value_counts()


# In[36]:


destination=df_train[['Destination']]
destination=pd.get_dummies(destination,drop_first=True)
destination.head()


# In[37]:


df_train.head()


# In[38]:


df_train=pd.concat([df_train,airline,source,destination],axis=1)


# In[39]:


df_train.head()


# In[40]:


df_train.drop(['Airline','Source','Destination'],axis=1,inplace=True)


# In[41]:


df_train.head()


# In[42]:


df_train.shape


# In[43]:


df_test=pd.read_excel('flight_test.xlsx')


# In[44]:


df_test.head()


# In[45]:


df_test.info()


# In[46]:


df_test.isnull().sum()


# In[47]:


df_test['Journey_day']=pd.to_datetime(df_test.Date_of_Journey,format='%d/%m/%Y').dt.day
df_test['Journey_month']=pd.to_datetime(df_test.Date_of_Journey,format='%d/%m/%Y').dt.month
df_test.drop(['Date_of_Journey'],axis=+1,inplace=True)


# In[48]:


df_test["Dep_hour"] = pd.to_datetime(df_test["Dep_Time"]).dt.hour
df_test["Dep_min"] = pd.to_datetime(df_test["Dep_Time"]).dt.minute
df_test.drop(["Dep_Time"], axis = 1, inplace = True)


# In[49]:


df_test["Arrival_hour"] = pd.to_datetime(df_test.Arrival_Time).dt.hour
df_test["Arrival_min"] = pd.to_datetime(df_test.Arrival_Time).dt.minute
df_test.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[50]:


duration = list(df_test["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   
        else:
            duration[i] = "0h " + duration[i]           

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[51]:


df_test["Duration_hours"] = duration_hours
df_test["Duration_mins"] = duration_mins
df_test.drop(["Duration"], axis = 1, inplace = True)


# In[52]:


airline=df_test[['Airline']]
airline=pd.get_dummies(airline,drop_first=True)
airline.head()


# In[53]:


source=df_test[['Source']]
source=pd.get_dummies(source,drop_first=True)
source.head()


# In[54]:


destination=df_test[['Destination']]
destination=pd.get_dummies(destination,drop_first=True)
destination.head()


# In[55]:


df_test.replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4},inplace=True)


# In[56]:


df_test.head()


# In[57]:


df_test.drop(['Route','Additional_Info'],axis=1,inplace=True)


# In[58]:


df_test.head()


# In[59]:


df_test=pd.concat([df_test,airline,source,destination],axis=1)


# In[60]:


df_test.drop(['Airline','Source','Destination'],axis=1,inplace=True)


# In[61]:


df_test.head()


# In[62]:


df_test.shape


# In[63]:


df_train.columns


# In[64]:


x=df_train.loc[:,['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'duration_hours',
       'duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
x.head()


# In[65]:


y=df_train['Price']
y.shape


# In[66]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
sns.heatmap(df_train.corr(),annot=True)


# In[67]:


from sklearn.ensemble import ExtraTreesRegressor
etr=ExtraTreesRegressor(n_estimators=100)
etr.fit(x,y)


# In[68]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)


# In[69]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)
pred=rfr.predict(x_test)
pred


# In[73]:


rfr.score(x_train,y_train)


# In[72]:


rfr.score(x_test,y_test)


# In[78]:


sns.distplot(y_test-pred)


# In[79]:


from sklearn import metrics


# In[84]:


print('MAE:',metrics.mean_absolute_error(y_test,pred))
print('MSE:',metrics.mean_squared_error(y_test,pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,pred)))


# In[85]:


metrics.r2_score(y_test,pred)


# In[106]:


from sklearn.model_selection import RandomizedSearchCV


# In[107]:


estimator=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
max_feat=['auto','sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


# In[108]:


random_grid = {'n_estimators': estimator,
               'max_features': max_feat,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[109]:


rfr_random = RandomizedSearchCV(estimator = rfr, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[111]:


rfr_random.fit(x_train,y_train)


# In[115]:


rfr_random.best_params_


# In[117]:


ped_r=rfr_random.predict(x_test)


# In[120]:


sns.distplot(y_test-ped_r)


# In[124]:


print('MAE:',metrics.mean_absolute_error(y_test,ped_r))
print('MSE:',metrics.mean_squared_error(y_test,ped_r))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,ped_r)))


# In[133]:


import pickle
file=open('flight_rf.pkl','wb')
pickle.dump(rfr,file)


# In[134]:


model=open('flight_rf.pkl','rb')
forest=pickle.load(model)


# In[137]:


pred_n=forest.predict(x_test)
pred_n


# In[138]:


metrics.r2_score(y_test,ped_r)


# In[ ]:




