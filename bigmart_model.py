#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv("Train.csv")


# In[4]:


data.sample(5)


# ### Find Shape of Our Dataset (Number of Rows And Number of Columns)

# In[5]:


data.shape


# ### Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement

# In[6]:


data.describe()


# ### Check Null Values In The Dataset

# In[7]:


data.isnull().sum()


# In[8]:


per = data.isnull().sum() * 100 / len(data)
print(per)


# ### Taking Care of Duplicate Values

# In[9]:


data.duplicated().any()


# In[ ]:





# ### Handling The missing Values

# In[10]:


data['Item_Weight']


# In[11]:


data['Outlet_Size']


# ### Univariate Imputation

# In[12]:


mean_weight = data['Item_Weight'].mean()
median_weight = data['Item_Weight'].median()


# In[13]:


print(mean_weight,median_weight)


# In[14]:


data['Item_Weight_mean']=data['Item_Weight'].fillna(mean_weight)
data['Item_Weight_median']=data['Item_Weight'].fillna(median_weight)


# In[15]:


data.head(1)


# In[16]:


print("Original Weight variable variance",data['Item_Weight'].var())
print("Item Weight variance after mean imputation",data['Item_Weight_mean'].var())
print("Item Weight variance after median imputation",data['Item_Weight_median'].var())


# In[17]:


data['Item_Weight'].plot(kind = "kde",label="Original")

data['Item_Weight_mean'].plot(kind = "kde",label = "Mean")

data['Item_Weight_median'].plot(kind = "kde",label = "Median")

plt.legend()
plt.show()


# In[18]:


data[['Item_Weight','Item_Weight_mean','Item_Weight_median']].boxplot()


# In[19]:


data['Item_Weight_interploate']=data['Item_Weight'].interpolate(method="linear")


# In[20]:


data['Item_Weight'].plot(kind = "kde",label="Original")

data['Item_Weight_interploate'].plot(kind = "kde",label = "interploate")

plt.legend()
plt.show()


# In[ ]:





# ### Multivariate Imputaion

# In[21]:


from sklearn.impute import KNNImputer # type: ignore


# In[22]:


knn = KNNImputer(n_neighbors=10,weights="distance")


# In[23]:


data['knn_imputer']= knn.fit_transform(data[['Item_Weight']]).ravel()


# In[24]:


data['Item_Weight'].plot(kind = "kde",label="Original")

data['knn_imputer'].plot(kind = "kde",label = "KNN imputer")

plt.legend()
plt.show()


# In[25]:


data = data.drop(['Item_Weight','Item_Weight_mean','Item_Weight_median','knn_imputer'],axis=1)


# In[26]:


data.head(1)


# In[27]:


data.isnull().sum()


# ### Outlet_Size 

# In[28]:


data['Outlet_Size'].value_counts()


# In[29]:


data['Outlet_Type'].value_counts()


# In[30]:


mode_outlet = data.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x:x.mode()[0]))


# In[31]:


mode_outlet


# In[32]:


missing_values = data['Outlet_Size'].isnull()


# In[33]:


missing_values


# In[34]:


data.loc[missing_values,'Outlet_Size'] = data.loc[missing_values,'Outlet_Type'].apply(lambda x :mode_outlet[x])


# In[35]:


data.isnull().sum()


# ### Item_Fat_Content

# In[36]:


data.columns


# In[37]:


data['Item_Fat_Content'].value_counts()


# In[38]:


data.replace({'Item_Fat_Content':{'Low Fat':'LF','low fat':'LF','reg':'Regular'}},inplace=True)


# In[39]:


data['Item_Fat_Content'].value_counts()


# ### Item_Visibility

# In[40]:


data.columns


# In[41]:


data['Item_Visibility'].value_counts()


# In[42]:


data['Item_Visibility_interpolate']=data['Item_Visibility'].replace(0,np.nan).interpolate(method='linear')


# In[43]:


data.head(1)


# In[44]:


data['Item_Visibility_interpolate'].value_counts()


# In[45]:


data['Item_Visibility'].plot(kind="kde",label="Original")

data['Item_Visibility_interpolate'].plot(kind="kde",color='red',label="Interpolate")

plt.legend()
plt.show()


# In[46]:


data = data.drop('Item_Visibility',axis=1)


# In[47]:


data.head(1)


# ### Item_Type

# In[48]:


data.columns


# In[49]:


data['Item_Type'].value_counts()


# ### Item_Identifier

# In[50]:


data.columns


# In[51]:


data['Item_Identifier'].value_counts().sample(5)


# In[52]:


data['Item_Identifier'] =data['Item_Identifier'].apply(lambda x : x[:2])


# In[53]:


data['Item_Identifier'].value_counts()


# ### Outlet_Establishment_Year

# In[54]:


data.columns


# In[55]:


data['Outlet_Establishment_Year']


# In[56]:


import datetime as dt


# In[57]:


current_year = dt.datetime.today().year


# In[58]:


current_year


# In[59]:


data['Outlet_age']= current_year - data['Outlet_Establishment_Year']


# In[60]:


data.head(1)


# In[61]:


data = data.drop('Outlet_Establishment_Year',axis=1)


# In[62]:


data.head()


# ### Handling Categorical Columns

# In[63]:


from sklearn.preprocessing import LabelEncoder

data_encoded = data.copy()

# List to store label encoders for each categorical column
label_encoders = {}
cat_cols = data.select_dtypes(include=['object']).columns
# Iterate over each categorical column
for col in cat_cols:
    # Create a LabelEncoder object
    label_encoders[col] = LabelEncoder()
    
    # Fit the encoder on the unique values of the column
    label_encoders[col].fit(data_encoded[col])
    
    # Transform the column using the fitted encoder
    data_encoded[col] = label_encoders[col].transform(data_encoded[col])

    # Print the categories for reference
    print(f'Categories for {col}: {label_encoders[col].classes_}')


# In[64]:


data_encoded.head(3)


# In[65]:


X = data_encoded.drop(['Item_Outlet_Sales' ], axis=1)
y = data_encoded['Item_Outlet_Sales']


# In[66]:


y


# ### Random Forest Regressor

# In[67]:


from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(n_estimators=100,random_state=42)
scores = cross_val_score(rf,X,y,cv=5,scoring='r2')
print(scores.mean())


# In[68]:


from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score

# Lasso regression
lasso = Lasso(alpha=0.1, random_state=42)
lasso_scores = cross_val_score(lasso, X, y, cv=5, scoring='r2')
print("Lasso R2 Score:", lasso_scores.mean())

# Ridge regression
ridge = Ridge(alpha=0.1, random_state=42)
ridge_scores = cross_val_score(ridge, X, y, cv=5, scoring='r2')
print("Ridge R2 Score:", ridge_scores.mean())


# ### XGBRFRegressor

# In[69]:


from xgboost import XGBRFRegressor

xg = XGBRFRegressor(n_estimators=100,random_state=42)
scores = cross_val_score(xg,X,y,cv=5,scoring='r2')
print(scores.mean())


# ### XGBRFRegressor Feature importances

# In[70]:


xg = XGBRFRegressor(n_estimators=100,random_state=42)

xg1 = xg.fit(X,y)
pd.DataFrame({
    'feature':X.columns,
    'XGBRF_importance':xg1.feature_importances_
    
}).sort_values(by='XGBRF_importance',ascending=True)


# In[71]:


['Item_Visibility_interpolate','Item_Weight_interploate',
'Item_Type','Outlet_Location_Type','Item_Identifier','Item_Fat_Content']


# In[72]:


from xgboost import XGBRFRegressor

xg = XGBRFRegressor(n_estimators=100,random_state=42)
scores = cross_val_score(xg1,X.drop(['Item_Visibility_interpolate','Item_Weight_interploate',
'Item_Type','Outlet_Location_Type','Item_Identifier','Item_Fat_Content'],axis=1),y,cv=5,scoring='r2')
print(scores.mean())


# In[73]:


final_data = X.drop(columns=['Item_Visibility_interpolate','Item_Weight_interploate',
'Item_Identifier','Item_Fat_Content','Outlet_age'],axis=1)




# In[74]:


final_data


# In[ ]:





# ### Best Model

# In[75]:


from xgboost import XGBRFRegressor


# In[76]:


xg_final = XGBRFRegressor()


# In[77]:


xg_final.fit(final_data,y)


# In[78]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[79]:


X_train,X_test,y_train,y_test = train_test_split(final_data,y,
                                                 test_size=0.20,
                                                 random_state=42)


# In[80]:


xg_final.fit(X_train,y_train)


# In[81]:


y_pred = xg_final.predict(X_test)


# In[82]:


mean_absolute_error(y_test,y_pred)


# In[83]:


plt.figure(figsize=(30,6))
sns.countplot(x="Item_Type",data=data)
plt.show


# ### Prediction on Unseen Data

# In[84]:


pred = xg_final.predict(np.array([[9.0,141.6180,9.0,1.0,1.0,1.0]]))[0]
print(pred)


# In[85]:


print(f"Sales Value is between {pred-714.42} and {pred+714.42}")


# ### Save Model Using Joblib

# In[86]:


import joblib


# In[87]:


joblib.dump(xg_final,'bigmart_model')

