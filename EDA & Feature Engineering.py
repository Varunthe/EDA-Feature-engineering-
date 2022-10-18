#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#comment
#observations


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[5]:


data= pd.read_csv('student.csv')


# In[6]:


data


# In[7]:


data.head()


# In[8]:


data.tail()


# In[10]:


data.shape


# In[11]:


data.info()


# In[12]:


data['gender'].dtypes


# In[13]:


data.columns


# In[15]:


[fea for fea in data.columns]


# In[21]:


cat_col= [fea for fea in data.columns if data[fea].dtype == 'O']


# In[22]:


num_col= [fea for fea in data.columns if data[fea].dtype != 'O']


# In[23]:


data[num_col]


# In[25]:


data[cat_col]


# In[27]:


data.memory_usage()


# # missing value

# In[30]:


data.isnull()


# In[31]:


data.isnull().sum()


# In[32]:


data.isnull().sum().sum()


# # duplicate value

# In[33]:


data.duplicated()


# In[34]:


data.isnull().sum().sum()


# # unique value

# In[35]:


data.nunique()


# In[36]:


data.describe()


# In[37]:


data.describe().T


# In[38]:


data.corr()


# In[39]:


data.cov()


# In[40]:


data.skew()


# In[41]:


sns.distplot(data['math score'])


# In[42]:


(data['math score'] + data['reading score']+ data['writing score'])/3


# In[47]:


data['average']= (data['math score'] + data['reading score']+ data['writing score'])/3


# In[48]:


data


# In[49]:


data.groupby('gender')


# In[50]:


data.groupby('gender').mean()


# In[51]:


data.groupby('gender').count()


# # Question: Find out number of students whoever is having less than 30 marks in maths

# In[53]:


data[data['math score']< 30 ]


# In[54]:


data[data['math score']< 30 ].count()


# In[58]:


data_num=data[num_col]


# In[59]:


data_num


# In[60]:


data_num.head()


# In[61]:


from scipy.stats import normaltest


# In[62]:


normaltest(data_num['math score'])[1]


# In[63]:


normaltest(data_num['math score'])[1]*100


# In[66]:


##if p> 0.05 then my data will be normally distributed


# In[65]:


sns.distplot(data['math score'])


# In[67]:


sns.displot(data['math score'])


# # Finding Outliers

# In[68]:


sns.boxplot(data=data['math score'])


# In[69]:


sns.boxplot(data=data['writing score'])


# In[70]:


sns.distplot(data['reading score'])


# In[71]:


sns.distplot(data['writing score'])


# In[96]:


q1=data['math score'].quantile(0.25)


# In[97]:


q3=data['math score'].quantile(0.75)


# In[98]:


data['math score'].min()


# In[99]:


data['math score'].max()


# In[100]:


data['math score'].quantile(1.00)


# In[101]:


IQR=q3-q1


# In[102]:


IQR


# In[103]:


upper_limit=q3+(1.5*IQR)


# In[104]:


upper_limit


# In[105]:


lower_limit=q1-(1.5*IQR)


# In[106]:


lower_limit


# In[107]:


data_outlier=data[data['math score']<lower_limit]


# In[108]:


data_outlier


# In[110]:


data[data['math score']>upper_limit]


# # Graph Analysis

# In[111]:


data


# In[112]:


sns.countplot(data['gender'])


# In[113]:


sns.countplot(data['race/ethnicity'])


# In[114]:


df=data.groupby('gender').mean()


# In[115]:


df


# In[121]:


df['average'][0]


# In[122]:


df['math score'][0]


# In[123]:


df['math score'][1]


# In[125]:


plt.figure(figsize=(10,10))
X=['Total avg','math avg']
female_score=df['average'][0],df['math score'][0]
male_score=df['average'][1],df['math score'][1]
X_axis=np.arange(len(X))
plt.bar(X_axis-0.2,male_score,0.4,label='male')
plt.bar(X_axis+0.2,female_score,0.4,label='female')

plt.xticks(X_axis,X)
plt.ylabel("marks")
plt.title("total avg vs math avg",fontweight='bold')
plt.legend()
plt.show()


# In[129]:


sns.heatmap(df.corr(), annot=True, cmap='icefire')


# In[132]:


sns.heatmap(data_num.corr(),annot=True,cmap='icefire',linewidths=0.3)
fig=plt.gcf()
fig.set_size_inches(15,10)
plt.title("corr between variable",color='black',size=25)
plt.show()


# In[133]:


sns.pairplot(data_num)


# In[134]:


sns.violinplot(data=data_num)


# In[136]:


sns.violinplot(data=data)


# In[ ]:




