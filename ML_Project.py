#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PERTH
import pandas as pd
df=pd.read_csv('Perth_Data.csv')


# In[2]:


print(df.tail()) 
print(df.head()) 
print(df.describe()) 
print(df.info()) 


# In[3]:


df.isna().sum()


# In[4]:


df1=df.dropna()
print(df1)


# In[5]:


y=df1['AW'].values
print(y)
x=df1.drop(['AW'],axis=1).values
print(x)


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[7]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
train_pred=lr.predict(x_train)
test_pred=lr.predict(x_test)


# In[8]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
train_error_mean=mean_squared_error(y_train,train_pred)
test_error_mean=mean_squared_error(y_test,test_pred)
train_error_r2=r2_score(y_train,train_pred)
test_error_r2=r2_score(y_test,test_pred)
print('Mean Squared Error for training:',train_error_mean)
print('Mean Squared Error for testing:',test_error_mean)
print('R2-Score for testing:',train_error_r2)
print('R2-Score for testing:',test_error_r2)


# In[9]:


print(lr.coef_)
print(lr.intercept_)


# In[10]:


#LASSO Ridge ElasticNet
import numpy as np
import pandas as pd
df2=pd.DataFrame(x)
df3=pd.DataFrame(y,columns=['Target'])
df4=pd.concat([df2,df3],axis=1)
df4['Target'].value_counts()
df4.describe().T


# In[11]:


x1=np.array(df4[5]).reshape(-1,1)
y1=y.reshape(-1,1)
lr=LinearRegression()
lr.fit(x1,y1)


# In[12]:


print(lr.coef_)
print(lr.intercept_)


# In[13]:


import matplotlib.pyplot as plt

plt.scatter(x1,y1,s=70,c='blue',edgecolor='white')
plt.plot(x1,lr.predict(x1),color='black',lw=2)


# In[14]:


from sklearn.linear_model import Ridge,Lasso,ElasticNet
ri=Ridge()
las=Lasso()
en=ElasticNet()


# In[15]:


ri.fit(x_train,y_train)
las.fit(x_train,y_train)
en.fit(x_train,y_train)


# In[16]:


ri_train_pred=ri.predict(x_train)
ri_test_pred=ri.predict(x_test)
las_train_pred=las.predict(x_train)
las_test_pred=las.predict(x_test)
en_train_pred=en.predict(x_train)
en_test_pred=en.predict(x_test)
print(ri_train_pred)
print(ri_test_pred)
print(las_train_pred)
print(las_test_pred)
print(en_train_pred)
print(en_test_pred)


# In[17]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[18]:


train_error_mean_ri=mean_squared_error(y_train,ri_train_pred)
train_error_mean_las=mean_squared_error(y_train,las_train_pred)
train_error_mean_en=mean_squared_error(y_train,en_train_pred)

test_error_mean_ri=mean_squared_error(y_test,ri_test_pred)
test_error_mean_las=mean_squared_error(y_test,las_test_pred)
test_error_mean_en=mean_squared_error(y_test,en_test_pred)

train_error_r2_ri=r2_score(y_train,ri_train_pred)
train_error_r2_las=r2_score(y_train,las_train_pred)
train_error_r2_en=r2_score(y_train,en_train_pred)

test_error_r2_ri=r2_score(y_test,ri_test_pred)
test_error_r2_las=r2_score(y_test,las_test_pred)
test_error_r2_en=r2_score(y_test,en_test_pred)

print(train_error_mean_ri)
print(train_error_mean_las)
print(train_error_mean_en)

print(test_error_mean_ri)
print(test_error_mean_las)
print(test_error_mean_en)

print(train_error_r2_ri)
print(train_error_r2_las)
print(train_error_r2_en)

print(test_error_r2_ri)
print(test_error_r2_las)
print(test_error_r2_en)


# In[19]:


#SYDNEY
import pandas as pd
df=pd.read_csv('Sydney_Data.csv')


# In[20]:


print(df.tail()) 
print(df.head()) 
print(df.describe()) 
print(df.info()) 


# In[21]:


df.isna().sum()


# In[22]:


df1=df.dropna()
print(df1)


# In[23]:


y=df1['AW'].values
print(y)
x=df1.drop(['AW'],axis=1).values
print(x)


# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[25]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
train_pred=lr.predict(x_train)
test_pred=lr.predict(x_test)


# In[26]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
train_error_mean=mean_squared_error(y_train,train_pred)
test_error_mean=mean_squared_error(y_test,test_pred)
train_error_r2=r2_score(y_train,train_pred)
test_error_r2=r2_score(y_test,test_pred)
print(train_error_mean)
print(test_error_mean)
print(train_error_r2)
print(test_error_r2)


# In[27]:


print(lr.coef_)
print(lr.intercept_)


# In[28]:


import numpy as np
import pandas as pd
df2=pd.DataFrame(x)
df3=pd.DataFrame(y,columns=['Target'])
df4=pd.concat([df2,df3],axis=1)
df4['Target'].value_counts()
df4.describe().T


# In[29]:


x1=np.array(df4[5]).reshape(-1,1)
y1=y.reshape(-1,1)
lr=LinearRegression()
lr.fit(x1,y1)


# In[30]:


print(lr.coef_)
print(lr.intercept_)


# In[31]:


import matplotlib.pyplot as plt

plt.scatter(x1,y1,s=70,c='blue',edgecolor='white')
plt.plot(x1,lr.predict(x1),color='black',lw=2)


# In[32]:


from sklearn.linear_model import Ridge,Lasso,ElasticNet
ri=Ridge()
las=Lasso()
en=ElasticNet()


# In[33]:


ri.fit(x_train,y_train)
las.fit(x_train,y_train)
en.fit(x_train,y_train)


# In[34]:


ri_train_pred=ri.predict(x_train)
ri_test_pred=ri.predict(x_test)
las_train_pred=las.predict(x_train)
las_test_pred=las.predict(x_test)
en_train_pred=en.predict(x_train)
en_test_pred=en.predict(x_test)
print(ri_train_pred)
print(ri_test_pred)
print(las_train_pred)
print(las_test_pred)
print(en_train_pred)
print(en_test_pred)


# In[35]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[36]:


train_error_mean_ri=mean_squared_error(y_train,ri_train_pred)
train_error_mean_las=mean_squared_error(y_train,las_train_pred)
train_error_mean_en=mean_squared_error(y_train,en_train_pred)

test_error_mean_ri=mean_squared_error(y_test,ri_test_pred)
test_error_mean_las=mean_squared_error(y_test,las_test_pred)
test_error_mean_en=mean_squared_error(y_test,en_test_pred)

train_error_r2_ri=r2_score(y_train,ri_train_pred)
train_error_r2_las=r2_score(y_train,las_train_pred)
train_error_r2_en=r2_score(y_train,en_train_pred)

test_error_r2_ri=r2_score(y_test,ri_test_pred)
test_error_r2_las=r2_score(y_test,las_test_pred)
test_error_r2_en=r2_score(y_test,en_test_pred)

print(train_error_mean_ri)
print(train_error_mean_las)
print(train_error_mean_en)

print(test_error_mean_ri)
print(test_error_mean_las)
print(test_error_mean_en)

print(train_error_r2_ri)
print(train_error_r2_las)
print(train_error_r2_en)

print(test_error_r2_ri)
print(test_error_r2_las)
print(test_error_r2_en)


# In[37]:


#TASMANIA
import pandas as pd
df=pd.read_csv('Tasmania_Data.csv')


# In[38]:


print(df.tail()) 
print(df.head()) 
print(df.describe()) 
print(df.info()) 


# In[39]:


df.isna().sum()


# In[40]:


df1=df.dropna()
print(df1)


# In[41]:


y=df1['AW'].values
print(y)
x=df1.drop(['AW'],axis=1).values
print(x)


# In[42]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[43]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
train_pred=lr.predict(x_train)
test_pred=lr.predict(x_test)


# In[44]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
train_error_mean=mean_squared_error(y_train,train_pred)
test_error_mean=mean_squared_error(y_test,test_pred)
train_error_r2=r2_score(y_train,train_pred)
test_error_r2=r2_score(y_test,test_pred)
print(train_error_mean)
print(test_error_mean)
print(train_error_r2)
print(test_error_r2)


# In[45]:


print(lr.coef_)
print(lr.intercept_)


# In[46]:


import numpy as np
import pandas as pd
df2=pd.DataFrame(x)
df3=pd.DataFrame(y,columns=['Target'])
df4=pd.concat([df2,df3],axis=1)
df4['Target'].value_counts()
df4.describe().T


# In[47]:


x1=np.array(df4[5]).reshape(-1,1)
y1=y.reshape(-1,1)
lr=LinearRegression()
lr.fit(x1,y1)


# In[48]:


print(lr.coef_)
print(lr.intercept_)


# In[49]:


import matplotlib.pyplot as plt

plt.scatter(x1,y1,s=70,c='blue',edgecolor='white')
plt.plot(x1,lr.predict(x1),color='black',lw=2)


# In[50]:


from sklearn.linear_model import Ridge,Lasso,ElasticNet
ri=Ridge()
las=Lasso()
en=ElasticNet()


# In[51]:


ri.fit(x_train,y_train)
las.fit(x_train,y_train)
en.fit(x_train,y_train)


# In[52]:


ri_train_pred=ri.predict(x_train)
ri_test_pred=ri.predict(x_test)
las_train_pred=las.predict(x_train)
las_test_pred=las.predict(x_test)
en_train_pred=en.predict(x_train)
en_test_pred=en.predict(x_test)
print(ri_train_pred)
print(ri_test_pred)
print(las_train_pred)
print(las_test_pred)
print(en_train_pred)
print(en_test_pred)


# In[53]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[54]:


train_error_mean_ri=mean_squared_error(y_train,ri_train_pred)
train_error_mean_las=mean_squared_error(y_train,las_train_pred)
train_error_mean_en=mean_squared_error(y_train,en_train_pred)

test_error_mean_ri=mean_squared_error(y_test,ri_test_pred)
test_error_mean_las=mean_squared_error(y_test,las_test_pred)
test_error_mean_en=mean_squared_error(y_test,en_test_pred)

train_error_r2_ri=r2_score(y_train,ri_train_pred)
train_error_r2_las=r2_score(y_train,las_train_pred)
train_error_r2_en=r2_score(y_train,en_train_pred)

test_error_r2_ri=r2_score(y_test,ri_test_pred)
test_error_r2_las=r2_score(y_test,las_test_pred)
test_error_r2_en=r2_score(y_test,en_test_pred)

print(train_error_mean_ri)
print(train_error_mean_las)
print(train_error_mean_en)

print(test_error_mean_ri)
print(test_error_mean_las)
print(test_error_mean_en)

print(train_error_r2_ri)
print(train_error_r2_las)
print(train_error_r2_en)

print(test_error_r2_ri)
print(test_error_r2_las)
print(test_error_r2_en)


# In[55]:


#Adelaide
import pandas as pd
df=pd.read_csv('Adelaide_Data.csv')


# In[56]:


print(df.tail()) 
print(df.head()) 
print(df.describe()) 
print(df.info()) 


# In[57]:


df.isna().sum()


# In[58]:


df1=df.dropna()
print(df1)


# In[59]:


y=df1['AW'].values
print(y)
x=df1.drop(['AW'],axis=1).values
print(x)


# In[60]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[61]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
train_pred=lr.predict(x_train)
test_pred=lr.predict(x_test)


# In[62]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
train_error_mean=mean_squared_error(y_train,train_pred)
test_error_mean=mean_squared_error(y_test,test_pred)
train_error_r2=r2_score(y_train,train_pred)
test_error_r2=r2_score(y_test,test_pred)
print(train_error_mean)
print(test_error_mean)
print(train_error_r2)
print(test_error_r2)


# In[63]:


print(lr.coef_)
print(lr.intercept_)


# In[64]:


import numpy as np
import pandas as pd
df2=pd.DataFrame(x)
df3=pd.DataFrame(y,columns=['Target'])
df4=pd.concat([df2,df3],axis=1)
df4['Target'].value_counts()
df4.describe().T


# In[65]:


x1=np.array(df4[5]).reshape(-1,1)
y1=y.reshape(-1,1)
lr=LinearRegression()
lr.fit(x1,y1)


# In[66]:


print(lr.coef_)
print(lr.intercept_)


# In[67]:


import matplotlib.pyplot as plt

plt.scatter(x1,y1,s=70,c='blue',edgecolor='white')
plt.plot(x1,lr.predict(x1),color='black',lw=2)


# In[68]:


from sklearn.linear_model import Ridge,Lasso,ElasticNet
ri=Ridge()
las=Lasso()
en=ElasticNet()


# In[69]:


ri.fit(x_train,y_train)
las.fit(x_train,y_train)
en.fit(x_train,y_train)


# In[70]:


ri_train_pred=ri.predict(x_train)
ri_test_pred=ri.predict(x_test)
las_train_pred=las.predict(x_train)
las_test_pred=las.predict(x_test)
en_train_pred=en.predict(x_train)
en_test_pred=en.predict(x_test)
print(ri_train_pred)
print(ri_test_pred)
print(las_train_pred)
print(las_test_pred)
print(en_train_pred)
print(en_test_pred)


# In[71]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[72]:


train_error_mean_ri=mean_squared_error(y_train,ri_train_pred)
train_error_mean_las=mean_squared_error(y_train,las_train_pred)
train_error_mean_en=mean_squared_error(y_train,en_train_pred)

test_error_mean_ri=mean_squared_error(y_test,ri_test_pred)
test_error_mean_las=mean_squared_error(y_test,las_test_pred)
test_error_mean_en=mean_squared_error(y_test,en_test_pred)

train_error_r2_ri=r2_score(y_train,ri_train_pred)
train_error_r2_las=r2_score(y_train,las_train_pred)
train_error_r2_en=r2_score(y_train,en_train_pred)

test_error_r2_ri=r2_score(y_test,ri_test_pred)
test_error_r2_las=r2_score(y_test,las_test_pred)
test_error_r2_en=r2_score(y_test,en_test_pred)

print(train_error_mean_ri)
print(train_error_mean_las)
print(train_error_mean_en)

print(test_error_mean_ri)
print(test_error_mean_las)
print(test_error_mean_en)

print(train_error_r2_ri)
print(train_error_r2_las)
print(train_error_r2_en)

print(test_error_r2_ri)
print(test_error_r2_las)
print(test_error_r2_en)

