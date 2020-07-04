#!/usr/bin/env python
# coding: utf-8

# # Importing The Libraries

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
from fbprophet import Prophet
from datetime import datetime


import matplotlib.pyplot as plt
 
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')


# In[2]:


pd.plotting.register_matplotlib_converters()


# # Read The Data

# In[3]:


sales_df = pd.read_csv('data/filtered.csv')


# In[4]:


sales_df.head(5)


# # Drop The Unrelated Columns

# In[5]:


cleanDF = sales_df.drop(columns=['Unnamed: 0', 'PermitNo', 'Type', 'Status', 'SubType', 'Desc', 'StateValue','Contractor'])
cleanDF


# # Check The DataTypes

# In[6]:


cleanDF.info()


# # Change The DataTypes

# In[7]:


cleanDF['Date'] = pd.to_datetime(cleanDF['Date'], utc=True)
cleanDF['EstValue'] = cleanDF['EstValue'].astype(np.int64)


# In[8]:


cleanDF.info()


# In[9]:


cleanDF


# # Remove The TimeZone

# In[10]:


cleanDF['Date'] = cleanDF['Date'].astype(str).str[:-6]


# In[11]:


cleanDF.head()


# In[12]:


cleanDF.info()


# In[13]:


cleanDF['Date'] = pd.to_datetime(cleanDF['Date'])
cleanDF.info()


# # Keep The Date Only

# In[14]:


cleanDF['Date'] = pd.to_datetime(
    # cleanDF['Date'], format='%d-%b-%y %H.%M.%S.%f %p', errors='coerce'
    cleanDF['Date'], format='%d-%b-%y', errors='coerce'
    ).dt.floor('D')

cleanDF.head()


# # Sort The Date By Ascending

# In[15]:


cleanDF


# In[16]:


df = cleanDF.sort_values(by='Date')


# In[17]:


df


# # Check Empty Cell or Invalid

# In[18]:


df.isnull() 


# In[19]:


df.isnull().values.any()


# # Rename The Columns Name

# In[20]:


df = df.rename(columns={'Date':'ds', 'EstValue':'y'})


# In[21]:


df


# In[22]:


fig = df[['ds', 'y']].plot.scatter(y='ds', x='y')
fig


# In[23]:


df.set_index('ds').y.plot().figure


# In[24]:


df.set_index('y').ds.plot().figure


# In[25]:


promotions = pd.DataFrame({
  'holiday': 'december_promotion',
  'ds': pd.to_datetime(['2009-12-01', '2010-12-01', '2011-12-01', '2012-12-01',
                        '2013-12-01', '2014-12-01', '2015-12-01']),
  'lower_window': 0,
  'upper_window': 0,
})


# In[26]:


promotions


# In[27]:


df['y'] = np.log(df['y'])


# In[28]:


df.tail()


# In[29]:


model = Prophet(holidays=promotions, weekly_seasonality=True, daily_seasonality=True)
model.fit(df)


# In[30]:


future = model.make_future_dataframe(periods=24, freq = 'm')
future.tail()


# In[31]:


forecast = model.predict(future)


# In[32]:


forecast.tail()


# In[33]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[34]:


model.plot(forecast);


# In[35]:


model.plot_components(forecast);


# In[36]:


model_no_holiday = Prophet()
model_no_holiday.fit(df);


# In[37]:


future_no_holiday = model_no_holiday.make_future_dataframe(periods=24, freq = 'm')
future_no_holiday.tail()


# In[38]:


forecast_no_holiday = model_no_holiday.predict(future)


# In[39]:


forecast.set_index('ds', inplace=True)
forecast_no_holiday.set_index('ds', inplace=True)
compared_df = forecast.join(forecast_no_holiday, rsuffix="_no_holiday")


# In[40]:


compared_df= np.exp(compared_df[['yhat', 'yhat_no_holiday']])


# In[41]:


compared_df['diff_per'] = 100*(compared_df['yhat'] - compared_df['yhat_no_holiday']) / compared_df['yhat_no_holiday']
compared_df.tail()


# In[42]:


compared_df['diff_per'].mean()


# In[ ]:




