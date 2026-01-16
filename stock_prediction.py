#!/usr/bin/env python
# coding: utf-8

# ## **`📈 Stock Price Prediction using LSTM`**
# * 🌐 Susanta Sekhar Palai
# * 📧 susantasekhar1@gmail.com

# >  ╰┈➤ 📊 Dataset

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('Reliance.csv')  


# In[3]:


stock = "Reliance" 


# In[4]:


print(df.head())


# In[5]:


print(df.tail())


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df['Date'] = pd.to_datetime(df['Date'] , dayfirst=True)


# In[10]:


df.info()


# In[11]:


df = df.sort_values('Date')
df = df.reset_index(drop=True)


# In[12]:


df[df.isnull().any(axis=1)]


# In[13]:


df.fillna(method='ffill', inplace=True)


# In[14]:


df.fillna(method='bfill', inplace=True)


# In[15]:


df.isnull().sum()


# In[16]:


df.dropna(inplace=True)


# In[17]:


df.shape


# In[18]:


df.columns


# In[19]:


import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])
    ]
)
fig.update_layout(
    xaxis_rangeslider_visible=False
)
fig.show()


# In[20]:


df.info()


# In[21]:


df = df.drop(['Date', 'Adj Close'], axis = 1)


# In[22]:


print(df.head())


# #### 📉 Closing Prices Over Time

# In[23]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label = f'{stock} Closing Price', linewidth = 2)
plt.title(f'{stock} Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price ( Rupees )' )
plt.legend()
plt.show()


# #### 📈 Opening Prices Over Time

# In[24]:


plt.figure(figsize=(12, 6))
plt.plot(df['Open'], label = f'{stock} Opening Price', linewidth = 2)
plt.title(f'{stock} Opening Prices Over Time')
plt.legend()
plt.show()


# #### 📈 High Prices Over Time

# In[25]:


plt.figure(figsize=(12, 6))
plt.plot(df['High'], label = f'{stock} Volume', linewidth = 1)
plt.title(f'{stock} High Prices Over Time')
plt.legend()
plt.show()


# #### 📈 Low Prices Over Time

# In[26]:


plt.figure(figsize=(12, 6))
plt.plot(df['Low'], label = f'{stock} Low', linewidth = 1)
plt.title(f'{stock} Low Prices Over Time')
plt.legend()
plt.show()


# #### 💹 Checking Moving Average

# In[27]:


df01 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 10, 130, 140, 150, 160, 170, 180, 190, 200]

print('Moving Average : ', sum(df01[0 : 5]) / 5)
print('Moving Average : ', sum(df01[1 : 6]) / 5)
print('Moving Average : ', sum(df01[2 : 7]) / 5)
print('Moving Average : ', sum(df01[3 : 9]) / 5)
print('Moving Average : ', sum(df01[4 : 10]) / 5)
print('Moving Average : ', sum(df01[5 : 11]) / 5)


# In[28]:


df1 = pd.DataFrame(df01)


# In[29]:


df1.rolling(5).mean()


#  #### 💹 100-day moving average (MA)

# In[30]:


ma100 = df.Close.rolling(100).mean()


# In[31]:


ma100


#  #### 💹 200-day moving average (MA)

# In[32]:


ma200 = df.Close.rolling(200).mean()


# In[33]:


ma200


# In[34]:


plt.figure(figsize=(12, 6))
plt.plot(df.Close, label = f'{stock} Closing Price', linewidth = 1)
plt.plot(ma100, label = f'{stock} Moving Average 100', linewidth = 1)
plt.plot(ma200, label = f'{stock} Moving Average 200', linewidth = 1)
plt.title(f'{stock} Moving Average ')
plt.legend()
plt.show()


# #### 📊 100-period Exponential Moving Average

# In[35]:


ema100 = df.Close.ewm(span=100, adjust=False).mean()


# #### 📊 200-period Exponential Moving Average

# In[36]:


ema200 = df['Close'].ewm(span=200, adjust=False).mean()


# #### Percentage change Of Data ██████ 100%

# In[37]:


df.pct_change()


# #### ███░░ EMA 100 VS EMA 200

# In[38]:


plt.figure(figsize=(12, 6))
plt.plot(df.Close, label = f'{stock} Closing Price', linewidth = 1)
plt.plot(ema100, label = f'{stock} EMA 100', linewidth = 1)
plt.plot(ema200, label = f'{stock} EMA 200', linewidth = 1)
plt.title(f'{stock} EMA ')
plt.legend()
plt.show()


# In[39]:


df.info()


# #### 🔎 Training & Testing

# In[40]:


data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70) :int(len(df))])


# In[41]:


data_training.shape


# In[42]:


df.shape


# In[43]:


data_testing.shape


# In[44]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))
data_training_array = scaler.fit_transform(data_training)


# In[45]:


data_training_array[0 : 5]


# In[46]:


data_training_array.shape[0]


# In[47]:


data_training_array.shape


# In[48]:


import numpy as np


# In[49]:


x_train = []
y_train = []

for i in range (100 , data_training_array.shape[0]) :
    x_train.append(data_training_array[i - 100 : i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)


# In[50]:


x_train.shape


# ## `Model Building`

# In[51]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

model = Sequential()

# 1st LSTM layer
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

# 2nd LSTM layer
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

# 3rd LSTM layer
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

# 4th (LAST) LSTM layer
model.add(LSTM(units=100, activation='relu', return_sequences=False))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1))


# In[52]:


print(x_train.shape)


# In[53]:


model.summary()


# In[54]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)


# In[55]:


past_100_days = data_training.tail(100)


# In[56]:


past_100_days.columns


# In[57]:


past_100_days = data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)


# In[58]:


final_df.head()


# In[85]:


input_data = scaler.transform(final_df)


# In[86]:


x_test = []
y_test = []

for i in range (100 , input_data.shape[0]) :
    x_test.append(input_data[i - 100 : i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)


# In[87]:


x_test.shape


# In[88]:


y_predicted = model.predict(x_test)


# In[89]:


y_predicted.shape


# In[90]:


scaler.scale_


# In[91]:


scaler_factor = 1 / 1000
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor


# ### 📊 Original Price vs Predicted Price

# In[92]:


plt.figure(figsize=(12, 6))
plt.plot(y_test, label = 'Original Price', linewidth = 2)
plt.plot(y_predicted, label = 'Predicted Price', linewidth = 2)

plt.legend()
plt.show()


# In[93]:


model.save('stock_dl_model.h5')

