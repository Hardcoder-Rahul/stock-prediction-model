import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import time


# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Running please wait.......')
  bar.progress(i + 1)
  time.sleep(0.1)



# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'What are you interested in?',
    ('Companies with their tickers	', 'AMAZON	AMZN', 'Tesla	TSLA','Apple	AAPL',
    'Google	GOOG','Microsoft	MSFT')
)






# Or even better, call Streamlit functions inside a "with" block:

    
    
    
start = '2010-01-01'
end = '2021-11-01'

st.title("Stock Trend Prediction")

name  = st.text_input("Enter Company name")
user_input  = st.text_input("Enter Stock Ticker")
df = data.DataReader(user_input, 'yahoo', start, end)





#Describing Data
st.subheader("Retrieved Data")
st.write(f"Retrieved data of {name} is")
df.head()
st.subheader("Data from 2010 -2021")
st.write(df.describe())


#visualizations
st.subheader("Closing Price vs Time chart")
st.write(f"{name}")
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("Closing Price vs Time chart with 100MA")
st.write(f"{name}")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("Closing Price vs Time chart with 100MA and 200MA")
st.write(f"{name}")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


#Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)





#Load my model
model = load_model('keras_model.h5')

#Testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
    
    
x_test, y_test =  np.array(x_test), np.array(y_test)
    
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

#Final Graph


# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Predicting.........')
  bar.progress(i + 1)
  time.sleep(0.1)

'...here are the results!'




st.subheader("Prediction vs Orignal")
st.write(f"{name}")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)  










    
