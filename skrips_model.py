import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image  
from urllib.request import urlopen 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional

st.header('PREDIKSI HARGA CRYPTOCURRENCY MENGUNAKAN METODE LONG SHORT-TERM MEMORY')


st.write("""
Harga Cryptocurrency yang fluktuatif menjadika investasi Cryptocurrency tergolong menjadi investasi dengan resiko yang sangat tinggi oleh sebab itu perlunya informasi yang dapat membantu dalam mengetahui kemungkinan harga di masa depan,  agar dapat meminimalkan risiko dan mendapatkan keuntungan. Salah satu cara yang bisa digunakan adalah memprediksi harga cryptocurrency Mengunkana metode LSTM.\n
""")


imageBTC = Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1.png'))

st.image(imageBTC, use_column_width=False)
st.write("""
## History data Cryptocurency 
""")


data_btc = pd.read_csv('BTC-USD (3).csv')
st.dataframe(data_btc)
data_btc = data_btc[['Date','Open','Close']] # Extracting required columns
data_btc['Date'] = pd.to_datetime(data_btc['Date'].apply(lambda x: x.split()[0])) # Selecting only date
data_btc.set_index('Date',drop=True,inplace=True) # Setting date column as index

st.write("""
## Tampilan Untuk Melihat History data Cryptocurency yang akan di gunakan untuk prediksi
""")

st.dataframe(data_btc)

st.write("""
## Tampilan Dalam bentuk Grafik History data Cryptocurency yang akan di gunakan untuk prediksi
""")

st.line_chart(data_btc.Open,  use_container_width=True)
st.line_chart(data_btc.Close,  use_container_width=True)


MMS = MinMaxScaler()
data_btc[data_btc.columns] = MMS.fit_transform(data_btc)


training_size = round(len(data_btc) * 0.80) # Selecting 80 % for training and 20 % for testing

train_data = data_btc[:training_size]
test_data  = data_btc[training_size:]
# Function to create sequence of data for training and testing

def create_sequence(dataset):
  sequences = []
  labels = []

  start_idx = 1

  for stop_idx in range(50,len(dataset)): # Selecting 50 rows at a time
    sequences.append(dataset.iloc[start_idx:stop_idx])
    labels.append(dataset.iloc[stop_idx])
    start_idx += 1
  return (np.array(sequences),np.array(labels))


train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)


# models
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))

model.add(Dropout(0.1)) 
model.add(LSTM(units= 100))


model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history =model.fit(train_seq, train_label, batch_size=72, epochs=100,validation_data=(test_seq, test_label), verbose=1)

test_predicted = model.predict(test_seq)


test_inverse_predicted = MMS.inverse_transform(test_predicted) # Inversing scaling on predicted data

from sklearn.metrics import r2_score
print('RSquared :','{:.2%}'.format(r2_score(test_label,test_predicted)))

# Merging actual and predicted data for better visualization

gs_slic_data = pd.concat([data_btc.iloc[-284:].copy(),pd.DataFrame(test_inverse_predicted,columns=['Open_predicted','Close_predicted'],index=data_btc.iloc[-284:].index)], axis=1)
gs_slic_data[['Open','Close']] = MMS.inverse_transform(gs_slic_data[['Open','Close']]) # Inverse scaling



# Creating a dataframe and adding 10 days to existing index 
st.write("""
# Perbandingan Hasil Prediksi dengan data actual
""")
gs_slic_data = gs_slic_data.append(pd.DataFrame(columns=gs_slic_data.columns,index=pd.date_range(start=gs_slic_data.index[-1], periods=100 , freq='D', closed='right')))
gs_slic_data['2022-07-01	':'2022-07-30']

upcoming_prediction = pd.DataFrame(columns=['Open','Close'],index=gs_slic_data.index)
upcoming_prediction.index=pd.to_datetime(upcoming_prediction.index)


st.line_chart(gs_slic_data)



st.write("""
# Hasil Prediksi CRYPTOCURRENCY
""")
st.write("""
untuk melihat hasil prediksi scroll kebawa sampai menemukan angkanya.\n
""")
curr_seq = test_seq[-1:]

for i in range(-100,0):
  up_pred = model.predict(curr_seq)
  upcoming_prediction.iloc[i] = up_pred
  curr_seq = np.append(curr_seq[0][1:],up_pred,axis=0)
  curr_seq = curr_seq.reshape(test_seq[-1:].shape)

upcoming_prediction[['Open','Close']] = MMS.inverse_transform(upcoming_prediction[['Open','Close']])
upcoming_prediction

st.write("""
Dibuat untuk Memenuhi Persyaratan Akademis dalam Menyelesaikan Program Sarjana Strata Satu pada Program Studi Teknik Informatika Universitas Catur Insan Cendekia.\n
""")
