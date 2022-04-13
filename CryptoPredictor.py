# !pip install python-coinmarketcap
# !pip install yfinance

from coinmarketcapapi import CoinMarketCapAPI, CoinMarketCapAPIError
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import SimpleRNN, LSTM, GRU
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
import requests
import io

# Za eksportovanje podataka
'''
cmc = CoinMarketCapAPI('db7039e8-5bc7-4faf-96a7-05723b66ae3f')
r = cmc.cryptocurrency_map()
tables = pd.DataFrame(r.data)
tables = tables.sort_values(by='rank')
list_of_symbols = tables['symbol'].to_list()


end = datetime.datetime.now()
start = end - datetime.timedelta(days=3*365)

# create empty dataframe
stock_final = pd.DataFrame()
max_iters = 50
# iterate over each symbol
for i in list_of_symbols:  
  if max_iters == 0:
    break
  # print the symbol which is being downloaded
  print(str(list_of_symbols.index(i)) + str(' : ') + i, sep=',', end=',', flush=True)  

  try:
    # download the stock price 
    stock = []
    stock = yf.download(i+'-USD',start=start, end=end, progress=False)

    # append the individual stock prices 
    if len(stock) == 0:
      None
    else:
      stock['Name']=i+'-USD'
      stock_final = stock_final.append(stock,sort=False)
      max_iters -= 1
      print('*************************', i+'-USD', max_iters)
  except Exception:
    None

"""# Export data"""

ls = stock_final.index.to_list()
st = set()
for l in ls:
  st.add(l)

dates = list(st)
dates.sort()
cryptovals = stock_final.loc[dates[1090]]['Name'].to_list()

stock_final_cleaned = pd.DataFrame(index=dates, columns=cryptovals, dtype=np.double)
stock_final_cleaned = stock_final_cleaned.fillna(0.0)


for dat in dates:
  curr_cryptovals = stock_final.loc[dat]['Name']
  if len(curr_cryptovals) <= 8:
    continue
  curr_cryptovals = curr_cryptovals.to_list()
  curr_close_vals = stock_final.loc[dat]['Close'].to_list()
  for i, cc in enumerate(curr_cryptovals):
    stock_final_cleaned.loc[dat][cc] = curr_close_vals[i]
  print(dat)

stock_final_cleaned.head(7)
stock_final_cleaned.to_csv('cryptochart.csv')

'''

def predict(crypto):
  test_size = 1096 - train_size
  if crypto not in crypto_names:
    print('Cryptovalue is non-existent.')
    return

  embeding = embeddings[crypto]
  i = 0

  while i<len(Y_test):
      count = 0
      for col in X_test[i][0][1:]:
          if (math.isclose(col, embeding[count], rel_tol=1e-10)):
            count+=1
      if count==embedded_size:
        break
      i+=test_size

  predicted_stock_price = model.predict(X_test[i:i+test_size])
  predicted_stock_price = sc.inverse_transform(predicted_stock_price)

  real_stock_price = sc.inverse_transform(Y_test[i:i+test_size])
  
  #Plotting 
  plt.plot(real_stock_price, label="Actual Price", color='green')
  plt.plot(predicted_stock_price, label="Predicted Price", color='red')
  
  plt.title(crypto + ' price prediction')
  plt.xlabel('Time [days]')
  plt.ylabel('Price')
  plt.legend(loc='best')
  
  plt.show()

"""# ML"""

data = pd.read_csv('cryptochart.csv', index_col=0)

crypto_names = data.columns.to_list()
crypto_values = {}
crypto_train_values = {}
crypto_test_values = {}
train_size = int(1096*0.85)
len_series=7
embedded_size=10
batch_size = 32
num_epochs = 15
crypto_all_vals = []


for cn in crypto_names:
  crypto_values[cn] = data[cn].to_list()
  crypto_train_values[cn] = crypto_values[cn][:train_size]
  crypto_test_values[cn] = crypto_values[cn][train_size:]
  crypto_all_vals += crypto_values[cn]

sc = MinMaxScaler(feature_range=(0, 1))
crypto_all_vals = sc.fit_transform(np.array(crypto_all_vals).reshape(-1, 1))

for cn in crypto_names:
  crypto_train_values[cn] = sc.transform(np.array(crypto_train_values[cn]).reshape(-1, 1))
  crypto_test_values[cn] = sc.transform(np.array(crypto_test_values[cn]).reshape(-1, 1))

le = LabelEncoder()
le.fit(crypto_names)

model_emb = Sequential()
model_emb.add(Embedding(len(crypto_names), embedded_size, input_length=1))
model_emb.compile(optimizer='adam', loss='binary_crossentropy')

embeddings = {}

X_train,Y_train,X_test,Y_test=[],[],[],[]
for name in crypto_names:

  embedded = model_emb.predict(le.transform([name]))
  embedded = np.array(embedded[0][0])
  embedded.round(10)
  print(embedded)
  
  embeddings[name] = embedded

  for i in range(len(crypto_train_values[name])-len_series):
    series=crypto_train_values[name][i:i+len_series]
    joined_data = []
    for j in range(len_series):                 
      curr = [*series[j], *embedded]
      joined_data.append(curr)
    X_train.append(joined_data)
    Y_train.append(crypto_train_values[name][i+len_series])

  for i in range(len(crypto_test_values[name])-len_series):
    series=crypto_test_values[name][i:i+len_series]
    joined_data = []
    for j in range(len_series):                 
      curr = [*series[j], *embedded]
      joined_data.append(curr)
    X_test.append(joined_data)
    Y_test.append(crypto_test_values[name][i+len_series])

X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_train = np.array(Y_train)
X_train = np.array(X_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], embedded_size+1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], embedded_size+1))


# Model:
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], embedded_size+1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs)

crypto_name = input('Input crypto name or exit: ')
while crypto_name != 'exit':
  predict(crypto_name+'-USD')
  crypto_name = input('Input crypto name or exit: ')