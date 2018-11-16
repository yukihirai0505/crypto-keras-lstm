# ref: https://medium.com/@siavash_37715/how-to-predict-bitcoin-and-ethereum-price-with-rnn-lstm-in-keras-a6d8ee8a5109
import datetime
import gc
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Activation, Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential

# LSTMレイヤーの隠れ層の数
neurons = 512
# 活性化関数
activation_function = 'tanh'  # activation function for LSTM and Dense layer
loss = 'mse'  # loss function for calculating the gradient, in this case Mean Squared Error
optimizer = 'adam'  # optimizer for appljying gradient decent
dropout = 0.25  # dropout ratio used after each LSTM layer to avoid overfitting
batch_size = 128
epochs = 53
window_len = 7  # is an intiger to be used as the look back window for creating a single input sample.
training_size = 0.8  # porportion of data to be used for training
merge_date = '2016-01-01'  # the earliest date which we have data for both ETH and BTC or any other provided coin


def show():
    while True:
        try:
            plt.show()
        except UnicodeDecodeError:
            continue
        break


def get_market_data(market, tag=True):
    market_data = pd.read_html("https://coinmarketcap.com/currencies/" + market +
                               "/historical-data/?start=20130428&end=" + time.strftime("%Y%m%d"), flavor='html5lib')[0]
    market_data = market_data.assign(Date=pd.to_datetime(market_data['Date']))
    market_data['Volume'] = (pd.to_numeric(market_data['Volume'], errors='coerce').fillna(0))
    if tag:
        market_data.columns = [market_data.columns[0]] + [tag + '_' + i for i in market_data.columns[1:]]
    return market_data


def merge_data(a, b, from_date=merge_date):
    merged_data = pd.merge(a, b, on=['Date'])
    merged_data = merged_data[merged_data['Date'] >= from_date]
    return merged_data


def add_volatility(data, coins=['BTC', 'ETH']):
    for coin in coins:
        # calculate the daily change
        kwargs = {coin + '_change': lambda x: (x[coin + '_Close**'] - x[coin + '_Open*']) / x[coin + '_Open*'],
                  coin + '_close_off_high': lambda x: 2 * (x[coin + '_High'] - x[coin + '_Close**']) / (
                      x[coin + '_High'] - x[coin + '_Low']) - 1,
                  coin + '_volatility': lambda x: (x[coin + '_High'] - x[coin + '_Low']) / (x[coin + '_Open*'])}
        data = data.assign(**kwargs)
    return data


def create_model_data(data):
    data = data[['Date'] + [coin + metric for coin in ['BTC_', 'ETH_'] for metric in ['Close**', 'Volume']]]
    data = data.sort_values(by='Date')
    return data


def split_data(data, training_size=0.8):
    return data[:int(training_size * len(data))], data[int(training_size * len(data)):]


def create_inputs(data, coins=['BTC', 'ETH'], window_len=window_len):
    norm_cols = [coin + metric for coin in coins for metric in ['_Close**', '_Volume']]
    inputs = []
    for i in range(len(data) - window_len):
        temp_set = data[i:(i + window_len)].copy()
        inputs.append(temp_set)
        for col in norm_cols:
            inputs[i].loc[:, col] = inputs[i].loc[:, col] / inputs[i].loc[:, col].iloc[0] - 1
    return inputs


def create_outputs(data, coin, window_len=window_len):
    return (data[coin + '_Close**'][window_len:].values / data[coin + '_Close**'][:-window_len].values) - 1


def to_array(data):
    x = [np.array(data[i]) for i in range(len(data))]
    return np.array(x)


def build_model(inputs, output_size, neurons, activ_func=activation_function, dropout=dropout, loss=loss,
                optimizer=optimizer):
    model = Sequential()
    model.add(
        LSTM(neurons, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2]), activation=activ_func))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, return_sequences=True, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    model.summary()
    return model


def date_labels():
    last_date = market_data.iloc[0, 0]
    date_list = [last_date - datetime.timedelta(days=x) for x in range(len(X_test))]
    return [date.strftime('%m/%d/%Y') for date in date_list][::-1]


def plot_results(model, coin):
    ax1 = plt.subplot()
    plt.plot(test_set[coin + '_Close**'][window_len:].values.tolist())
    plt.plot(((np.transpose(model.predict(X_test)) + 1) * test_set[coin + '_Close**'].values[:-window_len])[0])
    plt.xlabel('Dates')
    plt.ylabel('Price')
    plt.title(coin + ' Single Point Price Prediction on Test Set')
    plt.legend(['Actual', 'Predicted'])

    date_list = date_labels()
    print(len(date_list))
    print(date_list)

    ax1.set_xticks([x for x in range(len(date_list))])
    ax1.set_xticklabels([date for date in date_list], rotation='vertical')
    show()


btc_data = get_market_data("bitcoin", tag='BTC')
eth_data = get_market_data("ethereum", tag='ETH')

market_data = merge_data(btc_data, eth_data)
model_data = create_model_data(market_data)
train_set, test_set = split_data(model_data)

train_set = train_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)

X_train = create_inputs(train_set)
Y_train_btc = create_outputs(train_set, coin='BTC')
X_test = create_inputs(test_set)
Y_test_btc = create_outputs(test_set, coin='BTC')

Y_train_eth = create_outputs(train_set, coin='ETH')
Y_test_eth = create_outputs(test_set, coin='ETH')

X_train, X_test = to_array(X_train), to_array(X_test)

print(np.shape(X_train), np.shape(X_test), np.shape(Y_train_btc), np.shape(Y_test_btc))
print(np.shape(X_train), np.shape(X_test), np.shape(Y_train_eth), np.shape(Y_test_eth))

# clean up the memory
gc.collect()

# random seed for reproducibility
np.random.seed(202)

# initialise model architecture
btc_model = build_model(X_train, output_size=1, neurons=neurons)

# train model on data
btc_history = btc_model.fit(X_train, Y_train_btc, epochs=epochs, batch_size=batch_size, verbose=1,
                            validation_data=(X_test, Y_test_btc), shuffle=False)

plot_results(btc_model, coin='BTC')

# For Ethereum predict

# # clean up the memory
# gc.collect()
#
# # random seed for reproducibility
# np.random.seed(202)
#
# # initialise model architecture
# eth_model = build_model(X_train, output_size=1, neurons=neurons)
#
# # train model on data
# eth_history = eth_model.fit(X_train, Y_train_eth, epochs=epochs, batch_size=batch_size, verbose=1,
#                             validation_data=(X_test, Y_test_eth), shuffle=False)
#
# plot_results(eth_history, eth_model, Y_train_eth, coin='ETH')
