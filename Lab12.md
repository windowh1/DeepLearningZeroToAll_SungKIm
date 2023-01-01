---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Lab 12-3

```python
import tensorflow as tf
import numpy as np

sample = "if you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea."
idx2char = list(set(sample))
char2idx = {c : i for i, c in enumerate(idx2char)}

dic_size = len(char2idx) # RNN input size (one hot size)
rnn_hidden_size = len(char2idx) # RNN output size
num_classes = len(char2idx) # final output size (RNN or softmax, etc.)
batch_size = 1
sequence_length = len(sample) - 1
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[: -1]]
y_data = [sample_idx[1 :]]

x_one_hot = tf.one_hot(x_data, dic_size) # (1 * 12 * 10) = (batch_size * sequence_length * num_classes)
y_one_hot = tf.one_hot(y_data, num_classes) # (1 * 12 * 10) = (batch_size * sequence_length * num_classes)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.LSTM(units = num_classes
                                , input_shape = (sequence_length, dic_size)
                                , return_sequences = True))
tf.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units = num_classes, activation = 'softmax')))
# TimeDistributed : 각 sequence 마다 Dense layer 적용
tf.model.summary()
tf.model.compile(loss = 'categorical_crossentropy'
                , optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
                , metrics = ['accuracy'])
tf.model.fit(x_one_hot, y_one_hot, epochs = 50) 

prediction = tf.model.predict(x_one_hot)

for i, prediction in enumerate(prediction):
    result_str = [idx2char[c] for c in np.argmax(prediction, axis = 1)]
    print("\tPrediction str:", ''.join(result_str))
```

# Lab 12-4

```python
import tensorflow as tf
import numpy as np

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dict = {w : i for i, w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10 # Any arbitrary number
learning_rate = 0.01

dataX = []
dataY = []
for i in range(len(sentence) - sequence_length):
    x_str = sentence[i : i + sequence_length]
    y_str = sentence[i + 1 : i + 1 + sequence_length]
    print(i, x_str, '->', y_str)
    
    x = [char_dict[c] for c in x_str]
    y = [char_dict[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)
```

```python
batch_size = len(dataX)

x_one_hot = tf.one_hot(dataX, data_dim) # (170 * 10 * 25) = (batch_size * sequence_length * num_classes)
y_one_hot = tf.one_hot(dataY, data_dim) # (170 * 10 * 25) = (batch_size * sequence_length * num_classes)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.LSTM(units = num_classes
                                , input_shape = (sequence_length, x_one_hot.shape[2])
                                , return_sequences = True))
tf.model.add(tf.keras.layers.LSTM(units = num_classes
                                , return_sequences = True)) # Stacked RNN (2nd layer)
tf.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units = num_classes
                                                                , activation = 'softmax')))
tf.model.summary()
tf.model.compile(loss = 'categorical_crossentropy'
                , optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
                , metrics = ['accuracy'])
tf.model.fit(x_one_hot, y_one_hot, epochs = 200)

results = tf.model.predict(x_one_hot)
for j, result in enumerate(results):
    index = np.argmax(result, axis = 1)
    if j == 0:
        print(''.join([char_set[t] for t in index]), end = '') #if you wan
    else :
        print(char_set[index[-1]], end = '') # t to build ~ # 마지막 글자만 이어붙임
```

# Lab 12-6

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

seq_length = 7
data_dim = 5
output_dim = 1
learning_rate = 0.01
iterations = 500

xy = np.loadtxt('data-02-stock_daily.csv', delimiter = ',')
xy = xy[::-1] # reverse order (chronically ordered)

train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size:]

train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(len(time_series) - seq_length):
        x = time_series[i : i + seq_length, :]
        y = time_series[i + seq_length, -1]
        print(x, '->', y)
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

print(trainX.shape)
print(trainY.shape)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.LSTM(units = 1, input_shape = (seq_length, data_dim)))
# units : rnn cell (일종의 hidden layer) 의 unit 개수
# units = 1 말고 다른 수로 설정해도 결과는 비슷하게 나옴
tf.model.add(tf.keras.layers.Dense(units = output_dim, activation = 'tanh'))
tf.model.summary()

tf.model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate))
tf.model.fit(trainX, trainY, epochs = iterations)

test_predict = tf.model.predict(testX)

plt.plot(testY)
plt.plot(test_predict)
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()
```
