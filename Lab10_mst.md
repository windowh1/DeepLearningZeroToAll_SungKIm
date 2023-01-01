---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Basic (AdamOptimizer)

```{code-cell} ipython3
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import random

tf.disable_v2_behavior()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# parameters 1
nb_classes = 10
lr = 0.01

x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1,784])

# tf.one_hot 사용했더니 오류나서 못 씀
# y_train, y_test = tf.one_hot(y_train, depth = 10), tf.one_hot(y_test, depth = 10)

y_train_new = np.zeros((len(y_train), nb_classes))       #60000 * 10 배열 생성
for i in range(len(y_train_new)):                        
  y_train_new[i, y_train[i]] = 1                          #one-hot encoding 

y_test_new = np.zeros((len(y_test), nb_classes))       #60000 * 10 배열 생성
for i in range(len(y_test_new)):                        
  y_test_new[i, y_test[i]] = 1     


X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# parameters 2
training_epochs = 15  #traing을 몇번 돌릴것인지
batch_size = 100  #한번에 몇건씩 읽은것인지
total_batch = int(len(x_train) / batch_size)

for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_xs = x_train[(i * batch_size) : (i+1) * batch_size]
        batch_ys = y_train_new[(i * batch_size) : (i+1) * batch_size]
        _, c = sess.run([optimizer, cost], feed_dict = {X : batch_xs, Y : batch_ys})
        avg_cost = avg_cost + c/total_batch
        
    print('Epoch:', (epoch + 1), 'Cost=', avg_cost)

print('Learning Finished')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy:', sess.run(accuracy, feed_dict = {X : x_test, Y : y_test_new}))
```

# NN for MNIST

```{code-cell} ipython3
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, nb_classes]))
b3 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_xs = x_train[(i * batch_size) : (i+1) * batch_size]
        batch_ys = y_train_new[(i * batch_size) : (i+1) * batch_size]
        _, c = sess.run([optimizer, cost], feed_dict = {X : batch_xs, Y : batch_ys})
        avg_cost = avg_cost + c/total_batch
        
    print('Epoch:', (epoch + 1), 'Cost=', avg_cost)

print('Learning Finished')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy:', sess.run(accuracy, feed_dict = {X : x_test, Y : y_test_new}))

```

# Xavier for MNIST

```{code-cell} ipython3
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# get_variable 사용 시 위에서 동일한 변수 이름 정의했을 경우 Error 발생 -> scope 사용 필요
# 참조 : https://calcifer1009-dev.tistory.com/10
with tf.variable_scope("fi") as scope: # 이거 재실행할 때는 scope 이름 바꿔주면 working
    # tf v2부터는 contrib 사라졌기 때문에 비슷한 효과 내는 것으로 대체함
    # 참조 : https://limitsinx.tistory.com/49
    W1 = tf.get_variable("W1", shape = [784, 256], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b1 = tf.Variable(tf.random_normal([256]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    W2 = tf.get_variable("W2", shape = [256, 256], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b2 = tf.Variable(tf.random_normal([256]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    W3 = tf.get_variable("W3", shape = [256, nb_classes], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b3 = tf.Variable(tf.random_normal([nb_classes]))
    hypothesis = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_xs = x_train[(i * batch_size) : (i+1) * batch_size]
        batch_ys = y_train_new[(i * batch_size) : (i+1) * batch_size]
        _, c = sess.run([optimizer, cost], feed_dict = {X : batch_xs, Y : batch_ys})
        avg_cost = avg_cost + c/total_batch
        
    print('Epoch:', (epoch + 1), 'Cost=', avg_cost)

print('Learning Finished')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy:', sess.run(accuracy, feed_dict = {X : x_test, Y : y_test_new}))
```

# Deep NN for MNIST

```{code-cell} ipython3
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
lr = 0.01

with tf.variable_scope("fa") as scope:
    W1 = tf.get_variable("W1", shape = [784, 512], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b1 = tf.Variable(tf.random_normal([512]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    W2 = tf.get_variable("W2", shape = [512, 512], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b2 = tf.Variable(tf.random_normal([512]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    W3 = tf.get_variable("W3", shape = [512, 512], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b3 = tf.Variable(tf.random_normal([512]))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

    W4 = tf.get_variable("W4", shape = [512, 512], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b4 = tf.Variable(tf.random_normal([512]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

    W5 = tf.get_variable("W5", shape = [512, nb_classes], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b5 = tf.Variable(tf.random_normal([nb_classes]))
    hypothesis = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_xs = x_train[(i * batch_size) : (i+1) * batch_size]
        batch_ys = y_train_new[(i * batch_size) : (i+1) * batch_size]
        _, c = sess.run([optimizer, cost], feed_dict = {X : batch_xs, Y : batch_ys})
        avg_cost = avg_cost + c/total_batch
        
    print('Epoch:', (epoch + 1), 'Cost=', avg_cost)

print('Learning Finished')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy:', sess.run(accuracy, feed_dict = {X : x_test, Y : y_test_new}))
```

# Dropout for MNIST

```{code-cell} ipython3
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
kp = tf.placeholder(tf.float32)

nb_classes = 10
lr = 0.01
training_epochs = 15  #traing을 몇번 돌릴것인지
batch_size = 100  #한번에 몇건씩 읽은것인지
total_batch = int(len(x_train) / batch_size)

with tf.variable_scope("fooo") as scope:
    W1 = tf.get_variable("W1", shape = [784, 512], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b1 = tf.Variable(tf.random_normal([512]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    L1 = tf.nn.dropout(L1, keep_prob = kp)

    W2 = tf.get_variable("W2", shape = [512, 512], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b2 = tf.Variable(tf.random_normal([512]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.dropout(L2, keep_prob = kp)

    W3 = tf.get_variable("W3", shape = [512, 512], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b3 = tf.Variable(tf.random_normal([512]))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob = kp)

    W4 = tf.get_variable("W4", shape = [512, 512], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b4 = tf.Variable(tf.random_normal([512]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob = kp)

    W5 = tf.get_variable("W5", shape = [512, nb_classes], initializer = tf.truncated_normal_initializer(stddev=0.1))
    b5 = tf.Variable(tf.random_normal([nb_classes]))
    hypothesis = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_xs = x_train[(i * batch_size) : (i+1) * batch_size]
        batch_ys = y_train_new[(i * batch_size) : (i+1) * batch_size]
        _, c = sess.run([optimizer, cost], feed_dict = {X : batch_xs, Y : batch_ys, kp : 0.9})
        avg_cost = avg_cost + c/total_batch
        
    print('Epoch:', (epoch + 1), 'Cost=', avg_cost)

print('Learning Finished')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy:', sess.run(accuracy, feed_dict = {X : x_test, Y : y_test_new, kp : 1.0}))
```
