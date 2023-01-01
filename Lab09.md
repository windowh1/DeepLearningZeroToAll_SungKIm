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

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

tf.reset_default_graph()

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype = np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)

X = tf.placeholder(tf.float32, name = "x")
Y = tf.placeholder(tf.float32, name = "y")

with tf.name_scope("Layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2, 2]), name = "weight_1")
    b1 = tf.Variable(tf.random_normal([2]), name = "bias_1")
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

w1_hist = tf.summary.histogram("weights1", W1)
b1_hist = tf.summary.histogram("biases1", b1)
layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("Layer2") as scope:
    W2 = tf.Variable(tf.random_normal([2, 1]), name = "weight_2")
    b2 = tf.Variable(tf.random_normal([1]), name = "bias_2")
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

w2_hist = tf.summary.histogram("weights2", W2)
b2_hist = tf.summary.histogram("biases2", b2)
hypotheis_hist = tf.summary.histogram("hypothesis", hypothesis)

with tf.name_scope("Cost"):
    cost = - tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    tf.summary.scalar("cost", cost)

with tf.name_scope("Train"):
    tr = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))
tf.summary.scalar("accuracy", accuracy)

sess = tf.Session()
summary = tf.summary.merge_all()

writer = tf.summary.FileWriter("/Users/hyewon/Desktop/DeepLearning_SungKim/logs/xor_logs")
writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())

for step in range(10001):
    _, s, c = sess.run([tr, summary, cost], feed_dict = {X : x_data, Y : y_data})
    writer.add_summary(s, global_step = step)
    
    if step % 1000 == 0 :
        print(step, c)


h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {X : x_data, Y : y_data})
print(f"\nHypothesis : {h} \nPredicted : {p} \nAccuracy : {a}")
        
```

Multiple runs (with learning_rate = 0.01)

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

tf.reset_default_graph()

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype = np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)

X = tf.placeholder(tf.float32, name = "x")
Y = tf.placeholder(tf.float32, name = "y")

with tf.name_scope("Layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2, 2]), name = "weight_1")
    b1 = tf.Variable(tf.random_normal([2]), name = "bias_1")
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

w1_hist = tf.summary.histogram("weights1", W1)
b1_hist = tf.summary.histogram("biases1", b1)
layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("Layer2") as scope:
    W2 = tf.Variable(tf.random_normal([2, 1]), name = "weight_2")
    b2 = tf.Variable(tf.random_normal([1]), name = "bias_2")
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

w2_hist = tf.summary.histogram("weights2", W2)
b2_hist = tf.summary.histogram("biases2", b2)
hypotheis_hist = tf.summary.histogram("hypothesis", hypothesis)

with tf.name_scope("Cost"):
    cost = - tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    tf.summary.scalar("cost", cost)

with tf.name_scope("Train"):
    tr = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))
tf.summary.scalar("accuracy", accuracy)

sess = tf.Session()
summary = tf.summary.merge_all()

writer = tf.summary.FileWriter("/Users/hyewon/Desktop/DeepLearning_SungKim/logs/xor_logs_r0_01")
writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())

for step in range(10001):
    _, s, c = sess.run([tr, summary, cost], feed_dict = {X : x_data, Y : y_data})
    writer.add_summary(s, global_step = step)
    
    if step % 1000 == 0 :
        print(step, c)


h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {X : x_data, Y : y_data})
print(f"\nHypothesis : {h} \nPredicted : {p} \nAccuracy : {a}")
```
