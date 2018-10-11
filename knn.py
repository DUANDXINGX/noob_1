import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])
print(vectors_set)
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
b = tf.Variable(tf.zeros([1]), name='b')

y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss, name='train')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("W = ", sess.run(W), "b = ", sess.run(b), "loss = ", sess.run(loss))
    for _ in range(50):
        sess.run(train)
        print("W = ", sess.run(W), "b = ", sess.run(b), "loss = ", sess.run(loss))
    plt.scatter(x_data, y_data, c='r')
    plt.scatter(np.arange(min(x_data), max(x_data), 0.01), sess.run(W) * np.arange(min(x_data), max(x_data), 0.01) + sess.run(b))

plt.show()
