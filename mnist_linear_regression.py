import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('mnist/', one_hot=True)

training = mnist.train.images
train_label = mnist.train.labels
testing = mnist.test.images
test_label = mnist.test.labels

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

actv = tf.nn.softmax(tf.matmul(x, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))

optm = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(pred, "float"))

train_epochs = 50
batch_size = 100
display_step = 5
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(train_epochs):
        avg_cost = 0.
        num_batch = int(mnist.train.num_examples / batch_size)
        for i in range(num_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            feeds = {x: batch_x, y: batch_y}
            sess.run(optm, feed_dict=feeds)
            avg_cost += sess.run(cost, feed_dict=feeds)
        avg_cost /= num_batch

        if epoch % display_step == 0:
            feeds_trains = {x: batch_x, y: batch_y}
            feeds_test = {x: testing, y: test_label}
            train_acc = sess.run(accr, feed_dict=feeds_trains)
            test_acc = sess.run(accr, feed_dict=feeds_test)
            print("Eopch: %3d/%3d cost: %.9f train_acc %.3f test_acc: %.3f"
                  %(epoch, train_epochs, avg_cost, train_acc, test_acc))
print("DONE")



