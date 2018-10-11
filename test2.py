import tensorflow as tf
import input_data
mnist = input_data.read_data_sets('mnist/', one_hot=True)

training = mnist.train.images
train_label = mnist.train.labels
testing = mnist.test.images
test_label = mnist.test.labels
#数据初始化
n_input = 784
n_out = 10
n_hidden_1 = 256
n_hidden_2 = 128
#输入输出
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_out])
#模型参数
stddev = 0.1
weight = {
    'W1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'W2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_out], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_out])),
}

#定义前项传播
def multilayer_perceptron(_X, _weight, _biases):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weight['W1']), _biases['b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, _weight['W2']), _biases['b2']))
    return tf.add(tf.matmul(layer2, _weight['out']), _biases['out'])


pred = multilayer_perceptron(x, weight, biases)
#定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#定义优化函数
optm = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#计算准确度和误差值
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))

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
                  % (epoch, train_epochs, avg_cost, train_acc, test_acc))
print("DONE")
