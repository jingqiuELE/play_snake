import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Network Parameters
n_hidden = 50 # 1st layer number of neurons
n_classes = 4 # play snake total operations (KEY_UP, KEY_DOWN, KEY_RIGHT, KEY_LEFT)
learning_rate = 0.001
training_steps = 50000
timesteps = 2

# Import data
x = np.load("./data/screen.npy")
y = np.load("./data/operation.npy")
total_size = x.shape[0]
n_input = x.shape[1] * x.shape[2]

# Reshape X to be a 2D array, for X with each row to be features
x = x.flatten().reshape(total_size, -1)

# Setup train data and test data
train_size = int(total_size * 0.6)
x_train = x[0 : train_size - 1, :]
y_train = y[0 : train_size - 1, :]
x_test = x[train_size : total_size - 1, :]
y_test = y[train_size : total_size - 1, :]

# tf Graph input
X = tf.placeholder("float", [None, timesteps, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def prepare_batch(x):
    features = x.shape[1]
    x_next = np.delete(x, 0, axis = 0)
    x_prev = np.delete(x, -1, axis = 0)
    x_batch = np.concatenate((x_prev, x_next), axis = 1)
    x_batch = x_batch.reshape(-1, 2, features)
    return x_batch


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

y_train = np.delete(y_train, -1, axis=0)

loss_op = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(loss_op)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for epoch in range(training_steps):
    x_batch = prepare_batch(x_train)
    _, loss = sess.run([train_step, loss_op], feed_dict={X: x_batch, Y: y_train})
    if epoch % 100 == 0:
        print("Epoch:", '%04d' % epoch, "loss={:.9f} ".format(loss))


x_test_batch = prepare_batch(x_test)
y_test = np.delete(y_test, -1, axis=0)
print(sess.run(accuracy, feed_dict={X: x_test_batch, Y: y_test}))
