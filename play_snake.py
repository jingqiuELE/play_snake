import tensorflow as tf
import numpy as np

# Network Parameters
n_hidden_1 = 20 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons
n_out = 4 # play snake total operations (KEY_UP, KEY_DOWN, KEY_RIGHT, KEY_LEFT)

def main(_):
    # Import data
    X = np.load("./data/screen.npy")
    Y = np.load("./data/operation.npy")
    total_size = X.shape[0]
    n_input = X.shape[1] * X.shape[2]

    # Reshape X and Y to be a 2D array, for X with each row to be features
    X = X.flatten().reshape(total_size, -1)

    # Setup train data and test data
    train_size = int(total_size * 0.6)
    x_train = X[0 : train_size - 1, :]
    y_train = Y[0 : train_size - 1, :]
    x_test = X[train_size : total_size - 1, :]
    y_test = Y[train_size : total_size - 1, :]

    # Create the model
    x = tf.placeholder(tf.float32, [None, n_input])
    W1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
    b1 = tf.Variable(tf.random_normal([n_hidden_1]))
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

    W2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
    b2 = tf.Variable(tf.random_normal([n_hidden_2]))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, W2), b2))

    W3 = tf.Variable(tf.random_normal([n_hidden_2, n_out]))
    b3 = tf.Variable(tf.random_normal([n_out]))
    y = tf.matmul(layer_2, W3) + b3

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 4])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    optimizer = tf.train.GradientDescentOptimizer(0.02)
    train_step = optimizer.minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for epoch in range(15000):
        _, loss = sess.run([train_step, cross_entropy], feed_dict={x: x_train, y_: y_train})
        if epoch % 100 == 0:
            print("Epoch:", '%04d' % epoch, "loss={:.9f} ".format(loss))

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: x_test,
                                        y_: y_test}))

if __name__ == '__main__':
    tf.app.run(main=main)
