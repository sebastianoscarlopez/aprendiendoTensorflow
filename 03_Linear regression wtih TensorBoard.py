#By aplying Y = W * x + b. It get a line that fit to the points.
"""
    In commandline execute "tensorboard --logdir tmp"
    copy the url and paste in browser
    run the program
    refresh the browser
"""
import tensorflow as tf
import numpy as np

with tf.name_scope("data"):
    lengthPoints = 100
    dots = []
    for i in range(lengthPoints):
        x1 = np.random.normal(0.0, 0.55)
        y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        dots.append([x1, y1])
    x_data = [v[0] for v in dots]
    y_data = [v[1] for v in dots]

with tf.name_scope("model"):
    w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="w")
    b = tf.Variable(tf.zeros([1]), name="b")
    y = w * x_data + b

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y - y_data))

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    tf.summary.histogram('weight', w)
    tf.summary.histogram('bias', b)
    cost = tf.summary.scalar('loss', loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writerTB = tf.summary.FileWriter("tmp", sess.graph)
    sess.run(init)

    # Start train
    for step in range(8):
        sess.run(train)
        result = sess.run(merged)
        writerTB.add_summary(result, global_step=step)
