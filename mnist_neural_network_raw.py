from __future__ import print_function

import tensorflow as tf
import cv2
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#parameters
learning_rate=0.1
epochs=500
batch_size=128
display_step=100

#network parameters

n_hidden_1=256 #first layer neurons
n_hidden_2=256 #second layer neurons
num_input=784 #mnist data input
num_classes=10

#graph input

X=tf.placeholder("float",[None, num_input])
Y=tf.placeholder("float",[None,num_classes])

#define layers weights and biases

weights={
    'h1':tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

biases={
    'b1':tf.Variable(tf.Variable(tf.random_normal([n_hidden_1]))),
    'b2':tf.Variable(tf.Variable(tf.random_normal([n_hidden_2]))),
    'out':tf.Variable(tf.Variable(tf.random_normal([num_classes])))
}

#create model

def neural_net(X):
    
    #hidden layer 256 fully connected
    layer_1=tf.add(tf.matmul(X, weights['h1']), biases['b1'])

    #hidden layer 256 fully connected
    layer_2=tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    #output fully connected layer with 10 output
    out_layer=tf.add(tf.matmul(layer_2, weights['out']),biases['out'])

    return out_layer

#build model
logits=neural_net(X)
prediction= tf.nn.softmax(logits)

#define loss function and optimizer
loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(loss_op)

#to evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

#training time

with tf.Session() as sess:

    #run the initializer
    sess.run(init)
    
    for epoch in range(1, epochs+1):
        #load batch
        batch_x, batch_y=mnist.train.next_batch(batch_size)
        #run optimization
        sess.run(train_op, feed_dict={X:batch_x, Y:batch_y})

        if epoch % display_step == 0 or epoch == 1:

                #calculate batch loss and accuracy

            loss, acc = sess.run([loss_op, accuracy], feed_dict={X:batch_x, Y:batch_y})

            print ("epoch--{}   Loss--{:.4f}  Accuracy--{:.4f}".format(epoch, loss, acc)) 

    print("Optimization Finished")

    print("Testing Accuracy:",sess.run(accuracy, feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
    
    labels=["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]
                                                            
    img=cv2.imread("/home/user/Downloads/number0.png")
    grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(grey_img,(28,28))
    img=img.reshape(1,784)
    output=sess.run(prediction,feed_dict={X:img, Y:[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]})
    op=output[0].tolist()
    #print (op)
    print (labels[op.index(max(op))])
