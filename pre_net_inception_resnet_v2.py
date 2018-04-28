#-*-coding=utf-8-*-
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from net import inception_resnet_v2
from read_data import data_reader
import random

height = 299
width=299
channel=3
checkpoint_dir="H:/new_CNN_split/inception_resnet_v2.ckpt"
img_train_path="H:/new_CNN_split/imgs_train"
label_train_path="H:/new_CNN_split/label_train.txt"
img_val_path="H:/new_CNN_split/imgs_val"
label_val_path="H:/new_CNN_split/label_val.txt"
batch_size = 32

if __name__=="__main__":
    X = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channel])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_,_=inception_resnet_v2.inception_resnet_v2(X,num_classes=1001,reuse=None)
    sp=logits_.get_shape().as_list()[-1]
    with tf.variable_scope("add_scope"):
        weights_fc_1=tf.get_variable(name="weights_fc_1",shape=(sp,100),initializer=tf.contrib.layers.xavier_initializer())
        biases_fc_1=tf.get_variable(name="biases_fc_1",shape=(100),initializer=tf.contrib.layers.xavier_initializer())
        fc_rec=tf.nn.bias_add(tf.matmul(logits_,weights_fc_1),biases_fc_1)
        weights_fc_2 = tf.get_variable(name="weights_fc_2", shape=(fc_rec.get_shape().as_list()[-1], 2), initializer=tf.contrib.layers.xavier_initializer())
        biases_fc_2 = tf.get_variable(name="biases_fc_2", shape=(2), initializer=tf.contrib.layers.xavier_initializer())
    logits = tf.nn.bias_add(tf.matmul(fc_rec, weights_fc_2), biases_fc_2)
    final_tensor=tf.nn.softmax(logits=logits)
    cross_entropy_mean=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))
    train_step = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9).minimize(cross_entropy_mean)
    correct_pred=tf.equal(tf.argmax(final_tensor,1),tf.argmax(y,1))
    eval_op=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    loader = data_reader(img_train_path, label_train_path, batch_size)
    loader_val = data_reader(img_val_path, label_val_path, batch_size)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    init_fn = slim.assign_from_checkpoint_fn(checkpoint_dir, slim.get_model_variables(),
                                             ignore_missing_vars=True)
    init_fn(sess)

    for epoch in range(10000):
        t_all_acc = 0
        t_all_loss = 0
        for iter in range(1000):
            batch_data = loader.read_data()
            _, eval_acc, t_loss = sess.run([train_step, eval_op, cross_entropy_mean],
                                           feed_dict={X: batch_data[0], y: batch_data[1]})
            t_all_acc += eval_acc
            t_all_loss += t_loss
            print("epoch:{0},iter:{1},acc:{2},loss:{3}".format(epoch, iter, t_all_acc / (iter + 1),
                                                               t_all_loss / (iter + 1)))

        print("第" + str(epoch) + "次验证集的准确率：")
        val_acc = 0
        for iter in range(10):
            val_data = loader_val.read_data()
            eval_acc = sess.run(eval_op, feed_dict={X: val_data[0], y: val_data[1]})
            val_acc += eval_acc
        print("第" + str(epoch) + "次epoch的验证集的准确率:" + str(val_acc / 10))
        saver.save(sess, "model/inception_v1_{0}".format(epoch))