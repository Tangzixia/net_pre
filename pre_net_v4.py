#-*-coding=utf-8-*-
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from inception_v4 import inception_v4

def reshape(X):
    list=X.get_shape().as_list()
    dim=1
    for item in list[1:]:
        dim*=item
    X=tf.reshape(X,[-1,dim])
    return X,dim

height=299
width=299
channel=3
checkpoint_dir="H:/inception_v4.ckpt"
X=tf.placeholder(dtype=tf.float32,shape=[None,height,width,channel])
y=tf.placeholder(dtype=tf.float32,shape=[None,2])

if __name__=="__main__":
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_,_=inception_v4.inception_v4(X,num_classes=1001,reuse=None)
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
    train_step=tf.train.GradientDescentOptimizer(learning_rate=0.003).minimize(cross_entropy_mean)
    correct_pred=tf.equal(tf.argmax(final_tensor,1),tf.argmax(y,1))
    eval_op=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    #其中这个里面有一个exclude属性，用于除去不想加载的属性
    #inception_restore_variables=slim.get_variables_to_restore()
    init_fn=slim.assign_from_checkpoint_fn(checkpoint_dir,slim.get_model_variables("InceptionV4"),ignore_missing_vars=True)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        init_fn(sess)
        print("loaded")