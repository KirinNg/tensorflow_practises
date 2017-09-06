import tensorflow as tf
#准备待拟合数据
import numpy as np
X_train = np.linspace(-1,1,300)[:,np.newaxis] # 转为列向量  
noise = np.random.normal(0,0.05,X_train.shape)  
Y_train = np.square(X_train)+noise
print(X_train)
print(Y_train)
#构建神经网络
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
#第一层
W_first = tf.Variable(tf.random_normal([1,3]))
b_first = tf.Variable(tf.random_normal([1,3]))
W_plus_b_first = tf.matmul(xs,W_first)+b_first
A_first = tf.nn.sigmoid(W_plus_b_first)
#第二层
W_second = tf.Variable(tf.random_normal([3,1]))
b_second = tf.Variable(tf.random_normal([1,1]))
W_plus_b_second = tf.matmul(A_first,W_second)+b_second
A_second = tf.nn.sigmoid(W_plus_b_second)
#output
loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys-A_second)),reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(100000):
    sess.run(train_step,feed_dict={xs:X_train,ys:Y_train})
    if i%50 == 0:
        print(sess.run(loss,feed_dict={xs:X_train,ys:Y_train}))
