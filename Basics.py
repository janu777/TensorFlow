import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #To eliminate initial warnings
import tensorflow as tf
import numpy as np
sess=tf.Session() #tf.Session is used for evaluation
#######################################################################
# 							Constants								  #
####################################################################### 
node1=tf.constant(0.3,dtype=tf.float32)
node2=tf.constant(0.2) #By default dtype of tf.constant is tf.float32 
node3=tf.add(node1,node2)
print(node3)
print('Printing Node value:%f' %(sess.run(node3)))
#######################################################################
# 						   Placeholders								  #
#######################################################################
a=tf.placeholder(dtype=tf.float32)
b=tf.placeholder(dtype=tf.float32)
adder_node=a+b
print(sess.run(adder_node,{a:3, b:2}))
print(sess.run(adder_node,{a:[1,2], b:[1,2]}))
add_and_square=tf.square(adder_node)
print(sess.run(add_and_square,{a:3, b:2}))
#######################################################################
#							Variables 								  #
#######################################################################
W=tf.Variable([0.3],dtype=tf.float32)
b=tf.Variable([-0.3])
x=tf.placeholder(dtype=tf.float32)
linear_model=W*x + b
init=tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model,{x:[1,2,3,4]}))
#######################################################################
fixW=tf.assign(W,[-1.0])
fixb=tf.assign(b,[1.0])
sess.run([fixW,fixb])
y=tf.placeholder(dtype=tf.float32)
loss=tf.reduce_sum(tf.square(linear_model-y))
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
#######################################################################
#					         Optimizer                                #
#######################################################################
Optimizer=tf.train.GradientDescentOptimizer(0.01)
train=Optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
	sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
print(sess.run([W,b]))	
#######################################################################