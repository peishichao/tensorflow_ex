# - * - coding: utf-8 -*-

import tensorflow as tf

#define net struct about 

Input_node = 784
Output_node = 10

Image_size = 28
Num_channels = 1
Num_labels = 10

Conv1_deep = 32
Conv1_size = 5
Conv2_deep = 64
Conv2_size = 5
Fc_size = 512

def  get_weight_variable(shape,regularizer):
	weights = tf.get_variable(
		"weights",shape,
		initializer = tf.truncated_normal_initializer(stddev=0.1))	
	if regularizer != None:
		tf.add_to_collection('losses',regularizer(weights))
	return weights

def inference(input_tensor,train,regularizer):
	with tf.variable_scope('layer1-conv1'):
		conv1_weights = tf.get_variable("weights",
			[Conv1_size,Conv1_size,Num_channels,Conv1_deep],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable(
			"biases",[Conv1_deep],
			initializer = tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
	with tf.variable_scope('layer2-pool1'):
		pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	with tf.variable_scope('layer3-conv2'):
		conv2_weights = tf.get_variable("weight",[Conv2_size,Conv2_size,Conv1_deep,Conv2_deep],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable(
			"biases",[Conv2_deep],
			initializer = tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
	with tf.name_scope('layer4-pool2'):
		pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding ='SAME')
	pool_shape = pool2.get_shape().as_list()
	nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
	reshaped = tf.reshape(pool2,[pool_shape[0],nodes])
	with tf.variable_scope('layer5-fc1'):
		fc1_weights = tf.get_variable("weight",[nodes,Fc_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None:
			tf.add_to_collection('losses',regularizer(fc1_weights))
		fc1_biases = tf.get_variable("bias",[Fc_size],initializer = tf.constant_initializer(0.1))
		fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
		if train: fc1 = tf.nn.dropout(fc1,0.5)
	with tf.variable_scope('layer6-fc2'):
		fc2_weights = tf.get_variable("weight",[Fc_size,Num_labels],initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None:
			tf.add_to_collection('losses',regularizer(fc2_weights))
		fc2_biases = tf.get_variable("bias",[Num_labels],initializer=tf.constant_initializer(0.1))
		logit = tf.matmul(fc1,fc2_weights)+fc2_biases
	return logit
	
	
	
