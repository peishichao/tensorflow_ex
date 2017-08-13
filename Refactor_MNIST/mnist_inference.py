# - * - coding: utf-8 -*-

import tensorflow as tf

#define net struct about 

Input_node = 784
Output_node = 10
Layer1_node = 500

def  get_weight_variable(shape,regularizer):
	weights = tf.get_variable(
		"weights",shape,
		initializer = tf.truncated_normal_initializer(stddev=0.1))	
	if regularizer != None:
		tf.add_to_collection('losses',regularizer(weights))
	return weights

def inference(input_tensor,regularizer):
	with tf.variable_scope('layer1'):
		weights = get_weight_variable(
			[Input_node,Layer1_node],regularizer)
		biases = tf.get_variable(
			"biases",[Layer1_node],
			initializer = tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
	with tf.variable_scope('layer2'):
		weights = get_weight_variable(
			[Layer1_node,Output_node],regularizer)
		biases = tf.get_variable(
			"biases",[Output_node],
			initializer = tf.constant_initializer(0.0))
		layer2 = tf.matmul(layer1,weights) + biases
	return layer2
	
	
	
