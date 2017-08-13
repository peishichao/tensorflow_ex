# -*- coding = utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os 
Batch_size = 100
Learning_rate_base = 0.8
Learning_rate_decay = 0.99
Regularaztion_rate = 0.0001
Training_steps = 30000
moving_average_decay = 0.99
Model_save_path = "/path/to/model"
Model_name = "model.ckpt"

def train(mnist):
	x = tf.placeholder(tf.float32,[None,mnist_inference.Input_node],name = 'x_input')
	y_ = tf.placeholder(tf.float32,[None,mnist_inference.Output_node],name = 'y_input')
	regularizer = tf.contrib.layers.l2_regularizer(Regularaztion_rate)
	y = mnist_inference.inference(x,regularizer)
	global_step = tf.Variable(0,trainable = False)
	variable_averages = tf.train.ExponentialMovingAverage(
		moving_average_decay,global_step)
	variable_averages_op = variable_averages.apply(
		tf.trainable_variables())
	corss_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=y,labels=tf.argmax(y_,1))
	cross_entropy_mean = tf.reduce_mean(corss_entropy)
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
	learning_rate = tf.train.exponential_decay(
		Learning_rate_base,
		global_step,
		mnist.train.num_examples/Batch_size,
		Learning_rate_decay)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
	with tf.control_dependencies([train_step,variable_averages_op]):
		train_op = tf.no_op(name = 'train')
	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		for i in range(Training_steps):
			xs,ys = mnist.train.next_batch(Batch_size)
			_,loss_value,step = sess.run([train_op,loss,global_step],
							feed_dict = {x:xs,y_:ys})
			if i % 1000 == 0:
				print ("After %d traning steps,loss on training batch is %g." % (step,loss_value))
				saver.save(sess, os.path.join(Model_save_path,Model_name),global_step = global_step)
def main(argv=None):
	mnist = input_data.read_data_sets("/tmp/data",one_hot = True)
	train(mnist)
if __name__ == '__main__':
	tf.app.run()
