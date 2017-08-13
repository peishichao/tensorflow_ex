#-*- coding = utf-8 -*-
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

Eval_interval_secs = 10

def evalute(mnist):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32,[None, mnist_inference.Input_node], name='x_intput')
		y_ = tf.placeholder(tf.float32,[None, mnist_inference.Output_node], name='y_output')
		validata_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
		y = mnist_inference.inference(x,None)
		correct_prediction =tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		variable_averages = tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)
		while True:
			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(mnist_train.Model_save_path)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess,ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					accuracy_score = sess.run(accuracy,feed_dict = validata_feed)
					print("After %s training steps, validation accuracy = %g" % (global_step,accuracy_score))
				else:
					print('No checkpoint file found')
					return		
		time.sleep(Eval_interval_secs)
def main(argv =None):
	mnist = input_data.read_data_sets("/tmp/data",one_hot = True)
	evalute(mnist)
if __name__ == '__main__':
	tf.app.run()
