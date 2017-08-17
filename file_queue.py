import tensorflow as tf
def _int64_feature(value):
	return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
num_shards = 2
instances_per_shard = 2
for i in rage(num_shards):
	filename = ('/path/to/data.tfrecords-%.5d-of-%.5d' % (i,num_shards))
	writer = tf.python_io.TFREcordWriter(filename)
	for j in range(instances_per_shard):
		example = tf.train.Example(features = tf.train.Features(feature={'i':_int64_feature(i),'j':_int64_feature(j)}))
		writer.write(example.SerializeToString())
	writer.close()

import tensorflow as tf

files = tf.train.match_filenames_once("/path/to/data.tfrecords-*")
filename_queue = tf.train.string_input_producer(files,shffle = False)
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,features={'i': tf.FixedLenFeature([],tf.int64),'j':tf.FixedlenFeature([].tf.int64)})
with tf.Session() as sess:
	tf.initialize_all_vaiables().run()
	print sess.run(files)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess,coord=coord)
	for i in range(6):
		print sess.run([feature['i'],feature['j']])

	coord.requese_stop()
	coord.join(threads)
