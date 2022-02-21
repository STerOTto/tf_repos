import tensorflow as tf
print(tf.__version__)

data = [
    {
        'feat_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 3])),
        'feat_vals': tf.train.Feature(float_list=tf.train.FloatList(value=[0.1, 0.3])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))
    },
    {
        'feat_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=[2, 4])),
        'feat_vals': tf.train.Feature(float_list=tf.train.FloatList(value=[0.2, 0.4])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
    },
    {
        'feat_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=[3])),
        'feat_vals': tf.train.Feature(float_list=tf.train.FloatList(value=[0.3])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))
    }
]
writer = tf.python_io.TFRecordWriter('test.tfrecords')

for feature in data:
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()
