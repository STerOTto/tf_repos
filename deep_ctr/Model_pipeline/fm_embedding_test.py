import tensorflow as tf
import sys

SAMPLE_SIZE = 5


def read_tfrecords(file):
    """
    读取tfrecords的数据
    :return: None
    """
    # 1、构造文件队列
    file_queue = tf.train.string_input_producer([file])

    # 2、构造tfrecords读取器，读取队列
    reader = tf.TFRecordReader()

    # 默认也是只读取一个样本
    key, values = reader.read(file_queue)

    # tfrecords
    # 多了解析example的一个步骤
    feature_map = {
        "feat_ids": tf.io.VarLenFeature(tf.int64),
        "feat_vals": tf.io.VarLenFeature(tf.float32),
        'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    }
    data = tf.parse_single_example(values, features=feature_map)

    # feature_map = data['sparse_map']
    feature_map_batches = tf.train.batch(data, 3, capacity=10)
    return feature_map_batches


if __name__ == '__main__':
    exmpl = read_tfrecords('test.tfrecords')
    print('hahah')
    FM_V = tf.get_variable(name='fm_v', shape=[8, 3], initializer=tf.glorot_normal_initializer())
    with tf.Session() as sess:

        # 创建线程回收的协调员
        coord = tf.train.Coordinator()

        # 需要手动开启子线程去进行批处理读取到队列操作
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        data = sess.run(exmpl)
        print('='*10, '\n', data)
        print('FM_V:\n', sess.run(FM_V))
        feat_ids = exmpl['feat_ids']
        feat_vals = exmpl['feat_vals']
        dense_ids = tf.sparse_tensor_to_dense(feat_ids)
        print('dense_ids shape:', sess.run(dense_ids).shape)
        dense_vals = tf.expand_dims(tf.sparse_tensor_to_dense(feat_vals), axis=-1)
        print('dense_vals shape:', sess.run(dense_vals).shape)
        embeddings = tf.nn.embedding_lookup(FM_V, dense_ids)
        print('embedding_lookup shape:', sess.run(embeddings).shape)
        print('embeddings:\n', sess.run(embeddings))
        embeddings = tf.multiply(embeddings, dense_vals)
        print('embeddings x dense_vals shape:\n', sess.run(embeddings).shape)
        print('embeddings x dense_vals:\n', sess.run(embeddings))
        sum_square = tf.square(tf.reduce_sum(embeddings, 1))
        print('sum_square shape:\n', sess.run(sum_square).shape)
        print('sum_square:\n', sess.run(sum_square))
        square_sum = tf.reduce_sum(tf.square(embeddings), 1)
        print('square_sum shape:\n', sess.run(square_sum).shape)
        print('square_sum:\n', sess.run(square_sum))
        y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)
        print('feat_ids:\n', sess.run(tf.sparse_tensor_to_dense(feat_ids)))
        print('feat_vals:\n', sess.run(tf.sparse_tensor_to_dense(feat_vals)))

        # 写入文件
        # 回收线程
        coord.request_stop()
        coord.join(threads)