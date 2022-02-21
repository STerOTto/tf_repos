import tensorflow as tf
import sys


def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=True):
    """
    数据集解析
    """

    def parse_single_exmpl(serialized_example):
        """样本解析
        样本格式:
        feat_ids: [1, 4, 6, 7]
        feat_vals:[1.0, 1.0, 1.0, 1.0]
        label:[1]

        Args:
            serialized_example (Tensor[string]): 输入样本

        Returns:
            [type]: data, label
        """
        feature_map = {
            "feat_ids": tf.io.VarLenFeature(tf.int64),
            "feat_vals": tf.io.VarLenFeature(tf.float32),
            'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        }
        data = tf.parse_single_example(serialized_example, feature_map)
        label = data.pop('label')
        label = tf.cast(label, tf.float32)
        return data, label

    dataset = tf.data.TFRecordDataset(filenames).map(lambda x: parse_single_exmpl(x), num_parallel_calls=10).prefetch(1000)
    # 数据shuffle, 窗口大小1024
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=1024)

    # 设置训练轮数
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


if __name__ == '__main__':
    data = input_fn('test.tfrecords', perform_shuffle=False)
    features = data[0]
    label = data[1]
    cons = tf.constant([1, 2, 3, 4, 5], shape=[5], dtype=tf.float16)
    FM_W = tf.get_variable(name='fm_w', initializer=cons)
    # feat_ids = features['feat_ids']
    # feat_vals = features['feat_vals']
    
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        print('data:\n', sess.run(data))
        # print('features:\n', sess.run([features, label]))
        # print('FM_V:\n', sess.run(FM_W))
        # feat_ids = features['feat_ids']
        # feat_vals = features['feat_vals']
        # tmp = sess.run(features)
        # print(
        #     'feat_ids:\n', tmp['feat_ids'],
        #     'feat_vals:\n', tmp['feat_vals'],
        #     'label:\n', sess.run(label)
        # )

        # print('embedding:\n')
        # y_w = tf.nn.embedding_lookup_sparse(FM_W, sp_ids=feat_ids, sp_weights=feat_vals, combiner='sum')
        # print('y_w:', sess.run(y_w))

        # print(sess.run(exmpl))
        # print(data['feat_ids'])
        # print(sess.run(exmpl['feat_ids']))
        # print(sess.run(exmpl['feat_vals']))