#!/usr/bin/env python
# coding=utf-8
'''
@File    :   dws_user_cate_score_d.py
@Time    :   2021/12/13 10:03:24
@Author  :   wangbo
@Version :   1.0
@Desc    :   FM 模型
样本格式:
feat_ids: [1, 4, 6, 7]
feat_vals:[1.0, 1.0, 1.0, 1.0]
label:[1]

数据文件说明:
训练集: tr*.tfrecords
验证集: va*.tfrecords
测试集: te*.tfrecords

训练&评估；
更多参数参考`FLAGS`的定义
>>> python FM.py --data_dir /data/rcmd_mart/supply_rank/fm_train_data/supply_rank_sparse_feature/ --model_dir /data/rcmd_mart/supply_rank/fm/tmp/ --task_type train --batch_size 500 --feature_size 71033 --num_epochs 10\
>>>    --num_threads 16 --embedding_size 64 --clear_existing_model true

评估:
>>> python FM.py --data_dir /data/rcmd_mart/supply_rank/fm_train_data/supply_rank_sparse_feature/ --model_dir /data/rcmd_mart/supply_rank/fm/tmp/ --task_type eval --batch_size 500

测试:
>>> python FM.py --data_dir /data/rcmd_mart/supply_rank/fm_train_data/supply_rank_sparse_feature/ --model_dir /data/rcmd_mart/supply_rank/fm/tmp/ --task_type infer --batch_size 500

导出pb文件:
>>> python FM.py --data_dir /data/rcmd_mart/supply_rank/fm_train_data/supply_rank_sparse_feature/ --model_dir /data/rcmd_mart/supply_rank/fm/tmp/ --task_type export --servable_model_dir /data/rcmd_mart/supply_rank/fm

参数导出json文件:
>>> python FM.py --data_dir /data/rcmd_mart/supply_rank/fm_train_data/supply_rank_sparse_feature/ --model_dir /data/rcmd_mart/supply_rank/fm/tmp/ --task_type export-json --servable_model_dir /data/rcmd_mart/supply_rank/fm

'''

# from __future__ import print_function

import shutil
import os
import glob
from datetime import date, timedelta
import json

import random
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# 参数配置
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_threads", 16, "并发线程数")
tf.app.flags.DEFINE_integer("feature_size", 71033, "特征维度")
tf.app.flags.DEFINE_integer("embedding_size", 32, "embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "训练epochs")
tf.app.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "每隔多少步打印Summary信息")
tf.app.flags.DEFINE_float("learning_rate", 0.003, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2正则化系数")
tf.app.flags.DEFINE_float("pos_weight", 20.0, "pos weight")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss类型 {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "优化器 {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("data_dir", '', "数据目录 { 训练集: tr*.tfrecords; 验证集: va*.tfrecords; 测试集: te*.tfrecords}")
tf.app.flags.DEFINE_string("dt_dir", '', "时间分区，每天训练保存一个模型")
tf.app.flags.DEFINE_string("model_dir", '/data/', "模型checkpoint目录")
tf.app.flags.DEFINE_string("servable_model_dir", '', "导出模型目录用于线上服务")
tf.app.flags.DEFINE_string("task_type", 'train', "任务类型 {train, infer, eval, export, export-json}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "清空checkpoint目录")


def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
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


def model_fn(features, labels, mode, params):
    """模型定义

    Args:
        features ([type]): 输入特征
        labels ([type]): 标签
        mode (string): 参考`tf.estimator.ModeKeys` train
        params (dict): 超参数

    Returns:
        [tf.estimator.EstimatorSpec]: estimator
    """
    # 超参
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    pos_weight = params["pos_weight"]

    # 模型参数
    FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
    FM_W = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    FM_V = tf.get_variable(name='fm_v', shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())

    # 取特征
    # tf.print(features, output_stream=tf.compat.v1.logging.INFO)
    feat_ids = features['feat_ids']
    feat_vals = features['feat_vals']

    # 线性部分
    with tf.variable_scope("Linear"):
        y_w = tf.nn.embedding_lookup_sparse(FM_W, sp_ids=feat_ids, sp_weights=feat_vals, combiner='sum')
        # tf.print(y_w, output_stream=sys.stderr)

    # 交叉部分
    with tf.variable_scope("Cross"):
        # sparse特征dense化
        dense_ids = tf.sparse_tensor_to_dense(feat_ids)  # (batch_size, batch_max_feature_size)
        dense_vals = tf.expand_dims(tf.sparse_tensor_to_dense(feat_vals), axis=-1)  # (batch_size, batch_max_feature_size, 1)
        embeddings = tf.nn.embedding_lookup(FM_V, dense_ids)  # (batch_size, batch_max_feature_size, embedding_size)
        # vij*xi
        embeddings = tf.multiply(embeddings, dense_vals)  # (batch_size, batch_max_feature_size, embedding_size)
        sum_square = tf.square(tf.reduce_sum(embeddings, 1))  # (batch_size, embedding_size)
        square_sum = tf.reduce_sum(tf.square(embeddings), 1)
        y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)

    # FM输出
    with tf.variable_scope("FM-output"):
        y_bias = FM_B * tf.ones_like(y_w, dtype=tf.float32)
        y = y_bias + y_w + y_v
        pred = tf.sigmoid(y)

    predictions = {"prob": pred}
    export_outputs = {tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # 预测
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)
    # 训练/评估
    # 计算loss
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=y, labels=labels, pos_weight=pos_weight)) + \
        l2_reg * tf.nn.l2_loss(FM_W) + \
        l2_reg * tf.nn.l2_loss(FM_V)

    # 计算auc
    eval_metric_ops = {
        "auc": tf.compat.v1.metrics.auc(labels, pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    # 优化器，先提供这些，使用默认参数
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.compat.v1.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step(), var_list=[FM_B, FM_W, FM_V])

    # 训练
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)

    # 训练和评估
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions=predictions,
        eval_metric_ops=eval_metric_ops)


def main(_):
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir
    print_args()
    # 训练集
    tr_files = glob.glob("%s/tr*.tfrecords" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("训练集:", tr_files)
    # 验证集
    va_files = glob.glob("%s/va*.tfrecords" % FLAGS.data_dir)
    print("验证集:", va_files)
    # 测试集
    te_files = glob.glob("%s/te*.tfrecords" % FLAGS.data_dir)
    print("测试集:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print("清除模型目录失败！！！", e)
        else:
            print(f"清除模型目录成功: {FLAGS.model_dir}")

    model_params = {
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
        "pos_weight": FLAGS.pos_weight
    }
    config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': FLAGS.num_threads}), log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    FM = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(FM, metric_name="loss", max_steps_without_decrease=20 * 100)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size), hooks=[early_stopping])
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(FM, train_spec, eval_spec)

    elif FLAGS.task_type == 'eval':
        FM.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))

    elif FLAGS.task_type == 'infer':
        preds = FM.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="prob")
        # 将结果写回原测试集目录
        with open(FLAGS.data_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))

    elif FLAGS.task_type == 'export':
        feature_spec = {
            'feat_ids': tf.io.VarLenFeature(tf.int64),
            'feat_vals': tf.io.VarLenFeature(tf.float32)
        }
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        FM.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)
    elif FLAGS.task_type == 'export-json':
        print(FM.get_variable_names())
        params = {'lr_b': FM.get_variable_value('fm_bias').tolist()[0], 'lr_w': FM.get_variable_value('fm_w').tolist(), 'embedding': FM.get_variable_value('fm_v').tolist()}
        with open(os.path.join(FLAGS.servable_model_dir, 'fm_params.json'), 'w') as fp:
            json.dump(params, fp)


def print_args():
    print('='*100)
    print("输入参数:\n")
    print("workspace", os.getcwd())
    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('feature_size ', FLAGS.feature_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('l2_reg ', FLAGS.l2_reg)
    print('pos_weight', FLAGS.pos_weight)
    print('='*100)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
