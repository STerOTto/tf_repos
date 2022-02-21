# encoding:utf-8
import logging
import time
import traceback

import findspark
findspark.init()
import os
import argparse
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.python import pywrap_tensorflow
from tensorflow.layers import batch_normalization as bn
import subprocess

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext


''' 一般常量设置 '''
PATH = os.path.abspath(os.path.dirname(__file__))
UP_PATH = os.path.abspath(os.path.dirname(PATH))
''' SPARK 环境 '''
SPARK_CONF = SparkConf()
SPARK_CONF.setAppName("sample train data")
SPARK_CONTEXT = SparkContext(conf=SPARK_CONF)
SPARK = (SparkSession.builder
         .enableHiveSupport()
         .config('spark.pyspark.python', '/data/python_env/rank-py3/bin/python')
         .config('spark.pyspark.driver.python', '/data/python_env/rank-py3/bin/python')
         .config('jars', '/usr/lib/hive-current/lib/json-serde-1.3.8-jar-with-dependencies.jar')
         .getOrCreate()
         )
SQL_CONTEXT = SQLContext(SPARK)


SAMPLE_SIZE = 640


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
    data = tf.parse_single_example(values, features={
        'sparse_map': tf.FixedLenFeature(shape=(), dtype=tf.string),
    })

    feature_map = data['sparse_map']
    feature_map_batches = tf.train.shuffle_batch([feature_map],
                                                 SAMPLE_SIZE,
                                                 capacity=SAMPLE_SIZE*3,
                                                 min_after_dequeue=SAMPLE_SIZE)
    return feature_map_batches


def write_to_tfrecords(file, feature_map_batches):
    """
    将数据写入TFRecords文件
    :param image_batch: 特征值
    :param label_batch: 目标值
    :return:
    """
    # 构造TFRecords存储器
    writer = tf.python_io.TFRecordWriter(file)

    # 循环将每个样本构造成一个example，然后序列化写入
    for i in range(SAMPLE_SIZE):
        # 每个样本的example
        feature_map = feature_map_batches[i].eval()
        example = tf.train.Example(features=tf.train.Features(feature={
            "sparse_map": tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature_map])),
        }))

        # 写入第i样本的example
        writer.write(example.SerializeToString())

    writer.close()


def sample(rf, wf, count=100):
    i = 0
    writer = tf.python_io.TFRecordWriter(wf)
    for record in tf.python_io.tf_record_iterator(rf):
        writer.write(record)
        i += 1
        if i >= count:
            break
    writer.close()


def download(hdfs_path, local_path, if_file=False):
    if if_file:
        local_path += '*'
    cmd = ["hadoop", "fs", "-getmerge", hdfs_path, local_path]
    if subprocess.call(cmd) != 0:
        return False
    fields = local_path.split('/')
    fields[-1] = '.' + fields[-1] + '.crc'
    os.remove('/'.join(fields))
    return True


def sample_data_from_hive(spark: SparkSession, count):
    day = '20211211'
    # https://reposhub.com/python/deep-learning/linkedin-spark-tfrecord.html
    sql_input = f"""
        SELECT
            REGEXP_REPLACE(location, "\\\\[|\\\\]", "") as feat_ids,
            REGEXP_REPLACE(value, "\\\\[|\\\\]", "") as feat_vals,
            array(is_click) as label
        FROM ymtcube.supply_rank_sparse_feature
        WHERE day = {day}
        limit {count}
    """
    df = spark.sql(sql_input)

    def trans_fn(row):
        feat_ids = row[0]
        feat_ids = [int(x) for x in feat_ids.split(',')]
        feat_vals = row[1]
        feat_vals = [float(x) for x in feat_vals.split(',')]
        label = row[2]
        return [feat_ids, feat_vals, label]
    df = df.rdd.map(trans_fn).toDF(["feat_ids", 'feat_vals', 'label'])
    df.show(10)
    hdfs_save_path = "wangbo/tmp_csv/"
    local_save_path = '/data/ymt_tmp_file/wangbo/supply_rank/'
    file_name = 'train_data/sample_data.tfrecords'
    # https://github.com/tlatkowski/tf-feature-selection/blob/master/methods/selection_wrapper.py
    df.repartition(10).write.format("tfrecords").option("recordType", "Example").save(hdfs_save_path + file_name, mode="overwrite")
    download(hdfs_save_path + file_name, local_save_path + file_name)
    return df


if __name__ == '__main__':
    local_save_path = "/data/rcmd_mart/supply_rank/"
    read_file = f'{local_save_path}train_data/supply_rank_0.tfrecords'
    write_file = '/data/ymt_tmp_file/wangbo/supply_rank/sample.tfrecords'
    # feature_map_batches = read_tfrecords(read_file)
    # with tf.Session() as sess:

    #     # 创建线程回收的协调员
    #     coord = tf.train.Coordinator()

    #     # 需要手动开启子线程去进行批处理读取到队列操作
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #     datas = sess.run(feature_map_batches)
    #     print(datas[0])

    #     # 写入文件
    #     write_to_tfrecords(write_file, feature_map_batches)
    #     # 回收线程
    #     coord.request_stop()
    #     coord.join(threads)
    # sample(read_file, write_file, 10000)
    sample_data_from_hive(SPARK, 10000)
