# encoding:utf-8
'''
@File    :   tfrecords_gennerator.py
@Time    :   2021/12/14 10:03:24
@Author  :   wangbo
@Version :   1.0
@Desc    :   生成tfrecords数据
'''
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
sys.path.append(UP_PATH)
from supply_rank_utils import *
from supply_rank_conf import *
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


def run(spark: SparkSession, day, table='supply_rank_sparse_feature', duration=3, ratio=[0.9, 0.05, 0.05]):
    """
    TODO 均衡正负样本
    Args:
        day ([str]): 最新日期
        table ([str]): 下载数据集表
        duration ([int]): 往前推x天
        ratio (list): 训练集、验证集、测试集比例
    """
    # spark 数据类型与tfrecords数据类型映射 https://reposhub.com/python/deep-learning/linkedin-spark-tfrecord.html
    min_day = datetime_calc(day, -duration)
    sql_input = f"""
        SELECT
            row_number() over(order by imp_time ASC) as rn,
            REGEXP_REPLACE(location, "\\\\[|\\\\]", "") as feat_ids,
            REGEXP_REPLACE(value, "\\\\[|\\\\]", "") as feat_vals,
            array(is_click) as label
        FROM ymtcube.{table}
        WHERE day <= {str(day)}
        AND day > {str(min_day)}
        ORDER BY rn ASC
    """
    df = spark.sql(sql_input)

    def trans_fn(row):
        """ 数据类型转换 """
        rn = row[0]
        feat_ids = row[1]
        feat_ids = [int(x) for x in feat_ids.split(',')]
        feat_vals = row[2]
        feat_vals = [float(x) for x in feat_vals.split(',')]
        label = row[3]
        return [rn, feat_ids, feat_vals, label]
    df = df.rdd.map(trans_fn).toDF(['rn', 'feat_ids', 'feat_vals', 'label'])
    df.show(10)
    cnt = df.count()
    if cnt < 1000:
        print('训练数据过少！！！')
        return 1
    # 数据集划分
    sp = {
        # 数据范围，左开，右闭
        'tr': (0, int(cnt * ratio[0])),
        'va': (int(cnt * ratio[0]), int(cnt * sum(ratio[:2]))),
        'te': (int(cnt * sum(ratio[:2])), cnt)
    }
    hdfs_save_path = "wangbo/tmp_csv/"
    # local_save_path = '/data/ymt_tmp_file/wangbo/supply_rank/'
    for prefix, (min_rn, max_rn) in sp:
        file_name = f'fm_train_data/{table}/{prefix}_data_{str(int(day)%duration)}.tfrecords'
        write_df = df.where(f'rn = {min_rn} and rn <= {max_rn}').select(['feat_ids', 'feat_vals', 'label'])
        write_df.repartition(100).write.format("tfrecords").option("recordType", "Example").save(hdfs_save_path + file_name, mode="overwrite")
        download(hdfs_save_path + file_name, local_save_path + file_name)
    return 0


if __name__ == '__main__':
    # local_save_path = "/data/rcmd_mart/supply_rank/"
    parser = argparse.ArgumentParser(description='tfrecords_gennerator')
    parser.add_argument('-t',
                        default=time.strftime("%Y%m%d", time.localtime(time.time()-3600*24)),
                        help="statistics day (str): 20210308")
    parser.add_argument('--table',
                        type=str,
                        default='supply_rank_sparse_feature',
                        help="table name")
    parser.add_argument('-d', '--duration',
                        default=3,
                        help="训练数据时长")

    try:
        args = parser.parse_args()
    except Exception as e:
        print(e)
        exit(1)
    exit(run(SPARK, args.t, args.table, args.duration))
