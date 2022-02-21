import numpy as np
import tensorflow as tf
sess=tf.Session()

#特征数据
features = {
    'sale': [0.1, 0.2, 0.5, 1.0, 0.2]
}

#特征列
step_val = 1.0 / 2
boundaries = list(np.arange(0, 1, step_val))
sale = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('sale',default_value=0.0), boundaries=boundaries)
#组合特征列
columns = [
    sale
]

#输入层（数据，特征列）
inputs = tf.feature_column.input_layer(features, columns)

#初始化并运行
init = tf.global_variables_initializer()
sess.run(tf.tables_initializer())
sess.run(init)

v=sess.run(inputs)
print(v)
