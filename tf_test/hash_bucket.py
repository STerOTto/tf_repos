import tensorflow as tf
sess=tf.Session()

#特征数据
features = {
    'department': ['sport', 'sport', 'drawing', 'gardening', 'travelling'],
}

#特征列
department = tf.feature_column.categorical_column_with_hash_bucket('department', 4, dtype=tf.string)
department = tf.feature_column.indicator_column(department)
#组合特征列
columns = [
    department
]

#输入层（数据，特征列）
inputs = tf.feature_column.input_layer(features, columns)

#初始化并运行
init = tf.global_variables_initializer()
sess.run(tf.tables_initializer())
sess.run(init)

v=sess.run(inputs)
print(v)