'''
@File    :   FMEstimator.py
@Time    :   2021/12/21 10:03:24
@Author  :   wangbo
@Version :   1.0
@Desc    :   FM模型兼容sklearn，用于特征选择

'''
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from FM import model_fn
import shutil
import os
import numpy as np
from scipy.sparse import csr_matrix

he_init = tf.contrib.layers.variance_scaling_initializer()


def csr_matrix_gen_indices(X):
    coo = X.transpose(copy=True).tocoo().astype(np.int64)
    indices = np.mat([coo.col, coo.row]).transpose()
    return indices, coo.row


def csr_matrix_input_fn(X, y=None, batch_size=32, n_epochs=1):
    """dense存储输入转换

    Args:
        X ([type]): 特征dense矩阵
        y ([type], optional): label. Defaults to None.
        batch_size (int, optional): [description]. Defaults to 32.
        n_epochs (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    X = csr_matrix(X, dtype=np.float32)
    indices, feat_ids = csr_matrix_gen_indices(X)
    feat_ids = tf.SparseTensor(indices, feat_ids, X.shape)
    feat_vals = tf.SparseTensor(indices, X.data, X.shape)
    features = {"feat_ids": feat_ids, "feat_vals": feat_vals}
    if y is not None:
        labels = tf.constant(y, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((features))
    dataset = dataset.prefetch(100000)
    dataset = dataset.shuffle(1000).repeat(n_epochs).batch(batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def with_label_input_fn(feat_ids, feat_vals, labels, batch_size, n_epochs):
    """稀疏存储输入转换

    Args:
        feat_ids (list): 稀疏存储特征id列表；eg: [[1, 2, 3], [2, 4], [0, 5], [3]]
        feat_vals (list): 稀疏从存储特征值列表；eg: [[1.0, 0.2, 0.3], [0.2, 0.4], [1.0, 0.5], [0.3]]
        labels (list): 样本标签；eg: [1, 0, 1, 0]
        batch_size ([type]): [description]
        n_epochs ([type]): [description]

    Returns:
        [type]: [description]
    """
    row = [[i] * len(feat_ids[i]) for i in range(len(feat_ids))]
    row = [i for idxs in row for i in idxs]
    feat_ids = [i for idxs in feat_ids for i in idxs]
    feat_vals = [val for vals in feat_vals for val in vals]
    indices = [[row[i], feat_ids[i]] for i in range(len(row))]
    feat_ids = tf.SparseTensor(indices, feat_ids, (4, 8))
    feat_vals = tf.SparseTensor(indices, feat_vals, (4, 8))
    labels = tf.constant(labels, dtype=tf.float32)
    features = {"feat_ids": feat_ids, "feat_vals": feat_vals}
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(1000).repeat(1).batch(batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def without_label_input_fn(feat_ids, feat_vals, batch_size):
    """稀疏存储输入转换

    Args:
        feat_ids (list): 稀疏存储特征id列表；eg: [[1, 2, 3], [2, 4], [0, 5], [3]]
        feat_vals (list): 稀疏从存储特征值列表；eg: [[1.0, 0.2, 0.3], [0.2, 0.4], [1.0, 0.5], [0.3]]
        batch_size ([type]): [description]

    Returns:
        [type]: [description]
    """
    row = [[i] * len(feat_ids[i]) for i in range(len(feat_ids))]
    row = [i for idxs in row for i in idxs]
    feat_ids = [i for idxs in feat_ids for i in idxs]
    feat_vals = [val for vals in feat_vals for val in vals]
    indices = [[row[i], feat_ids[i]] for i in range(len(row))]
    feat_ids = tf.SparseTensor(indices, feat_ids, (4, 8))
    feat_vals = tf.SparseTensor(indices, feat_vals, (4, 8))
    features = {"feat_ids": feat_ids, "feat_vals": feat_vals}
    dataset = tf.data.Dataset.from_tensor_slices((features))
    dataset = dataset.shuffle(1000).repeat(1).batch(batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features = iterator.get_next()
    return batch_features


class CusLossHook(tf.train.SessionRunHook):
    """
    获取loss hook
    """
    def __init__(self, loss_name):
        self.loss_name = loss_name
        self.best_params = None
        self.step = 0
        self.loss = 0

    def begin(self):
        self._loss_tensor = tf.get_default_graph().as_graph_element(
            self.loss_name + ":0")

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._loss_tensor)

    def after_run(self, run_context, run_values):
        self.loss = run_values.results
        self.step += 1

    def get_best_params(self):
        return self.best_params

    def get_loss(self):
        return self.loss


class FMEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self,
                 feature_size=8,
                 embedding_size=4,
                 learning_rate=1e-5,
                 l1_reg=1e-3,
                 l2_reg=1e-3,
                 pos_weight=1.0,
                 optimizer='Ftrl',
                 gpu_cnt=0,
                 num_threads=10,
                 dt_dir='20211222',
                 init_dt_dir='20211222',
                 model_dir='/data/FMEstimator/model_dir',
                 servable_model_dir='/data/FMEstimator/servable_model_dir'):
        """所有的参数只能使用self.param = param, 不能进行修改和重命名, 否则导致无法clone
        参考: https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects

        Args:
            feature_size (int, optional): [description]. Defaults to 8.
            embedding_size (int, optional): [description]. Defaults to 8.
            learning_rate ([type], optional): [description]. Defaults to 1e-5.
            l1_reg ([type], optional): [description]. Defaults to 1e-3.
            l2_reg ([type], optional): [description]. Defaults to 1e-3.
            pos_weight (float, optional): [description]. Defaults to 1.0.
            optimizer (str, optional): [description]. Defaults to 'Ftrl'.
            gpu_cnt (int, optional): [description]. Defaults to 0.
            num_threads (int, optional): [description]. Defaults to 10.
            dt_dir (str, optional): [description]. Defaults to '20211222'.
            init_dt_dir (str, optional): [description]. Defaults to '20211222'.
            model_dir (str, optional): [description]. Defaults to './model_dir'.
            servable_model_dir (str, optional): [description]. Defaults to './servable_model_dir'.
        """
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.pos_weight = pos_weight
        self.optimizer = optimizer
        self.gpu_cnt = gpu_cnt
        self.num_threads = num_threads
        self.dt_dir = dt_dir
        self.init_dt_dir = init_dt_dir
        self.model_dir = model_dir
        self.servable_model_dir = servable_model_dir
        # 当且仅当 warm_start_dir 目录存在 且 model_dir 不存在时生效
        warm_start_dir = model_dir + self.init_dt_dir
        if not os.path.exists(warm_start_dir):
            warm_start_dir = None
        if os.path.exists(model_dir):
            warm_start_dir = None
        model_params = {
            "feature_size": self.feature_size,
            "embedding_size": self.embedding_size,
            "learning_rate": self.learning_rate,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
            "pos_weight": self.pos_weight,
            "opt": self.optimizer
        }
        self.FM = self._build_model(model_params, model_dir, gpu_cnt,
                                    num_threads, warm_start_dir)

    def _build_model(self, model_params, model_dir, gpu_cnt, num_threads,
                     warm_start_dir):
        config = tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(device_count={
                'GPU': gpu_cnt,
                'CPU': num_threads
            }),
            log_step_count_steps=100,
            save_summary_steps=100)
        return tf.estimator.Estimator(model_fn=model_fn,
                                      model_dir=model_dir,
                                      params=model_params,
                                      config=config,
                                      warm_start_from=warm_start_dir)

    def fit(self, X, y, batch_size: int = 32, n_epochs: int = 10, refit=True):
        """[summary]
        Args:
            X ([type]): {array-like, sparse matrix}  (n_samples, n_features)
            y ([type]): array-like of shape (n_samples,)
            batch_size (int, optional): [description]. Defaults to 32.
            n_epochs (int, optional): [description]. Defaults to 10.
            refit (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        if refit:
            try:
                print(f'要删除:{self.model_dir}')
                shutil.rmtree(self.model_dir)
            except Exception as e:
                print("清除模型目录失败！！！", e)
            else:
                print(f"清除模型目录成功: {self.model_dir}")
        loss_hook = CusLossHook('loss')
        self.FM.train(
            input_fn=lambda: csr_matrix_input_fn(X, y, batch_size, n_epochs),
            hooks=[loss_hook])

        if loss_hook.get_best_params():
            print('get_best_params')
        # TODO 暂时用loss识别特征重要程度
        # TODO 后续可用更复杂的方式
        self.feature_importances_ = np.array(
            [1 / (abs(loss_hook.get_loss()) + 1e-5)] * X.shape[1])
        # self.feature_importances_ = np.abs(self.FM.get_variable_value("fm_w")*(1 / (abs(loss_hook.get_loss())+1e-5)))
        return self

    def predict_proba(self, X):
        preds = self.FM.predict(input_fn=lambda: csr_matrix_input_fn(X),
                                predict_keys="prob")
        return preds

    def predict(self, X):
        preds = self.FM.predict(input_fn=lambda: csr_matrix_input_fn(X),
                                predict_keys="prob")
        return preds

    def get_feature_importances_(self):
        return self.feature_importances_

    def save(self, path):
        feature_spec = {
            'feat_ids': tf.io.VarLenFeature(tf.int64),
            'feat_vals': tf.io.VarLenFeature(tf.float32)
        }
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec)
        self.FM.export_savedmodel(self.servable_model_dir,
                                  serving_input_receiver_fn)


def test_iris():
    from sklearn.feature_selection import RFE
    from sklearn import datasets
    # 测试
    dataset = datasets.load_iris()
    fm = FMEstimator(feature_size=dataset.data.shape[1], embedding_size=2)
    rfe = RFE(fm, n_features_to_select=1, step=2)
    rfe = rfe.fit(dataset.data, dataset.target)
    print(rfe.support_)
    print(rfe.ranking_)


if __name__ == '__main__':
    test_iris()
