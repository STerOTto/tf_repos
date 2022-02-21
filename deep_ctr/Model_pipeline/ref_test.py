def test_iris():
    from sklearn.feature_selection import RFE
    from sklearn import datasets
    from sklearn.linear_model import LogisticRegression
    # 测试
    dataset = datasets.load_iris()
    model = LogisticRegression()
    rfe = RFE(model, n_features_to_select=1, step=2)
    rfe = rfe.fit(dataset.data, dataset.target)
    print(rfe.support_)
    print(rfe.ranking_)


if __name__ == '__main__':
    test_iris()