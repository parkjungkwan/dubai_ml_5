class BreasCancer:
    def __init__(self):
        pass

    def execute(self):
        import sklearn
        from sklearn.tree import DecisionTreeClassifier
        import sklearn.datasets
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, stratify=cancer.target, random_state=42
        )
        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(X_train, y_train)
        print('훈련세트의 정확도: {:.3f}'.format(tree.score(X_train, y_train)))
        print('테스트세트의 정확도: {:.3f}'.format(tree.score(X_test, y_test)))
        """
        독립변수의 개수가 많은 빅데이터에서는 과최적화가 쉽게 발생한다.
        """
