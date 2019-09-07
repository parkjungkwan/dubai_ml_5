class Iris():
    def __init__(self):
        pass

    def execute(self):
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        import numpy as np
        
        np.random.seed(0) # 랜덤값을 고정시키는 로직
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        # print(df.head())
        # print(df.columns)
        """
        Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
               'petal width (cm), 'species', 'is_train']
        """
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        # print(df.columns)
        df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75 # train set 75%
        # print(df.columns)
        train, test = df[df['is_train']==True], df[df['is_train']==False]
        features = df.columns[:4] # 앞에서부터 4번째 컬럼(=피쳐)까지 추출
        # print(features)
        y = pd.factorize(train['species'])[0]
        # print(y)
        # 학습 learning
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
        clf.fit(train[features], y)
        # print(clf.predict_proba(test[features])[0:10])
        # 정확도 평가
        preds = iris.target_names[clf.predict(test[features])]
        print(preds[0:5])
        # 크로스탭
        t = pd.crosstab(test['species'], preds, rownames=['Actual Species'],
                    colnames=['Predicted Species'])
        # print(t)
        # 피쳐별 중요도
        print(list(zip(train[features], clf.feature_importances_)))
        """
        [('꽃받침 sepal length (cm)', 0.11185992930506346), 
        ('꽃받침 sepal width (cm)', 0.016341813006098178), 
        ('꽃잎 petal length (cm)', 0.36439533040889194), 
        ('꽃잎 petal width (cm)', 0.5074029272799464)]
        """

