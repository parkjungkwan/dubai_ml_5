class RamPrice:
    def __init__(self):
        pass

    def execute(self):
        import pandas as pd
        import os
        import matplotlib.pyplot as plt
        import mglearn
        import numpy as np

        ram_price = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
        plt.semilogy(ram_price.date, ram_price.price)
        plt.xlabel("년")
        plt.ylabel("가격")
        # plt.show()

        from sklearn.tree import DecisionTreeRegressor
        from sklearn.linear_model import LinearRegression
        data_train = ram_price[ram_price['date'] < 2000]  # 2000 기준
        data_test = ram_price[ram_price['date'] >= 2000]  # 지도
        x_train = data_train['date'][:, np.newaxis]  # train data 를 1열로 만든다
        y_train = np.log(data_train['price'])
        tree = DecisionTreeRegressor().fit(x_train, y_train)
        lr = LinearRegression().fit(x_train, y_train)
        # test 는 모든 데이터(1960 ~ 2010)에 대해 적용한다
        x_all = ram_price['date'].values.reshape(-1, 1)  # x_all 을 1열로 만든다
        pred_tree = tree.predict(x_all)
        price_tree = np.exp(pred_tree)  # log 값 되돌리기
        pred_lr = lr.predict(x_all)
        price_lr = np.exp(pred_lr)  # log 값 되돌리기

        plt.semilogy(ram_price['date'], pred_tree,
                     label="TREE PREDIC", ls='-', dashes=(2, 1))
        plt.semilogy(ram_price['date'], pred_lr,
                     label="LINEAR REGRESSION PREDIC", ls=':')
        plt.semilogy(data_train['date'], data_train['price'], label='TRAIN DATA', alpha=0.4)
        plt.semilogy(data_test['date'], data_test['price'], label='TEST DATA')
        plt.legend(loc=1)
        plt.xlabel('year', size=15)
        plt.ylabel('price', size=15)
        plt.show()