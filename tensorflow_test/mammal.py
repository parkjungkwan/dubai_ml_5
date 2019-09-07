class Mammal:
    def __init__(self):
        pass
    def execute(self):
        import tensorflow as tf
        import numpy as np
        # [털, 날개] -> 기타, 포유류, 조류
        """
            [[0,0], -> [1, 0, 0] 기타
             [1,0], -> [0, 1, 0] 포유류
             [1,1], -> [0, 0, 1] 조류
             [0,0], -> [1, 0, 0] 기타
             [0,0], -> [1, 0, 0] 기타
             [0,1]  -> [0, 0, 1] 조류
        """
        x_data = np.array(
            [[0,0],
             [1,0],
             [1,1],
             [0,0],
             [0,0],
             [0,1]
             ]
        )
        y_data = np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 0, 0],
             [1, 0, 0],
             [0, 0, 1]
             ]
        )
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)
        W = tf.Variable(tf.random_uniform([2,3], -1, 1.))
        # -1 은 all
        # 신경망 neural network 앞으로느 nn 으로 표기
        # nn 은 2차원으로 [입력층(특성), 출력층(레이블)] ->[2,3] 으로 정합니다
        b = tf.Variable(tf.zeros([3]))
        # b 는 편향 bias
        # W 는 가중치 weight
        # b 는 각 레이어의 아웃풋 갯수로 설정함
        # b 는 최종 결과값의 분류갯수인 3으로 설정함

        L = tf.add(tf.matmul(X, W), b)
        # 가중치와 편향을 이용해 계산하 결과 값에
        L = tf.nn.relu(L)
        model = tf.nn.softmax(L)
        """
        softmax 소프트맥스 함수는 다음 처럼 결과값을 전체 합이 1인 확률로 만들어 주는 함수
        예) [8.04, 2.76, -6.52] -> [0.53, 0.24, 0.23]
        """ 
        print('----- 모델 내부 보기 ------')
        print(model)
        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis = 1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(cost)
        # 비용함수를 최소화 시키면 (=경사도를 0으로 만들면) 그 값이 최적화 된 값이다.
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for step in range(100):
            sess.run(train_op, {X: x_data, Y: y_data})
            if (step + 1) % 10 == 10:
                print(step +1, sess.run(cost, {X: x_data, Y: y_data}))

        # 결과확인

        prediction = tf.argmax(model, 1)
        target = tf.arg_max(Y, 1)
        print('예측값: ', sess.run(prediction, {X: x_data}))
        print('실제값: ', sess.run(target, {Y: y_data}))
        # tf.argmax : 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 가져옴
        # 예) [[0, 1, 1][1, 0, 0]] -> [1, 0]
        # [[0.2, 0.7, 0.1][0.9, 0.1, 0.]] ->[1,0]
        is_correct = tf.equal(prediction, target)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도 : %.2f' % sess.run(accuracy * 100, {X: x_data, Y: y_data}))


        
