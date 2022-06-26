import numpy as np
import tensorflow as tf

FILE_NAME = '서교동_기온_202106_202205.csv'

def load_data():
    data = []
    with open(f'./data/{FILE_NAME}') as f:
        header = f.readline()
        day_data = []
        while 1:
            line = f.readline() # day,hour,value location:59_126 Start : 20210601 - ',' 로 구분지어져 있음
            if line.strip() == '':
                break # 마지막줄은 공백이기 때문에 읽기를 종료하도록 설정
            split_line = line.split(',') # ["1", "200", "24"]
            if len(split_line) != 3: # 1달을 넘어가는 경우, 해당 열에는 데이터의 시작 날짜가 표기되기 때문에 ','로 구분했을때 4개의 값이 존재하지 않음
                pass
            else: # ','로 구분했을때 4개의 값이 있고, 각각 순서대로
                day, hour, temperature = split_line
                day_data.append(temperature)
            if len(day_data) == 24: # 하루에  (0~23시) 24시간씩 데이터를 나눠서 저장
                day_data = np.asarray(day_data, dtype=np.float32)
                if np.min(day_data) <= -50:
                    day_data[day_data<=-50] = np.mean(day_data)
                data.append(day_data)
                day_data = []
    X = np.asarray(data, dtype=np.float32)
    X = X
    Y = np.mean(X, axis=-1) # 하루의 평균 온도를 계산
    # 모은 데이터의 첫날,마지막날의 온도를 통해서 전후 데이터를 알 수 없으니까 제외시킨다.
    X = X[:-1]
    Y = Y[1:]

    return X, Y

def train_model(X, Y):
    inputs = tf.keras.layers.Input((24, ))
    hidden1 = tf.keras.layers.Dense(32, activation='relu')(inputs)
    hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
    outputs = tf.keras.layers.Dense(1)(hidden2)
    models = tf.keras.Model(inputs, outputs)

    models.compile(optimizer='adam', loss=tf.keras.losses.huber)
    models.fit(X, Y, epochs=500)
    models.save('./data/model.h5')

if __name__ == '__main__':
    train_model(*load_data())
