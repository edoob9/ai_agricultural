# import tensorflow as tf
# import json

def get_model(classes):

    inputs = tf.keras.layers.Input(shape=(64, 64, 3)) # dataloader.py에서 진행한 load_image에서 size를 64*64크기로 진행했으며, RGB channel 3개로 진행!
    hidden = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=2, padding='same')(inputs) # 이미지 사이즈 유지를 위해서 padding을 same으로 진행했다.
    hidden = tf.keras.layers.Activation('swish')(hidden)# swish라는 이름을 가진 활성화 함수를 적용

    hidden = tf.keras.layers.Conv2D(32, 3, 2, padding='same')(hidden)
    # 이미지에 존재하는 직선이나 곡선 등의 지난번 레이어에서 추출한 특징을 활용하여 보다 복잡한 특징을 얻어낸다.
    # 같은 방법으로 모델의 성능을 향상시키기 위해서 32
    hidden = tf.keras.layers.Activation('swish')(hidden)

    hidden = tf.keras.layers.Conv2D(32, 3, 2, padding='same')(hidden) # 이전에 뽑아낸 복잡한 특징으로 더 복잡한 특징을 뽑아낸다.
    hidden = tf.keras.layers.Activation('swish')(hidden)

    hidden = tf.keras.layers.GlobalAveragePooling2D()(hidden) # 뽑은 특징을 활용하기 편한 형태로 변환한다.
    hidden = tf.keras.layers.Dense(len(classes), activation='softmax')(hidden)
    '''
      activation에 설정된 softmax 경우 각 성분들의 합이 1이되도록 변환해주는 수학 함수으로 classes_name이 만약 ['apple', 'mandarine', 'galic'] 이면
      모델 반환이 [0.7, 0.1, 0.2] 과 같이 합이 1이되도록 값을 변경해주는 함수이다.
    '''

    return tf.keras.Model(inputs, hidden)
