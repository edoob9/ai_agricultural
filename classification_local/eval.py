# import os
# import json
# from model import get_model
# import glob
# from dataloader import load_image
# import numpy as np

MODEL_FILE_NAME = './models/train_model.h5' # 학습 딥러닝 모델 결과가 저장된 파일명

classes = ['apple', 'mandarine', 'galic']
train_folder_name = 'train'

model = get_model(classes) # get_model은 Naive Bayes 모델을 반환합니다.
model.load_weights(MODEL_FILE_NAME) # 학습된 딥러닝 모델을 불러오는 코드

file_list = glob.glob(os.path.join(train_folder_name, '**', '*.*'), recursive=True)
# train_folder_name폴더 안에 있는 이미지 이름을 모두 불러오는 코드

total = 0 # 총 데이터의 갯수가 들어갈 변수
correct = 0 # total 중 맞은 데이터의 갯수가 들어갈 변수

for file_name in file_list:
    img = load_image(file_name) # train폴더 안에 있는 이미지 파일 1개씩 읽기
    prediction = model.predict(img[None])[0]# 읽은 이미지를 딥러닝 모델을 통해서 분류 [사과 확률, 귤 확률, 마늘 확률]
    result = np.argmax(prediction)
    predict_label = classes[result]# 그 3개의 값 중, 0번째에 있는 값이 크다면 사과로 1번째에 있는 값이 크면 귤, 2번째 있는 값이 크면 마늘로 설정 해주는 코드
    if predict_label in file_name:
        correct = correct + 1
    total = total + 1

accuracy = round(correct / total * 100, 2)
print(f"학습한 데이터에 대해서 100개 중에 {accuracy}개를 맞췄습니다.")

'''
...(생략)
1/1 [==============================] - 0s 16ms/step
1/1 [==============================] - 0s 17ms/step
1/1 [==============================] - 0s 17ms/step
1/1 [==============================] - 0s 16ms/step
학습한 데이터에 대해서 100개 중에 57.87개를 맞췄습니다.

'''
