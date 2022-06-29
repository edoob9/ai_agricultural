# import os
# import shutil
# from model import get_model
# import glob
# from dataloader import load_image
# import numpy as np

MODEL_FILE_NAME = './models/train_model.h5'

classes = ['apple', 'mandarine', 'galic']
test_folder_name = 'test'
result_folder_name = 'result'
correct_folder_name = 'POSITIVE'
incorrect_folder_name = 'NEGATIVE'
# reslut 폴더에서 POSITIVE, NEGATIVE 폴더를 만든다.

os.makedirs(result_folder_name, exist_ok=True)
# exist_ok라는 파라미터를 True로 하면 해당 디렉토리가 기존에 존재하면 에러발생 없이 넘어가고, 없을 경우에만 생성합니다.
for class_name in classes:
    os.makedirs(f'{result_folder_name}/{class_name}', exist_ok=True)
    os.makedirs(f'{result_folder_name}/{class_name}/{correct_folder_name}', exist_ok=True)
    os.makedirs(f'{result_folder_name}/{class_name}/{incorrect_folder_name}', exist_ok=True)

model = get_model(classes)

model.load_weights(MODEL_FILE_NAME)

file_list = glob.glob(os.path.join(test_folder_name, '**', '*.*'), recursive=True)
np.random.shuffle(file_list)
idx = 0


for file_name in file_list:
    img = load_image(file_name)
    prediction = model.predict(img[None])[0]
    result = np.argmax(prediction)
    predict_label = classes[result]
    if predict_label in file_name:
        shutil.copy2(file_name, f'{result_folder_name}/{predict_label}/{correct_folder_name}/{idx}.png')
        pass
    else:
        shutil.copy2(file_name, f'{result_folder_name}/{predict_label}/{incorrect_folder_name}/{idx}.png')
    idx = idx + 1
