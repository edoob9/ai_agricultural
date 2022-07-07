import glob, os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from model import get_model
from dataloader import load_image
from utils import make_gradcam_heatmap
from PIL import Image

classes = ['apple', 'mandarine']
MODEL_FILE_NAME = './models/train_model.h5'

result_folder_name = 'result'
correct_folder_name = 'POSITIVE' # result 폴더에 맞춘 이미지가 들어 있는 폴더 이름
incorrect_folder_name = 'NEGATIVE' # result 폴더에 틀린 이미지가 들어 있는 폴더 이름

gradcam_folder_name = 'gradcam' # grad cam result가 저장될 폴더 이름
os.makedirs(gradcam_folder_name)
os.makedirs(f'{gradcam_folder_name}/{correct_folder_name}') # 각각 들린 이미지에 대한 결과는 NEGATIVE폴더안에
os.makedirs(f'{gradcam_folder_name}/{incorrect_folder_name}') # 맞춘 이미지에 대한 결과는 POSITIVE폴더 안에 저장됩니다.

model = get_model(classes)
model.load_weights(MODEL_FILE_NAME)

'''
<glob>
    glob은 파일들의 리스트를 뽑을 때 사용하는데 파일의 경로명을 이용
    file_path = glob.glob('dir/*.csv') -> dir폴더에 csv파일들의 이름만 file_path에 리스트에 저장된다.
    glob.glob(**, recursive = True) // 하위 디렉토리 검색 포함을 허용
    ['111.png' ,'aaa.txt', 'bbb.txt', 'example', 'example/333.png', 'example/ccc.txt']
    -> 현재 디렉토리 + 하위 폴더 내의 모든 파일 및 폴더들을 색인하고 싶은 경우
<os.path.join>
    함수 파라미터에 생성하고 싶은 경로의 문자열만 입력한다. a/b/c경로를 만들고 싶다!
    r = os.path.join(a,b,c)
    print(r) 
'''
incorrect_file_list = glob.glob(os.path.join(result_folder_name, '*', incorrect_folder_name, '*.*'), recursive=True)
# result folder에 각각 사과와 귤 폴더안에 NEGATIVE라는 폴더안에 있는 이미지 파일목록을 불러오는 코드
correct_file_list = glob.glob(os.path.join(result_folder_name, '*', correct_folder_name, '*.*'), recursive=True)
# result folder에 각각 사과와 귤 폴더안에 POISTIVE라는 폴더안에 있는 이미지 파일목록을 불러오는 코드

idx = 0
for file_name in correct_file_list: # 정답을 맞춘 이미지에 대해서 한장씩 gradcam 결과를 시각화
    img = load_image(file_name) # dataloader.py에서 제시한 load_image()함수를 이용해서!
    f, ax = plt.subplots(1, 3)
    # plt.subplot(2,1,2) = plt.subplots(nrows=2, ncols=1)
    plt.suptitle("Grad Cam Result")
    grads = make_gradcam_heatmap(img[None], model)
    jet_heatmap = np.uint8(cm.rainbow(grads)[..., :3] * 255) # np.uint8 -> 타입을 바꾸는 것
    jet_heatmap = Image.fromarray(jet_heatmap) # Image.fromarray함수를 사용하여 배열을 PIL 이미지 객체로 다시 변환한다.
    img = Image.open(file_name)
    img = np.asarray(img.resize((64, 64)).convert('RGB'))
    jet_heatmap = np.asarray(jet_heatmap.resize((64, 64)))
    # 3개의 이미지를 보여준다.
    ax[0].imshow(img)
    ax[1].imshow(jet_heatmap)
    ax[2].imshow(np.uint8(np.clip(img * 0.5 + jet_heatmap * 0.5, 0, 255)))
    # np.clip(img * 0.5 + jet_heatmap * 0.5, 0 , 255) -> numpy.clip(array,min,max) // array내의 element들에 대해서 min값보다 작은 값들을 min=0값으로 바꾸고, max값보다 큰 값들을 max값으로 바꾸는 함수
    plt.savefig(f'{gradcam_folder_name}/{correct_folder_name}/{os.path.split(file_name)[-1]}')
    # plt.savefig함수에 파일 이름을 입력해주면 이미지 파일이 저장된다.
    f.clf()
    plt.clf()
    plt.close()

idx = 0
for file_name in incorrect_file_list: # 정답을 맞추지 못한 이미지에 대해서 한장씩 gradcam 결과를 시각화
    img = load_image(file_name)
    f, ax = plt.subplots(1, 3)
    plt.suptitle("Grad Cam Result")
    grads = make_gradcam_heatmap(img[None], model)
    jet_heatmap = np.uint8(cm.rainbow(grads)[..., :3] * 255)
    jet_heatmap = Image.fromarray(jet_heatmap)
    img = Image.open(file_name)
    img = np.asarray(img.resize((64, 64)).convert('RGB'))
    jet_heatmap = np.asarray(jet_heatmap.resize((64, 64)))
    ax[0].imshow(img)
    ax[1].imshow(jet_heatmap)
    ax[2].imshow(np.uint8(np.clip(img * 0.5 + jet_heatmap * 0.5, 0, 255)))
    plt.savefig(f'{gradcam_folder_name}/{incorrect_folder_name}/{idx}.png')
    f.clf()
    plt.clf()
    plt.close()
    idx += 1
    
    
'''
학습이 잘되지않아서 layer를 더깊게 쌓아야한다.
model.py
hidden = tf.keras.layers.Conv2D(128,3,2,padding='same')(hidden)
hidden = tf.keras.layers.Activation('swish')(hidden)
더 추가하면, feature을 더 많이 추출한다


train.py ->evalute.py -> test.py(실행하기 전에 result folder를 삭제해준다.) -> 다시 visualization.py를 진행해준다.
'''






