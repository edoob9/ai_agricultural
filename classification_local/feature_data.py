# galic, onion,pear, potato, radish, persimmon, cabbage, apple, madarine 중 분석
# homepage에서 받은 데이터는 모두 압축 파일 형태로 되어있기 때문에 처리해줘야한다.
classes = ["apple", "mandarine", "galic"] 

# AI 허브에서 데이터셋을 다운 받은 폴더명을 적어준다.
download_folder_name = "농산물 품질(QC) 이미지"
train_folder_name = "train"
# specific_image_name = "FR45"

LABEL_FILE_NAME = '[라벨]'
CLASS_0_NAME = classes[0]  
CLASS_1_NAME = classes[1]  
CLASS_2_NAME = classes[2] 

# train_folder_name에 저장된 train이라는 이름의 폴더를 만들어 준다.
os.makedirs(train_folder_name, exist_ok=True)

# 각 클래스별로 train폴더 안에 이름을 가지는 폴더를 따로 만들어 준다.
idx = 0
os.makedirs(f'{train_folder_name}/{CLASS_0_NAME}', exist_ok=True)
os.makedirs(f'{train_folder_name}/{CLASS_1_NAME}', exist_ok=True)
os.makedirs(f'{train_folder_name}/{CLASS_2_NAME}', exist_ok=True)

# Training에 있는 압축파일만 활용하기 위해 Training추가
zip_file_list = os.listdir(f'{download_folder_name}/Training')

class0_file_list = []
class1_file_list = []
class2_file_list = []

for zip_file_name in zip_file_list:
    if not (LABEL_FILE_NAME in zip_file_name): # [라벨] 이 포함되지 않은 압축파일만 활용
        if CLASS_0_NAME in zip_file_name: # apple이 포함된 파일을 따로 추려 내는 코드
            class0_file_list.append(zip_file_name)
        elif CLASS_1_NAME in zip_file_name: # mandarine이 포함된 파일을 따로 추려 내는 코드
            class1_file_list.append(zip_file_name)
        elif CLASS_2_NAME in zip_file_name: # galic이 포함된 파일을 따로 추려 내는 코드
            class2_file_list.append(zip_file_name)
            
# 압축파일을 ZipFile을 이용한다.
for class0_file in class0_file_list: # '사과'
    zip_file = zipfile.ZipFile(f'{download_folder_name}/Training/{class0_file}')
    for file in zip_file.filelist:  # 압축 해제된 파일들을 하나씩 검색하는 코드
        if specific_image_name in file.filename:  # FR45라는 파일 이름이 있는 이미지만 선택하는 코드
            zip_file.extract(file.filename, f'{train_folder_name}/{CLASS_0_NAME}')  # 해당 이미지를 압축파일에서 압축을 풀어 train/class폴더로 옮기는 코드

for class1_file in class1_file_list: # '귤'
    zip_file = zipfile.ZipFile(f'{download_folder_name}/Training/{class1_file}')
    for file in zip_file.filelist:
        if specific_image_name in file.filename:
            zip_file.extract(file.filename, f'{train_folder_name}/{CLASS_1_NAME}')

for class2_file in class2_file_list:  # '마늘'
    zip_file = zipfile.ZipFile(f'{download_folder_name}/Training/{class2_file}')
    for file in zip_file.filelist:
       if specific_image_name in file.filename:
           zip_file.extract(file.filename, f'{train_folder_name}/{CLASS_2_NAME}')
