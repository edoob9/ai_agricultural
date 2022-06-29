# 현재, 하위 디렉토리를 모두 포함하여 파일/폴더 탐색 반환
def load_image(file_name):
    img = Image.open(file_name)
    img = img.resize((64, 64)).convert('RGB')
    img = np.asarray(img).astype(np.float32)  # 데이터 형태가 같을때 복사하지 않는다.
    
    return (img - np.mean(img, axis=(0, 1))) / 255

def get_dataloader(classes, train_folder_name):
    classes_list = classes
    file_list = glob.glob(os.path.join(train_folder_name, '**', '*.*'), recursive=True)
		#glob는 파일들의 리스트를 뽑을 때 사용하는데, 파일의 경로명을 이용해서 진행
    random.shuffle(file_list)
    return DataGenerator(file_list, classes_list)
  
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, a_file_list, a_classes, a_batch_size=32):
        self.file_list = a_file_list
        self.classes = a_classes
        self.batch_size = a_batch_size # 초기화

    def __len__(self):
        return len(self.file_list) // self.batch_size 
				# len(file_list)은 모든 파일의 개수를 batch 사이즈 별로 나누는 것!
    def __getitem__(self, a_idx):
        batch_image = []
        batch_label = []
        for f_idx, file_name in enumerate(self.file_list[a_idx * self.batch_size : (a_idx + 1) * self.batch_size]):
            try:
                img = load_image(file_name)
            except:
                file_name = self.file_list[a_idx * self.batch_size + f_idx - 1]
                img = load_image(file_name)

            label = [(class_name in file_name) for class_name in self.classes]

            batch_image.append(img)
            batch_label.append(label)
        return np.asarray(batch_image, dtype=np.float32), np.asarray(batch_label, dtype=np.float32)
