# import tensorflow as tf
# import json
# from model import get_model
# from dataloader import get_dataloader

# dataloader.py에서 구현해놓은 get_dataloader함수를 이용해서 진행

classes = ['apple', 'mandarine','galic']
train_folder_name = 'train'

model = get_model(classes)
dataloader = get_dataloader(classes, train_folder_name)
# DataGenerator(file_list, classes_list)을 return을 한다.
# -> dataloader.py에서 DataGenerator class에서 np.asarray(batch_image, dtype=np.float32), np.asarray(batch_label, dtype=np.float32)리턴한다.
model.compile(loss=tf.keras.losses.categorical_crossentropy, 
							metrics=tf.keras.metrics.categorical_accuracy)
model.fit(dataloader, callbacks=[tf.keras.callbacks.ModelCheckpoint('./models/train_model.h5')])
