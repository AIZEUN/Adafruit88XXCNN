from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf

#학습데이터 불러오기
caltech_dir = "./trensfer"
categories = ["null", "noise", "finger"]
nb_classes = len(categories)

X = []
Y = []

for idx, cat in enumerate(categories):
    
    #one-hot 인코딩.
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.txt")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        pixel = open(f,"r")
        y = pixel.read()
        pixel.close()
        
        y = y.replace("'","")
        y = y.replace("[","")
        y = y.replace("]","")
        y = y.replace("\n","")
        y = y.split(",")
        
        for j in range(0,len(y)):
            y[j] = float(y[j])

        data = np.array(y)
        data = data.reshape(8,8)
        X.append(data)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)

#학습데이터중 훈련데이터와 학습데이터로 분활
X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./numpy_data/multi_image_data.npy", xy)

print("ok", len(y))

X_train, X_test, y_train, y_test = np.load('./numpy_data/multi_image_data.npy',allow_pickle=True)
print("training dataset:", X_train.shape)
print("test dataset:", X_test.shape)

categories = ["null", "noise", "finger"]
nb_classes = len(categories)

X_train = X_train.reshape(len(X_train),8,8,1)
X_test = X_test.reshape(len(X_test),8,8,1)
X_train = X_train.astype(float) / 25
X_test = X_test.astype(float) / 25

#CNN모델링
model = Sequential()
model.add(Conv2D(5, (3,3), padding="same", input_shape=(8,8,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
    
model.add(Conv2D(10, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.75))
    
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_dir = './model'

#모델 학습 및 학습된 모델 저장 
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_path = model_dir + '/multi_img_classification.model'
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=6)

history = model.fit(X_train, y_train, batch_size=30, epochs=5, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])

#테스트 정확도
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
