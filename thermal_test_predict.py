from PIL import Image
import glob, numpy as np
from keras.models import load_model

#예측 데이터 불러오기
caltech_dir = "./trensfer/test"

X = []
filenames = []
files = glob.glob(caltech_dir+"/*.*")

for i, f in enumerate(files):
        pixel = open(f,"r")
        y = pixel.read()
        pixel.close()
        filenames.append(f)
        
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

X = np.array(X)
X = X.reshape(len(X),8,8,1)

#예측 모델 불러와서 모델 예측
model3 = load_model('./model/multi_img_classification.model')

prediction = model3.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0

#모든 모델에 대한 예측값 보여주
for i in prediction:
    pre_ans = i.argmax()
    print(i)
    print(pre_ans)
    pre_ans_str = ''
    
    if pre_ans >= 0.8:
        pre_ans_str = "null"
    elif pre_ans >= 0.8:
        pre_ans_str = "noise"
    else:
        pre_ans_str = "finger"
        
    if i[0] >= 0.8 :
        print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[1] >= 0.8:
        print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"으로 추정됩니다.")
    if i[2] >= 0.8:
        print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    cnt += 1

