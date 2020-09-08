from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import backend as K
from keras.preprocessing import image
import os
import numpy as np
def label_smooth(y_true, y_pred):

    y_true=((1-0.1)*y_true+0.05)
    return K.categorical_crossentropy(y_true, y_pred)
    

def weight_kappa(result,test_num):
    weight=np.zeros((5,5),dtype='float')
    for i in range(5):
        for j in range(5):
            weight[i,j]=(i-j)*(i-j)/16
    fenzi=0
    for i in range(5):
        for j in range(5):
            fenzi=fenzi+result[i,j]*weight[i,j]
    fenmu=0
    for i in range(5):
        for j in range(5):
            fenmu=fenmu+weight[i,j]*result[:,j].sum()*result[i,:].sum()

    weght_kappa=1-(fenzi/(fenmu/test_num))
    return  weght_kappa

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
image_size=512
batch_size=64
model_name='DDR.h5'
test_dir='./data/DDR/test/'  
custom_test=True


if custom_test== False:   
    model=load_model('new/'+model_name+model_id[k]+'.h5')
else:
    model=load_model('new/'+model_name+model_id[k]+'.h5',custom_objects={'label_smooth': label_smooth})

test_num=0
result=np.zeros((5,5),dtype=int)
recall=np.zeros((1,5),dtype=float)
for i in range(5):
        datadirs=test_dir+str(i)+'/'
        filenames=os.listdir(datadirs)
        num=len(filenames)
        test_num=test_num+num
        valid = ImageDataGenerator()
        valid_data=valid.flow_from_directory(directory=test_dir,target_size=(image_size,image_size),
                                             batch_size=batch_size,class_mode=None,classes=str(i))
        predict=model.predict_generator(valid_data,steps=num/batch_size,verbose=1,workers=1)
        predict=np.argmax(predict,axis=-1)
        for j in range(5):
            result[i,j]=np.sum(predict==j)

right=result[0,0]+result[1,1]+result[2,2]+result[3,3]+result[4,4]
print('Acc:',right/test_num)

w_kappa=weight_kappa(result,test_num)
print('w_kappa:',w_kappa)
