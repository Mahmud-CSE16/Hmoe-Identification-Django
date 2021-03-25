from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
# Create your views here.

import os
import numpy as np
import keras
from keras.preprocessing import image
# from keras.models import Sequential
# from keras.layers import Convolution2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense
# #import tensorflow as tf 
# from keras.preprocessing.image import ImageDataGenerator
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    
def index(request):
    return render(request,'index.html')
    
#page functions

def detection(request):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #model = tf.keras.models.load_model(os.path.join(base_dir,'static/detection/covid19.model'))
    model = keras.models.load_model(os.path.join(base_dir,'static/detection/home_model.h5'))
    
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name,uploaded_file)
        url = fs.url(name)
        print(url)
        path = os.path.join(base_dir,url[1:])
        print(path)
        img = image.load_img(path,target_size=(500,500))


        # YOUR CODE HERE))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        images=np.vstack([x])
        val=model.predict(images)
        print(val)

        message = "Home number: {}".format(val[0].tolist().index(val.max())+1)
        # x=classes[0]
        # print(x[0])
        # if x[0]>0.5:
        #     message = "This x-ray result is normal."
        # else:
        #     message = "Covid-19 exist in this x-ray result."
            
        return render(request,'detection/index.html',{'message':message,'url':url[1:]})
            
    else:
        message = "Choose home image to identify."
        
    return render(request,'detection/index.html',{'message':message})
