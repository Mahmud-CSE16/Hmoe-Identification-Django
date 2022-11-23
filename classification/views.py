from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
# Create your views here.

import os
import numpy as np
import keras
from keras.preprocessing import image

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

"""MobileNet"""
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
from PIL import Image, ImageFont


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(os.path.join(base_dir+'/static/mobilenet/pipeline.config'))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)


# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(base_dir+'/static/mobilenet/ckpt-21')).expect_partial


category_index = label_map_util.create_category_index_from_labelmap(os.path.join(base_dir+'/static/mobilenet/label_map.pbtxt'))
print(category_index)


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections



def detect_by_mobilenet(frame,fileName):
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    
#     print(category_index[detections['detection_classes'][0]+1])
#     print(detections['detection_scores'][0])
    

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=2,
                min_score_thresh=.50,
                agnostic_mode=False,
                line_thickness=5)
    
    cv2.imwrite(os.path.join(base_dir+'/media/{}'.format(fileName)), image_np_with_detections)





"""YOLO"""
## provide the path for testing cofing file and tained model form colab
net = cv2.dnn.readNetFromDarknet(os.path.join(base_dir+'/static/yolo/yolov3_custom.cfg'), os.path.join(base_dir+'/static/yolo/yolov3_custom_final.weights'))

### Change here for custom classes for trained model 
classes = ['Brick House','Mosque','Mud House','Skyscraper','Temple','Tin-shade House']

def detect_by_yolo(img,fileName):
    print(img.shape)
    hight,width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)

    detected_classes = []
    detected_boxes = []
    
    
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
        
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > CONFIDENCE:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
                
                #predict data
                detected_classes.append(classes[class_id])
                detected_boxes.append([x, y, x+w, y+h])
                
   
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,SCORE_THRESHOLD,IOU_THRESHOLD)
    
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    if  len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,label + " " + confidence, (x+5,y+27),font,1.5,color,2)

    
    
    cv2.imwrite(os.path.join(base_dir+'/media/{}'.format(fileName)), img)
   




    
def index(request):
    return render(request,'index.html')
    
#page functions

def classification(request):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #model = tf.keras.models.load_model(os.path.join(base_dir,'static/detection/covid19.model'))
    model = keras.models.load_model(os.path.join(base_dir,'static/classification/HybridCNNModel.h5'))
    
    if request.method == 'POST':
        selectedModel = request.POST["model"]
        print(selectedModel)
        if(selectedModel == "Hybrid CNN"):
            print("Hybrid CNN")
            model = keras.models.load_model(os.path.join(base_dir,'static/classification/HybridCNNModel.h5'))
        elif(selectedModel == "Sequencial CNN"):
            print("Sequencial CNN")
            model = keras.models.load_model(os.path.join(base_dir,'static/classification/Seq_CNN.h5'))
        elif(selectedModel == "DenseNet"):
            print("DenseNet")
            model = keras.models.load_model(os.path.join(base_dir,'static/classification/DenseNet.h5'))
        elif(selectedModel == "MobileNet"):
            print("MobileNet")
            model = keras.models.load_model(os.path.join(base_dir,'static/classification/MobileNet.h5'))
        elif(selectedModel == "VGG16"):
            print("VGG16")
            model = keras.models.load_model(os.path.join(base_dir,'static/classification/VGG16.h5'))
        elif(selectedModel == "Xception"):
            print("Xception")
            model = keras.models.load_model(os.path.join(base_dir,'static/classification/Xception.h5'))
        elif(selectedModel == "ResNet"):
            print("ResNet")
            model = keras.models.load_model(os.path.join(base_dir,'static/classification/ResNet.h5'))



        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name,uploaded_file)
        url = fs.url(name)
        print(url)
        path = os.path.join(base_dir,url[1:])
        print(path)
        # img = image.load_img(path,target_size=(500,500))
        img = image.load_img(path,target_size=(224,224))


        # YOUR CODE HERE))
        random_image=image.img_to_array(img)
        # x=np.expand_dims(x,axis=0)
        # images=np.vstack([x])
        test_image=random_image.reshape((1,)+random_image.shape)
        test_image=test_image/255.0
        print(test_image.shape)
        predicted_class=model.predict(test_image)
        print(predicted_class)
        #print('True classification')
        print(np.argmax(predicted_class))
        val=np.argmax(predicted_class)
        # val=model.predict(images)
        # print(val)
        classValue = ""
        if(val == 0):
            classValue = "Brick House"
        elif(val == 1):
            classValue = "Mosque"
        elif(val == 2):
            classValue = "Mud House"
        elif(val == 3):
            classValue = "SkyScraper"
        elif(val == 4):
            classValue = "Temple"
        elif(val == 5):
            classValue = "Tinshed House"

        # message = "Home number: {}".format(val[0].tolist().index(val.max())+1)
        message = "Detected Building: {}".format(classValue)
        # x=classes[0]
        # print(x[0])
        # if x[0]>0.5:
        #     message = "This x-ray result is normal."
        # else:
        #     message = "Covid-19 exist in this x-ray result."
            
        return render(request,'classification/index.html',{'message':message,'url':url[1:]})
            
    else:
        message = "Please choose an building image and upload to detect building"
        
    return render(request,'classification/index.html',{'message':message})



def detection(request):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   
    
    if request.method == 'POST':
        selectedModel = request.POST["model"]



        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name,uploaded_file)
        url = fs.url(name)
        print(url)
        path = os.path.join(base_dir,url[1:])
        print(path)
        # img = image.load_img(path,target_size=(500,500))
        image = cv2.imread(path)

        print(selectedModel)
        if(selectedModel == "MobileNet"):
            detect_by_mobilenet(image, "detected_image.jpg")
        elif(selectedModel == "YOLO"):
            print("YOLO")
            detect_by_yolo(image, "detected_image.jpg")

        # message = "Home number: {}".format(val[0].tolist().index(val.max())+1)
        message = "Building Detected"
        # x=classes[0]
        # print(x[0])
        # if x[0]>0.5:
        #     message = "This x-ray result is normal."
        # else:
        #     message = "Covid-19 exist in this x-ray result."
            
        return render(request,'detection/index.html',{'message':message,'url':url[1:],'detected_img':'media/detected_image.jpg'})
            
    else:
        message = "Please choose an building image and upload to detect building"
        
    return render(request,'detection/index.html',{'message':message})
