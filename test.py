
# coding: utf-8

# In[ ]:


'''

!mkdir tb1
LOG_DIR = './tb1'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

! wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
! unzip ngrok-stable-linux-amd64.zip

get_ipython().system_raw('./ngrok http 6006 &')
! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

get_ipython().system('mkdir tb1')
LOG_DIR = './tb1'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

#! wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
#! unzip ngrok-stable-linux-amd64.zip

get_ipython().system_raw('./ngrok http 6006 &')
get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python3 -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')



'''


# In[ ]:


import os
import numpy as np
np.shape(os.listdir('input/images'))


# In[ ]:


#get_ipython().system('ls')


# In[ ]:


#!ls gdrive/'My Drive'/Flipkart/Mob-30/ 
#!cp gdrive/'My Drive'/Flipkart/Mob-30/model-0.87.h5 /content/
#!ls


# In[ ]:


#get_ipython().system('ls')
#!rm -rf model-0.87.h5


# In[6]:


#get_ipython().system('rm -rf train.py')
#from google.colab import files
#files.upload()
#get_ipython().system('ls')


# In[ ]:


import os
import pandas as pd

import cv2
import numpy as np

from train import  IMAGE_SIZE, ALPHA, VALIDATION_DATASET_SIZE
from trainMobileNet import IMAGE_SIZE, ALPHA, VALIDATION_DATASET_SIZE
DEBUG = True
import matplotlib.pyplot as plt


import matplotlib.patches as patches
#WEIGHTS_FILE = "model_alfa025_size128-0.77.h5"
#WEIGHTS_FILE = "model_alfa05_size160-0.81.h5"
WEIGHTS_FILE = "weights/Res--610.53--0.83.h5"
#WEIGHTS_FILE = "Colab Outputs/Mobilenet 10 epochs/model-0.82.h5"
from keras.models import load_model
# Assuming your model includes instance of an "AttentionLayer" class
lab=pd.read_csv('input/test.csv')
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *
from keras.utils import Sequence
from keras.callbacks import TensorBoard
from keras import layers
from keras.activations import relu


# In[ ]:


from keras.applications.resnet50 import ResNet50
def create_model(size, alpha):
    
    model_net = ResNet50(input_shape=(size[1], size[0], 3), include_top=False, weights = None)
    #model_net =Xception(input_shape=(size[1], size[0], 3), include_top=False, weights = None)
   # for i in range(len(model_net.layers) - 13):
    #    model_net.layers[i].trainable=False
    #x = _depthwise_conv_block(model_net.layers[-1].output, 1024/4, alpha, 1, block_id=14)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(4, kernel_size=(1, 1), padding="same")(model_net.layers[-1].output)
    x = GlobalAveragePooling2D()(x)
    #x = Reshape((4, ))(x)
    Model1=Model(inputs=model_net.input, outputs=x)
    
    print(Model1.summary())
    #

    return Model(inputs=model_net.input, outputs=x)


# In[ ]:


xmin=[]
xmax=[]
ymin=[]
ymax=[]
new_df=pd.DataFrame()


# In[21]:


from tqdm import tqdm
def main():
    model = create_model([640,480], ALPHA)
    model.load_weights(WEIGHTS_FILE)
    #model = create_model(IMAGE_SIZE, ALPHA)
    #model.load_weights(WEIGHTS_FILE)

    ious = []
 
    
   # lbl2 = pd.read_csv('custom_labeled.csv')
   # lbl=pd.concat([lbl1, lbl2[1:]], ignore_index=True
   # labels=lbl[1000:]

    for image_name in tqdm(lab['image_name'].values):

        image_path=os.path.join('input/images/', image_name)
        image_ = cv2.imread(image_path)
        
        image = cv2.resize(image_, (640, 480))
        region = model.predict(np.array([image_]))
        #
        xmin.append(region[0][0])
        ymin.append(region[0][1])
        xmax.append(region[0][2])
        ymax.append(region[0][3])
        #print(xmin,ymin,xmax,ymax)
        #cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),3)
        #cv2.imshow('out',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        #cv2.rectangle(image,(region[0],region[1]),(region[1],region[3]),(255,0,0),15)
        #print(region)
        #plt.imshow(image)
        #plt.show()
'''

        box2 = [xmin, ymin, xmax, ymax]
        iou_ = iou(box1, box2)
        ious.append(iou_)
        if DEBUG:
            if iou_<0.7:
                print("IoU for {} is {}".format(image_name, iou_))
            
                cv2.rectangle(image_, (int(xmin/IMAGE_SIZE*width), int(ymin/IMAGE_SIZE*height)), (int(xmax/IMAGE_SIZE*width), int(ymax/IMAGE_SIZE*height)), (0, 0, 255), 1)
                cv2.rectangle(image_, (int(box1[0]/IMAGE_SIZE*width), int(box1[1]/IMAGE_SIZE*height)), (int(box1[2]/IMAGE_SIZE*width), int(box1[3]/IMAGE_SIZE*height)), (0, 255, 0), 1)
            
                cv2.imshow("image", image_)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        np.set_printoptions(suppress=True)
        print("\nAvg IoU: {}".format(np.mean(ious)))
        print("Highest IoU: {}".format(np.max(ious)))
        print("Lowest IoU: {}".format(np.min(ious)))

'''
    

if __name__ == "__main__":
    main()


# In[ ]:


new_df=pd.DataFrame()


# In[ ]:


new_df['image_name']=lab['image_name']
new_df['x1']=xmin
new_df['x2']=xmax
new_df['y1']=ymin
new_df['y2']=ymax


# In[24]:


new_df.head()


# In[ ]:


new_df.to_csv('Resnet_15E_14.csv', index=False)


# In[ ]:


from google.colab import files
files.download('Resnet_15E_14.csv')


# In[ ]:


#get_ipython().system('ls')


# In[ ]:


'''
from google.colab import files
from google.colab import auth
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
filename = "Mobile-net.h5"
def save_file_to_drive(name, path):
    file_metadata = {
    'name': name,
    'mimeType': 'application/octet-stream'
    }

    media = MediaFileUpload(path, 
                  mimetype='application/octet-stream',
                  resumable=True)

    created = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print('File ID: {}'.format(created.get('id')))

    return created


extension_zip = ""

zip_file = filename + extension_zip

# !rm -rf $zip_file
#!zip -r $zip_file {folders_or_files_to_save} # FOLDERS TO SAVE INTO ZIP FILE

auth.authenticate_user()
drive_service = build('drive', 'v3')

destination_name = zip_file
path_to_file = zip_file
save_file_to_drive(destination_name, path_to_file)


# In[ ]:


get_ipython().system('ls tb1')


'''