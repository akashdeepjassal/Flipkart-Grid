
# coding: utf-8

# In[ ]:


#from google.colab import files
#!rm -rf train.py
import numpy as np
import os
list1=np.array(os.listdir('input/images'))
list1.shape


# In[ ]:


#!cp gdrive/'My Drive'/Flipkart/Resnet5thEpoch.h5 /content/
#get_ipython().system('ls')


# In[ ]:


#files.upload()


# In[ ]:


'''

N
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



'''


# In[ ]:


import math
import pandas as pd
import cv2
import os
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.xception import Xception
from keras.applications.xception import Xception
from keras import regularizers
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping,  ReduceLROnPlateau

from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *
from keras.utils import Sequence
from keras.callbacks import TensorBoard
from keras import layers
from keras.activations import relu


# In[ ]:


ALPHA = 1
IMAGE_SIZE = [480,640]
BATCH_SIZE = 8
EPOCHS = 5
PATIENCE = 3000

IMAGE_DIR = 'input/images/'
VALIDATION_DATASET_SIZE = 2000


# In[ ]:


class DataSequence(Sequence):

    def __load_images(self, dataset):
        out = []

        for file_name in dataset:
            im = cv2.resize(cv2.imread(file_name), (self.image_size[0], self.image_size[1]))
            out.append(im)

        return np.array(out)

    def __init__(self, csv_file, image_size, dataset_type = 'train', batch_size=BATCH_SIZE):
        self.csv_file = csv_file
        lbl1 = pd.read_csv(self.csv_file)
        lbl2 = pd.read_csv('input/training.csv')
        lbl=pd.concat([lbl1, lbl2[1:]], ignore_index=True)
        self.datalen=len(lbl1)
        if dataset_type=='train':
            labels=lbl1[VALIDATION_DATASET_SIZE:]
        else:
            labels=lbl1[:VALIDATION_DATASET_SIZE]
        self.y = np.zeros((len(labels), 4))
        self.x = []
        self.image_size = image_size
        index = 0
        for image_name, x1, x2, y1, y2 in labels.values:
            image_path=os.path.join(IMAGE_DIR, image_name)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            self.y[index][0] = int(x1)
            self.y[index][1] = int(y1)
            self.y[index][2] = int(x2)
            self.y[index][3] = int(y2)
            self.x.append(image_path)
            index+=1
        self.batch_size = batch_size
 
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = self.__load_images(batch_x).astype('float32')

        return images, batch_y


# In[ ]:


def iou_metric(y_true, y_pred):
    # iou as metric for bounding box regression
    # input must be as [x1, y1, x2, y2]
    
    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0] + 1) * K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1)
    
    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] + 1) * K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    overlap_1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection
    
    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
  #  iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    return iou    

def iou_metric_mean(y_true, y_pred):
 
    return iou_metric(y_true, y_pred)


def iou_loss(y_true, y_pred):

    # loss for the iou value
    iou_loss = -K.log(iou_metric_mean(y_true, y_pred))

    return iou_loss


# In[ ]:


def iou_metric_mean(y_true, y_pred):
 
    return iou_metric(y_true, y_pred)


def iou_loss(y_true, y_pred):

    # loss for the iou value
    iou_loss = -K.log(iou_metric_mean(y_true, y_pred))

    return iou_loss


# In[ ]:


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)


# In[ ]:



def create_model(size, alpha):
    model_net = MobileNet(input_shape=(size[1], size[0], 3), include_top=False, weights = None, alpha=alpha)
    #model_net = ResNet50(input_shape=(size[1], size[0], 3), include_top=False, weights = None)
    #mode_net
    #model_net =Xception(input_shape=(size[1], size[0], 3), include_top=False, weights = None)
   # for i in range(len(model_net.layers) - 13):
    #    model_net.layers[i].trainable=False
    x = _depthwise_conv_block(model_net.layers[-1].output, 1024/4, alpha, 1, block_id=14)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(4, kernel_size=(1, 1), padding="same")(model_net.layers[-1].output)
    x = GlobalAveragePooling2D()(x)
    #x = Reshape((4, ))(x)
    Model1=Model(inputs=model_net.input, outputs=x)
    
    print(Model1.summary())
    #

    return Model(inputs=model_net.input, outputs=x)


# In[34]:


get_ipython().system('mkdir weights')
def train(model, epochs, image_size):
    train_datagen = DataSequence("input/training.csv", image_size,"train")
    validation_datagen = DataSequence("input/training.csv", image_size,"val")
    dir_weights='weights'
    model.compile(loss="mse", optimizer="adam",  metrics=["accuracy",iou_metric_mean])
    checkpoint = ModelCheckpoint("weights/Mob--{val_loss:.2f}--{val_iou_metric_mean:.2f}.h5", monitor="val_iou_metric_mean", verbose=1, 
                                 save_weights_only=True, mode="max", period=1)#save_best_only=True,
    stop = EarlyStopping(monitor="val_iou_metric_mean", patience=PATIENCE, mode="auto")
    tensorboard = TensorBoard(log_dir='./tb1', histogram_freq=0,
                          write_graph=True, write_images=False)
 #   print(model.metrics_names)
  #  model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
  #  checkpoint = ModelCheckpoint("model-{val_acc:.2f}.h5", monitor="val_acc", verbose=1, save_best_only=True,
   #                              save_weights_only=True, mode="auto", period=1)
   # stop = EarlyStopping(monitor="val_acc", patience=PATIENCE, mode="auto")

    model.fit_generator(train_datagen, steps_per_epoch=train_datagen.datalen//BATCH_SIZE, epochs=epochs, validation_data=validation_datagen,
                       validation_steps=21, callbacks=[checkpoint, stop, tensorboard])


# In[ ]:


#get_ipython().system('ls weights1/weights')


# In[ ]:


WEIGHTS_FILE = os.path.join('weights/','Res--751.66--0.82.h5')
def main():
    model = create_model(IMAGE_SIZE, ALPHA)
    model.load_weights(WEIGHTS_FILE)
    #print(model.summary())

    train(model, EPOCHS, IMAGE_SIZE)
    model.save('Res-net17E.h5')


# In[40]:


if __name__ == "__main__":
    main()


