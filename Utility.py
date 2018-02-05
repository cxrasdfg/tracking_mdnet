from keras.layers import Conv2D,Dense,\
    activations,Input,Flatten,MaxPooling2D,Dropout,\
    Conv3D,MaxPooling3D,Activation
from keras.optimizers import sgd,adam
import keras.backend as K
from keras.models import Model,Sequential
from keras.losses import cosine_proximity
from keras.applications.vgg16 import VGG16
from keras.utils import plot_model


from scipy.spatial.distance import cdist
import numpy as np
import gc
from scipy.io import loadmat

def save_video_mem(video_card=0):
    import os
    import keras.backend as k
    import  tensorflow as tf
    # 指定第一块GPU可用
    os.environ["CUDA_VISIBLE_DEVICES"] = str(video_card)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)

    k.tensorflow_backend.set_session(sess)


def IouRate(bb,gt):
    """
    :param bb: N x 4 estimated data
    :param gt: N x 4 ground truth data
    :return:IOU for each pair samples
    """

    Left=np.maximum(bb[:,0],gt[:,0])

    Right=np.minimum(bb[:,0]+bb[:,2],gt[:,0]+gt[:,2])

    Top=np.maximum(bb[:,1],gt[:,1])

    Bottom=np.minimum(bb[:,1]+bb[:,3],gt[:,1]+gt[:,3])

    Inter=np.maximum(0,Right-Left)*np.maximum(0,Bottom-Top)
    Union=bb[:,2]*bb[:,3]+gt[:,2]*gt[:,3]-Inter

    return Inter/Union

