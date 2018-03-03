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



def BinaryLoss(y_true,y_pred):
    y_pred=K.maximum(y_pred,1e-6)
    loss=-K.log(y_pred)
    loss=y_true*loss
    loss=K.sum(loss)

    return loss

def _BinaryLoss(y_true,y_pred):
    y_true=K.get_value(y_true)
    y_true=y_true[:,0]

    y_pred=K.get_value(y_pred)
    # find the pos index
    pos_idx=np.where(y_true==1)[0]

    # find the neg index
    neg_idx=np.where(y_true==0)[0]

    # get pos score
    pos_score=y_pred[pos_idx]
    pos_score=pos_score[:,0]
    pos_score=K.variable(pos_score)

    # get neg score
    neg_score=y_pred[neg_idx]
    neg_score=neg_score[:,1]
    neg_score=K.variable(neg_score)

    # get the loss
    pos_loss=-K.log(pos_score)
    neg_loss=-K.log(neg_score)

    loss=K.sum(pos_loss)+K.sum(neg_loss)

    return loss


def _BinaryLossNp(pos,neg):

    pos_loss=-np.log(pos)[:,0]
    neg_loss=-np.log(neg)[:,1]

    loss=pos_loss.sum()+neg_loss.sum()

    return loss


if __name__ == '__main__':
    # test the keras implementation
    data=np.array([[0.1,0.2],[0.7,0.1],[0.2,0.1],[0.6,0.9],[0.12,0.55]])
    label=np.array([[1,0],[1,0],[0,1],[0,1],[0,1]])

    print('keras:',K.get_value(_BinaryLoss(K.variable(label),K.variable(data))))

    # test the torch implementation
    from torch.autograd import  Variable
    print('torch:',_BinaryLossNp(data[0:2],data[2:5]))