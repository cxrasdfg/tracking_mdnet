#
from Utility import *
from tracking_data import *
from tracking_data import  _IMG_SIZE
from lrn import LRN2D


def LoadVggWeight():
    mat_layers=loadmat('./models/imagenet-vgg-m.mat')
    mat_layers=list(mat_layers['layers'])[0]
    w=[]
    for i in range(3):
        li_w,li_bias=mat_layers[4*i]['weights'].item()[0]
        w.append([li_w,li_bias[:,0]])

    return w


class MyOpt(sgd):
    def __init__(self,lr_list,**kwargs):
        super(MyOpt,self).__init__(**kwargs)
        self.lr_list=lr_list

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments

        assert len(params)==len(self.lr_list)

        for p, g, m, lr in zip(params, grads, moments,self.lr_list):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates


def LoadModel(seq_list,input_shape=(_IMG_SIZE,_IMG_SIZE,3),trainable=True):
    # input
    inputs=Input(shape=input_shape)

    # conv1
    x=Conv2D(96,(7,7),strides=2,activation='relu',use_bias=True, trainable=trainable)(inputs)
    # add lrn
    x=LRN2D()(x)
    x=MaxPooling2D((3,3),strides=2)(x)

    # conv2
    x=Conv2D(256,(5,5),strides=2,activation='relu',use_bias=True,trainable=trainable)(x)
    # add lrn
    x=LRN2D()(x)
    x=MaxPooling2D((3,3),strides=2)(x)

    # conv3
    x=Conv2D(512,(3,3),strides=1,activation='relu',use_bias=True,trainable=trainable)(x)
    x=Flatten()(x)

    # fc4
    x=Dropout(0.5)(x)
    x=Dense(512,activation='relu')(x)

    # fc5
    x=Dropout(0.5)(x)
    x=Dense(512,activation='relu')(x)

    # fc6-k
    x=Dropout(0.5)(x)
    myMDnet={key:Model(inputs=inputs,outputs=Dense(2,activation='softmax')(x)) for key in seq_list}

    w = LoadVggWeight()

    for _,model in myMDnet.items():
        for i in range(3):
            # 其实共享的层只需要设置一次就可以了
            model.layers[3 * i + 1].set_weights(w[i])

        # opt = MyOpt(lr_list=[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.001]
        #             ,clipnorm=10,decay=0.0005,momentum=0.9)

        model.compile(optimizer=sgd(lr=0.0001,clipnorm=10,decay=0.0005,momentum=0.9), loss=BinaryLoss)
    return myMDnet


def Train(iter_times=100000):
    # model.compile(optimizer='adam',loss='binary_crossentropy')

    # show the model
    # plot_model(model,to_file='./model.png',show_shapes=True)

    t_data=LoadTrainData()
    save_video_mem()
    model = LoadModel(seq_list=list(t_data.keys()))

    # fc_br={i:model.layers[-1].get_weights() for i in t_data.keys()}

    for _iter in range(iter_times):
        keys=np.array(list(t_data.keys()))
        np.random.shuffle(keys)
        for k in keys:
            v=t_data[k]
            [pos,neg]=GenTrainingSamples(k,v[0],v[1])
            x=np.concatenate([pos,neg])
            # prepare the label
            labels=[[1,0] for i in range(len(pos))]
            labels+=[[0,1] for i in range(len(neg))]
            y=np.array(labels)

            # set the weights
            # model.layers[-1].set_weights(fc_br[k])
            loss=model[k].fit(x,y,verbose=False)

            predict_y = model[k].predict(x)

            print('Iter:{}, Loss:{}, precision:{}'.format(
                _iter,
                loss.history['loss'],
                Precision(predict_y,y))
            )
            # get the weights
            # fc_br[k]=model.layers[-1].get_weights()

        if _iter % 10 ==0:
            print('Iter:{}, save the weights'.format(_iter))
            model[k].save_weights('weight_mdnet')

def GetTrainedWeight():
    model=list(LoadModel(seq_list=['test']).values())[0]
    model.load_weights('weight_mdnet')
    # model.set_weights(np.load('./weights_10k.npy').tolist())
    return model.get_weights()


def LoadVgg16Version():
    input_shape = Input((107, 107, 3))
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_shape)
    shared = Flatten()(base_model.output)
    fc1 = Dense(512, activation='relu')(shared)
    fc2 = Dense(512, activation='relu')(fc1)

    output = Dense(2, activation='softmax')(fc2)

    model = Model(inputs=base_model.inputs, outputs=output)

    model.compile(optimizer='adam', loss=BinaryLoss)

    return model

if __name__ == '__main__':
    print('Train the Model...')
    test=GetTrainedWeight()
    Train()