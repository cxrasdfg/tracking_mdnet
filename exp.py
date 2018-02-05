#

class A(object):

    def __init__(self):
        self.t=1

    def func(self):
        print("A:func")
        self._test_func()

    def _test_func(self):
        print("A:test func")

class B(A):
    def __init__(self):
        super(B,self).__init__()

    def _test_func(self):
        print('B:test func')


# B().func()

from Utility import *
def LoadTestModel(input_shape=(107,107,3)):
    # save mem
    save_video_mem()
    # input
    inputs = Input(shape=input_shape)

    # conv1
    x = Conv2D(96, (7, 7), strides=2, activation='relu', use_bias=True, trainable=False)(inputs)
    x = MaxPooling2D((3, 3), strides=2)(x)

    # conv2
    x = Conv2D(256, (5, 5), strides=2, activation='relu', use_bias=True, trainable=False)(x)
    x = MaxPooling2D((3, 3), strides=2)(x)

    # conv3
    x = Conv2D(512, (3, 3), strides=1, activation='relu', use_bias=True, trainable=False)(x)
    x = Flatten()(x)

    ConvModel=Model(inputs=inputs,outputs=x,name='conv_model')

    fc_inputs=Input(shape=(ConvModel.output_shape[1],))
    # fc4
    x = Dense(512, activation='relu')(fc_inputs)

    # fc5
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)

    # fc6-k
    x = Dropout(0.5)(x)
    x = Dense(2)(x)
    x_softmax=Activation(activation='softmax')(x)

    SoftMaxModel = Model(inputs=fc_inputs, outputs=x_softmax)
    ScoreModel=Model(inputs=fc_inputs,outputs=x)

    plot_model(ConvModel,'ConvModel.png',show_shapes=True)
    plot_model(SoftMaxModel,'SoftMaxModel.png',show_shapes=True)
    plot_model(ScoreModel,'ScoreModel.png',show_shapes=True)

    ConvModel.compile(optimizer='sgd',loss='mse')
    SoftMaxModel.compile(optimizer='sgd',loss='mse')
    ScoreModel.compile(optimizer='sgd',loss='mse')

    # test_x=np.random.randn(10,107,107,3)
    # test_y=np.random.randn(10,2)
    # conv_y=ConvModel.predict(test_x)

    # because the shared layers, weight changes of SoftMaxModel will lead to
    # the weights changes of the ScoreModels, they share the same weight

    # SoftMaxModel.fit(conv_y,test_y,epochs=1000)
    # scores=ScoreModel.predict(conv_y)
    # print('')

    return ConvModel, SoftMaxModel, ScoreModel

if __name__ == '__main__':
    LoadTestModel()
