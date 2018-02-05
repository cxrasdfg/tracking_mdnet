#
from Utility import *
import tensorflow as tf

# square error function
def MyLossFunc(y_true, y_pred):
    temp=K.square(y_true-y_pred)
    temp=K.sum(temp)
    return temp


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


if __name__ == '__main__':
    print('Test linear model')

    save_video_mem() # save the resource
    # prepare the data
    x=np.array([[1,2,3,4,5,6,7,8]]).T
    func=lambda param:3*param+10
    y=func(x)

    use_functional=True
    # 线性模型 Y=wX+b
    if use_functional:
        # 使用函数式模型
        inputs=Input(shape=(1,))
        linear_model=Dense(1,use_bias=True,trainable=True)(inputs)
        linear_model=Dense(1,use_bias=True,trainable=True)(linear_model)
        linear_model=Dense(1,use_bias=True,trainable=True)(linear_model)
        model=Model(inputs=inputs,outputs=linear_model)
    else:
        # 使用序贯模型
        model=Sequential()
        model.add(Dense(1,input_shape=(1,),use_bias=True))

    # opt=adam(lr=10)
    opt=MyOpt([0.0001,0.1,0.0001])
    model.compile(optimizer=opt,loss=MyLossFunc)

    model.fit(x,y,epochs=1000)

    test_x=np.array([[2,5,8,10]]).T

    # 查看权重
    # 注意: 序贯模型要比函数式模型少一层,
    #      所以实际上,函数式模型的层数从1开始才正确.
    # temp= model.layers[0].get_weights()

    gdt=func(test_x)
    print('ground truth:',gdt.T)
    print('predicted:',model.predict(test_x).T)
    model.fit(x,y,epochs=1)

