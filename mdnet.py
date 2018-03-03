from Utility import *

if __name__ == '__main__':
    save_video_mem()

    print('mdnet')

    input_shape=Input((107,107,3))
    base_model=VGG16(weights='imagenet',include_top=False,input_tensor=input_shape)
    shared=Flatten()(base_model.output)
    fc1=Dense(512,activation='relu')(shared)
    fc2=Dense(512,activation='relu')(fc1)

    output=Dense(2,activation='softmax')(fc2)

    model=Model(inputs=base_model.inputs,outputs=output)

    model.compile(optimizer='adam',loss='binary_crossentropy')

    plot_model(model,to_file='vgg16.png',show_shapes=True)
    gc.collect()

    