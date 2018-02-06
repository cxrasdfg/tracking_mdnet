#
from TrainModel import LoadTestData,GenSamples,GetTrainedWeight
from Utility import *
from bbreg import BBRegressor
from tracking_data import _IMG_SIZE,ExtractRegions
from matplotlib import  pyplot as plt,patches

class MDNet(object):
    def __init__(self,input_shape=(_IMG_SIZE,_IMG_SIZE,3)):
        # set variables
        self._seq_home='./dataset/TEST/'
        self._bbr=BBRegressor(_IMG_SIZE)

        # initialize the model
        models=self.CreateModels(input_shape)

        self._convModel = models[0]
        self._softmaxModel = models[1]
        self._scoreModel = models[2]
        self._t_frame=0

        self._enable_bbr=False  # switch of the bbr regressor

        small_test=False
        if small_test:
            self._first_update_iter = 1
            self._update_iter = 1
        else:
            self._first_update_iter = 30
            self._update_iter = 10

    def CreateModels(self,input_shape):
        """create models"""
        # save the mem
        save_video_mem()
        # input layer

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

        ConvModel = Model(inputs=inputs, outputs=x, name='conv_model')

        fc_inputs = Input(shape=(ConvModel.output_shape[1],))
        # fc4
        x = Dense(512, activation='relu')(fc_inputs)

        # fc5
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)

        # fc6-k
        x = Dropout(0.5)(x)
        x = Dense(2)(x)
        x_softmax = Activation(activation='softmax')(x)

        SoftMaxModel = Model(inputs=fc_inputs, outputs=x_softmax, name='SoftmaxModel')
        ScoreModel = Model(inputs=fc_inputs, outputs=x, name='ScoreModel')

        plot_model(ConvModel, 'ConvModel.png', show_shapes=True)
        plot_model(SoftMaxModel, 'SoftMaxModel.png', show_shapes=True)
        plot_model(ScoreModel, 'ScoreModel.png', show_shapes=True)

        ConvModel.compile(optimizer='sgd', loss='mse')
        SoftMaxModel.compile(optimizer='sgd', loss='binary_crossentropy')
        ScoreModel.compile(optimizer='sgd', loss='binary_crossentropy')

        # test_x=np.random.randn(10,107,107,3)
        # test_y=np.random.randn(10,2)
        # conv_y=ConvModel.predict(test_x)

        # because the shared layers, weight changes of SoftMaxModel will lead to
        # the weights changes of the ScoreModels, they share the same weight

        # SoftMaxModel.fit(conv_y,test_y,epochs=1000)
        # scores=ScoreModel.predict(conv_y)
        # print('')

        # set weight
        w = GetTrainedWeight()

        ConvModel.set_weights(w[0:6])
        SoftMaxModel.set_weights(w[6:12])

        self._convModel = ConvModel
        self._softmaxModel = SoftMaxModel
        self._scoreModel = ScoreModel

        return ConvModel,SoftMaxModel,ScoreModel

    def GetFeatures(self,img_path, bb_samples):
        regions=ExtractRegions(img_path,bb_samples)
        return self._convModel.predict(regions,batch_size=128)

    def GetScores(self,features):
        return self._scoreModel.predict(features,batch_size=128)

    def Update(self,SPos,SNeg,iter_times):
        """update the weights in fc1 fc2 fc3 """
        #Spos is a data structure likes [[img_path,bb_samples],...]

        # SPos=[[i[0],np.array([j])] for i in SPos for j in i[1]]
        # SNeg=[[i[0],np.array([j])] for i in SNeg for j in i[1]]

        PosRegions=np.empty([0,107,107,3])
        NegRegions=np.empty([0,107,107,3])

        for i in SPos:
            PosRegions=np.concatenate([PosRegions,ExtractRegions(i[0],i[1])])

        for i in SNeg:
            NegRegions=np.concatenate([NegRegions,ExtractRegions(i[0],i[1])])


        pos_idx=np.empty(0,dtype=np.int64)
        neg_idx=np.empty(0,dtype=np.int64)

        # generate the indices
        while len(pos_idx)< 32*iter_times:
            pos_idx=np.concatenate([pos_idx,np.random.permutation(len(SPos))])

        while len(neg_idx)< 1024*iter_times:
            neg_idx=np.concatenate([neg_idx,np.random.permutation(len(SNeg))])

        pos_idx=np.array(pos_idx)
        neg_idx=np.array(neg_idx)

        for i in range(iter_times):
            # bb_pos=[SPos[j] for j in pos_idx[i*32:i*32+32]]
            # bb_neg_cand=[SNeg[j] for j in neg_idx[i*1024:i*1024+1024]]

            pos_regions=PosRegions[pos_idx[i*32:i*32+32]]
            neg_can_regions=NegRegions[neg_idx[i*1024:i*1024+1024]]

            # hard mini batch mining
            # neg_can_regions=np.array([ExtractRegions(i[0],i[1])[0] for i in bb_neg_cand])
            neg_cand_features=self._convModel.predict(neg_can_regions,batch_size=128)

            neg_can_score=self._scoreModel.predict(neg_cand_features,batch_size=128)
            neg_features=neg_cand_features[neg_can_score[:,1].argsort()[-1:-97:-1]]

            # pos_regions=np.array([ExtractRegions(i[0],i[1])[0] for i in bb_pos])
            pos_features=self._convModel.predict(pos_regions,batch_size=32)

            # prepare labels
            labels=[[1,0] for i in range(len(pos_features))]
            labels+=[[0,1] for i in range(len(neg_features))]
            labels=np.array(labels)

            # update the model
            loss=self._softmaxModel.fit(np.concatenate([pos_features,neg_features]),labels,batch_size=128,verbose=False)
            print('Frame {}, update the net, iteration:{}, loss:{}'.format(self._t_frame,i,loss.history['loss']))


    def Tracking(self,seq_name):
        # assume given the first bounding box
        data = LoadTestData(seq_name)

        LongTermQueuePlus=[]
        ShortTermQueuePlus=[]
        ShortTermQueueMinus=[]
        bb_last=None
        self._t_frame=0

        for img_name,gt in zip(data[0],data[1]):
            img_path = self._seq_home + seq_name + '/img/' + img_name
            if self._t_frame==0:
                # randomly initialize the last layer...
                # give up...

                # train the bounding box regressor
                bbr_bb_samples=GenSamples(img_path,gt,1000,0,[0.7,0],region=False)[0]
                bbr_fea=self.GetFeatures(img_path,bbr_bb_samples)
                self._bbr.train(bbr_fea,bbr_bb_samples,gt)

                # generate positive samples S1+ and negative samples S1-
                [s1_plus,s1_minus]=GenSamples(img_path, gt,500,5000,[0.7,0.3],region=False,neg_trans=1)

                # add to long term queue and short term queue
                s1_plus=[img_path,s1_plus]
                s1_minus=[img_path,s1_minus]

                LongTermQueuePlus.append(s1_plus)
                ShortTermQueuePlus.append(s1_plus)
                ShortTermQueueMinus.append(s1_minus)

                # update the network
                self.Update(ShortTermQueuePlus,ShortTermQueueMinus,self._first_update_iter)

                # update bb_last
                bb_last=gt
            else:
                [pos_bb, neg_bb]=GenSamples(img_path,bb_last,50,200,[0.7,0.3],region=False,neg_trans=1)
                pos_regions=ExtractRegions(img_path,pos_bb)
                pos_features=self._convModel.predict(pos_regions,batch_size=50)
                scores= self._scoreModel.predict(pos_features, batch_size=50)
                idx=scores[:,0].argsort()[-1:-5:-1]

                print('Frame {}, scores:{}'.format(self._t_frame, scores[idx,0].mean()))
                if scores[idx,0].mean()>0.5:
                    pos_bb_cand = pos_bb[idx]
                    pos_features_cand = pos_features[idx]

                    if self._enable_bbr:
                        bb_last = self._bbr.predict(pos_features_cand, pos_bb_cand)
                        bb_last = bb_last.mean(0)
                    else:
                        bb_last = pos_bb_cand.mean(0)
                    LongTermQueuePlus.append([img_path,pos_bb])
                    ShortTermQueuePlus.append([img_path,pos_bb])
                    ShortTermQueueMinus.append([img_path,neg_bb])

                    if len(LongTermQueuePlus)>100:
                        del LongTermQueuePlus[0]

                    if len(ShortTermQueuePlus)>20:
                        del ShortTermQueuePlus[0]

                    if len(ShortTermQueueMinus)>20:
                        del ShortTermQueueMinus[0]

                    if divmod(self._t_frame,10)[1] ==0:
                        self.Update(LongTermQueuePlus,ShortTermQueueMinus,self._update_iter)
                else:
                    self.Update(ShortTermQueuePlus,ShortTermQueueMinus,self._update_iter)
                    bb_last = pos_bb[scores[:, 0].argsort()[-1]]

            plt.close()
            im = plt.imread(img_path)
            plt.imshow(im)
            rect = patches.Rectangle(bb_last[0:2], bb_last[2], bb_last[3], linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

            plt.show()
            self._t_frame+=1

if __name__ == '__main__':
    print('Tracking...')
    seq_name='DragonBaby'
    MDNet().Tracking(seq_name)