#
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize as imresize
from Utility import IouRate
_IMG_SIZE=107

def LoadTrainData():
    """Load the vot training data"""
    # load the sequences indices

    seq_list=np.loadtxt('./dataset/train_seq_list.txt',dtype=str)
    seq_home='./dataset/'

    # test the bb
    show_bb = False

    res={}
    for i,k in enumerate(seq_list):
        img_list=[]
        for tmp_file in os.listdir(seq_home+k):
            if tmp_file.endswith('.jpg'):
                img_list.append(tmp_file)

        # ????...
        img_list=sorted(img_list)
        gt=np.loadtxt(seq_home+k+'/groundtruth.txt',delimiter=',')
        if gt.shape[1]==8:
            # allocate space for bb
            bb=[[0,0,0,0]]*gt.shape[0]
            bb=np.array(bb)

            for gt_index,row in enumerate(gt):
                min_x=min(row[0],row[2],row[4],row[6])
                max_x=max(row[0],row[2],row[4],row[6])
                min_y=min(row[1],row[3],row[5],row[7])
                max_y=max(row[1],row[3],row[5],row[7])

                bb[gt_index]=[min_x,min_y,max_x-min_x,max_y-min_y]
            gt=bb

        if show_bb:
            for img_name,bb in zip(img_list,gt):
                im=plt.imread(seq_home+k+'/'+img_name)
                fig,ax=plt.subplots(1)
                ax.imshow(im)
                rect=patches.Rectangle(bb[0:2], bb[2],bb[3],linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)

                plt.show()

        res[k]=[img_list,gt]
    return res


def LoadTestData(seq_name):
    seq_home='./dataset/TEST/'
    img_seq_path=seq_home+seq_name+'/img'
    img_list = []
    for tmp_file in os.listdir(img_seq_path):
        if tmp_file.endswith('.jpg'):
            img_list.append(tmp_file)

    img_list=sorted(img_list)
    gt=np.loadtxt(seq_home+seq_name+'/groundtruth.txt',delimiter=',')

    return [img_list,gt]


def _ShowSequences(seq_name,_t='test'):
    if _t == 'test':
        seq_home='./dataset/TEST/'
        k=seq_name+'/img'
        seq=LoadTestData(seq_name)
    elif _t=='train':
        seq_home = './dataset/'
        k=seq_name
        seq=LoadTrainData()[k]
    # begin to draw ..
    fig, ax = plt.subplots(1)
    plt.ion()
    for img_name, bb in zip(seq[0],seq[1]):

        im = plt.imread(seq_home + k + '/' + img_name)

        plt.cla()
        # plt.clf()
        ax.imshow(im)
        rect = patches.Rectangle(bb[0:2], bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.pause(0.016)
        # rect.remove()


def show_rect(bb):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(
        patches.Rectangle(
            bb[0:2],  # (x,y)
            bb[2],  # width
            bb[3],  # height
        )
    )


def ExtractRegions(img_path,bb_samples):
    import cv2 as cv
    # cv.cvtColor()
    im = plt.imread(img_path)
    # im=im.astype(np.float64)
    # im *= 1. / 255
    res=[]
    for left,top,w,h in bb_samples:
        chip=im[int(top):int(top+h)+1,int(left):int(left+w)+1]
        chip=cv.resize(chip,(_IMG_SIZE,_IMG_SIZE),interpolation=cv.INTER_CUBIC)
        chip=cv.cvtColor(chip,cv.COLOR_BGR2RGB)
        chip=chip.astype(np.int8)
        # chip*=1./255
        res.append(chip)

    return np.array(res)


def GenTrainingSamples(seq_name,img_list,gt):
    index=np.random.permutation(len(gt))
    seq_home='./dataset/'

    pos_reg_list=np.empty([0,_IMG_SIZE,_IMG_SIZE,3])
    neg_reg_list=np.empty([0,_IMG_SIZE,_IMG_SIZE,3])

    for count,i in enumerate(index):
        if count>=8:
            break
        img_path=seq_home+seq_name+'/'+img_list[i]
        gt_i=gt[i]

        # generate 4 pos and 12 neg samples for each frame
        num_pos=4
        num_neg=12

        [pos_reg,neg_reg]=GenSamples(img_path,gt_i,num_pos,num_neg,[0.7,0.5])

        # extract regions
        try:
            pos_reg_list=np.concatenate([pos_reg_list,pos_reg])

            neg_reg_list=np.concatenate([neg_reg_list,neg_reg])
        except ValueError as ve:
            print('Value Error!:',ve)
            continue

    return [pos_reg_list,neg_reg_list]


def GenSamples(img_path,gt_i,num_pos,num_neg,iou_divs,region=True,
               pos_trans=0.3,pos_scale=1.05,neg_trans=0.3,neg_scale=1.05):
    pos_samples = np.empty([0, 4])
    neg_samples = np.empty([0, 4])

    # generate the positive samples
    power = 1
    while len(pos_samples) < num_pos and power <= 6:
        pos_cand = _GenSamplesHelper(gt_i, 'gaussian', 2 ** power * num_pos,pos_trans,pos_scale)
        IOU = IouRate(pos_cand, np.tile(gt_i, [len(pos_cand), 1]))
        for k, iou in enumerate(IOU):
            if iou >= iou_divs[0]:
                pos_samples = np.concatenate([pos_samples, [pos_cand[k]]])
        power += 1

    # need limited samples
    pos_samples = pos_samples[:min(len(pos_samples), num_pos)]

    # generate the negative samples
    power = 1
    while len(neg_samples) < num_neg and power <= 6:
        neg_cand = _GenSamplesHelper(gt_i, 'uniform', 2 ** power * num_neg,neg_trans,neg_scale)

        IOU = IouRate(neg_cand, np.tile(gt_i, [len(neg_cand), 1]))
        for k, iou in enumerate(IOU):
            if iou <= iou_divs[1]:
                neg_samples = np.concatenate([neg_samples, [neg_cand[k]]])
        power += 1

    # need limited samples
    neg_samples = neg_samples[:min(len(neg_samples), num_neg)]

    if region:
        return [ExtractRegions(img_path,pos_samples), ExtractRegions(img_path,neg_samples)]
    else:
        return [pos_samples,neg_samples]

# helper of sample generator
# n is the required number of samples
# bb is the bounding box
def _GenSamplesHelper(bb,_t,n,trans=0.3,scale=1.05):
    samples=np.tile(bb,[n,1])

    samples=samples.astype(np.float64)

    # change samples to [center_x,center_y,w,h]
    samples[:,:2]+=samples[:,2:]/2

    r = np.mean(bb[2:])
    if _t=='gaussian':
        samples[:,:2]+=trans*r*np.random.randn(n,2)  # variance is 0.09r^2
        samples[:,2:]*=scale**(0.5*np.random.randn(n,2))  # variance is 0.25

    elif _t=='uniform':
        samples[:,:2]+=trans*r*(np.random.rand(n,2)*2-1) # uniform distribution
        samples[:,2:]*=scale**(np.random.rand(n,2)*2-1)

    # change samples to [left_x,top_y,w,h]
    samples[:,:2]-=samples[:,2:]/2
    samples[:,:2]=np.maximum(samples[:,:2],0)

    return samples


if __name__ == '__main__':
    "main"
    # seq_name='DragonBaby'
    # _ShowSequences(seq_name)
    _ShowSequences('vot2014/bicycle', 'train')
    from itertools import permutations
    # for i in permutations([1,2,3]):
    #     print(i)

    # temp=LoadTrainData()
    # for k,v in temp.items():
    #     [pos,neg]=GenTrainingSamples(k,v[0],v[1])

