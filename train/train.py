import numpy as np
from pycocotools.coco import COCO
import os
import math
import keras
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Reshape, Dropout, BatchNormalization, Activation, Conv2D
from keras.utils import plot_model
from random import shuffle
from gaussian import gaussian, gaussian_multi_input_mp, gaussian_multi_output
import sys
import os
import json
from tempte import test_of_model
import keras.backend as K
import tensorflow as tf

base_dir = '/home/salih/Repo/ceng_783/'
exp_dir  = '/home/salih/Repo/ceng_783/train/' + sys.argv[1] + '/'



modelname = sys.argv[1]


if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

coeff = 1
width = int(18 * coeff)
height = int(28 * coeff)
thres = 0.21
num_of_keypoints = 16 # min 0 max 16 # Best 3 -4
coco_train = COCO(os.path.join(base_dir+'annotations/person_keypoints_train2017.json'))
coco_val = COCO(os.path.join(base_dir,'annotations/person_keypoints_val2017.json'))
batch_size = 8
number_of_epoch = 1

#############################################

last_layer_activation = 'softmax'
# 56 , 36

input = Input(shape=(height, width, 17))
y = Flatten()(input)
x = Dense(128, activation='relu')(y)
x = Dropout(0.5)(x)
x = Dense(width * height * 17, activation='relu')(x)
x = keras.layers.Add()([x,y])

x = keras.layers.Activation(last_layer_activation)(x)

x = Reshape((height, width, 17))(x)

model = Model(inputs=input, outputs=x)

print model.summary()

#######################################





name_of_model = sys.argv[1]


class My_Callback(keras.callbacks.Callback):

    # epoch = 0
    # def on_train_begin(self, logs={}):
    #    return

    # def on_train_end(self, logs={}):
    #    return

    def on_epoch_begin(self, epoch, logs={}):
        # model.load(exp_dir + 'model_epoch_{}.h5'format(epoch))
        print('\n')
        print('***************************** Epoch ' + str(
            epoch - 1) + ' Finished *******************************************')
        print('\n')
        #sys.stdout.close()
        #sys.stdout = open(exp_dir + 'test.txt', 'a')

        return

    def on_epoch_end(self, epoch, logs={}):
        model.save(exp_dir + sys.argv[1] + 'epoch_{}.h5'.format(epoch))
        print 'Epoch', epoch, 'has been saved'
        test_of_model(exp_dir + sys.argv[1] + 'epoch_{}.h5'.format(epoch), coeff, modelname, exp_dir, epoch, self.model)
        print 'Epoch', epoch, 'has been tested'
        # print checkpoint.filepath
        #sys.stdout = open(exp_dir + modelname + '.txt', 'a')
        return

    # def on_batch_begin(self, batch, logs={}):
    #    return

    # def on_batch_end(self, batch, logs={}):
    #    self.losses.append(logs.get('loss'))
    #    return


def get_data(ann_data, coco):
    weights = np.zeros((height, width, 17))
    output = np.zeros((height, width, 17))

    bbox = ann_data['bbox']
    x = int(bbox[0])
    y = int(bbox[1])
    w = float(bbox[2])
    h = float(bbox[3])

    x_scale = float(width) / math.ceil(w)
    y_scale = float(height) / math.ceil(h)

    kpx = ann_data['keypoints'][0::3]
    kpy = ann_data['keypoints'][1::3]
    kpv = ann_data['keypoints'][2::3]

    for j in range(17):
        if kpv[j] > 0:
            x0 = int((kpx[j] - x) * x_scale)
            y0 = int((kpy[j] - y) * y_scale)

            if x0 >= width and y0 >= height:
                output[height - 1, width - 1, j] = 1
            elif x0 >= width:
                output[y0, width - 1, j] = 1
            elif y0 >= height:
                output[height - 1, x0, j] = 1
            elif x0 < 0 and y0 < 0:
                output[0, 0, j] = 1
            elif x0 < 0:
                output[y0, 0, j] = 1
            elif y0 < 0:
                output[0, x0, j] = 1
            else:
                output[y0, x0, j] = 1

    img_id = ann_data['image_id']
    img_data = coco.loadImgs(img_id)[0]
    ann_data = coco.loadAnns(coco.getAnnIds(img_data['id']))

    for ann in ann_data:
        kpx = ann['keypoints'][0::3]
        kpy = ann['keypoints'][1::3]
        kpv = ann['keypoints'][2::3]

        for j in range(17):
            if kpv[j] > 0:
                if (kpx[j] > bbox[0] - bbox[2] * thres and kpx[j] < bbox[0] + bbox[2] * (1 + thres)):
                    if (kpy[j] > bbox[1] - bbox[3] * thres and kpy[j] < bbox[1] + bbox[3] * (1 + thres)):
                        x0 = int((kpx[j] - x) * x_scale)
                        y0 = int((kpy[j] - y) * y_scale)

                        if x0 >= width and y0 >= height:
                            weights[height - 1, width - 1, j] = 1
                        elif x0 >= width:
                            weights[y0, width - 1, j] = 1
                        elif y0 >= height:
                            weights[height - 1, x0, j] = 1
                        elif x0 < 0 and y0 < 0:
                            weights[0, 0, j] = 1
                        elif x0 < 0:
                            weights[y0, 0, j] = 1
                        elif y0 < 0:
                            weights[0, x0, j] = 1
                        else:
                            weights[y0, x0, j] = 1

    #for t in range(17):
    #    output[:, :, t] = gaussian(output[:, :, t])
    #output  =  gaussian(output, sigma=2, mode='constant', multichannel=True)
    weights = gaussian_multi_input_mp(weights)
    output = gaussian_multi_output(output)
    return weights, output


def train_bbox_generator():
    #ann_ids = coco_train.getAnnIds()
#     anns = get_anns(coco_train)
#     while 1:
#         shuffle(anns)
#         for i in range(0, len(anns) // batch_size, batch_size):
#             X = np.zeros((batch_size, height, width, 17))
#             Y = np.zeros((batch_size, height, width, 17))
#             for j in range(batch_size):
#                 #ann_data = coco_train.loadAnns(ann_ids[i + j])[0]
#                 ann_data = anns[i+j]
#                 #if ann_data['iscrowd'] != 0:
#                 try:
#                     x, y = get_data(ann_data, coco_train)
#                 except:
#                     continue
#                 X[j, :, :, :] = x
#                 Y[j, :, :, :] = y
#             yield X, Y

    anns = get_anns(coco_train)
    while 1:
        shuffle(anns)
        for i in range(0, len(anns) // batch_size, batch_size):
            X = np.zeros((batch_size, height, width, 17))
            Y = np.zeros((batch_size, height, width, 17))
            for j in range(batch_size):
                #ann_data = coco_train.loadAnns(ann_ids[i + j])[0]
                #ann_data = anns[batch_size * i+j]
                ann_data = anns[i+j]
                try:
                    x, y = get_data(ann_data, coco_train)
                except:
                    continue
                X[j, :, :, :] = x
                Y[j, :, :, :] = y
            yield X, Y


def get_anns(coco):
    
    #:param coco: COCO instance
    #:return: anns: List of annotations that contain person with at least 6 keypoints
    
    ann_ids = coco.getAnnIds()
    anns = []
    for i in ann_ids:
        ann = coco.loadAnns(i)[0]
        if ann['iscrowd'] == 0 and ann['num_keypoints'] > num_of_keypoints:
            anns.append(ann) # ann
    sorted_list = sorted(anns, key=lambda k: k['num_keypoints'], reverse=True)
    return sorted_list
    
            
            
def val_bbox_generator():
    ann_ids = coco_val.getAnnIds()
    while 1:
        shuffle(ann_ids)
        for i in range(len(ann_ids) // batch_size):
            X = np.zeros((batch_size, height, width, 17))
            Y = np.zeros((batch_size, height, width, 17))
            for j in range(batch_size):
                ann_data = coco_val.loadAnns(ann_ids[i + j])[0]
                try:
                    x, y = get_data(ann_data, coco_val)
                except:
                    continue
                X[j, :, :, :] = x
                Y[j, :, :, :] = y
            yield X, Y


def custom_loss(y_true, y_pred):
    score = 0.0
    for i in range(17):
        score += K.categorical_crossentropy(K.flatten(y_true[:,:, :, i]), K.flatten(y_pred[:, :,:, i]))
    return score / 17.0

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


#####
# test



#model.load_weights('/home/muhammed/salih/pose-grammar/train/nineth_/nineth_epoch_0.h5', by_name=True)

adam_optimizer = keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# todo change gaussian ; l2 no activation 

####3




with open(exp_dir + sys.argv[1] + '.txt', 'a') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

#plot_model(model, to_file=exp_dir + 'test.png', show_shapes=True)

adam_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='binary_crossentropy' , optimizer=adam_optimizer)

Own_callback = My_Callback()

checkpoint = keras.callbacks.ModelCheckpoint(exp_dir + 'weights.{epoch:02d}-{val_loss:.2f}.h5', verbose=1)
csv_log = keras.callbacks.CSVLogger(exp_dir + 'training_log.csv', separator=',', append=False)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

#sys.stdout = open(exp_dir + modelname + '.txt', 'a')
print('\n')
print("Window Rate :" + str(coeff))
print("Loss: " + str(model.loss_functions))
print("Last Layer Activation :" + last_layer_activation)
print("Threshold: " + str(thres))
print("Batch Size: " + str(batch_size))
print("Epoch: " + str(number_of_epoch))
print("Gaussian: " +  "Input new, output new, latest")
print("Number of Keypoints: " + str(num_of_keypoints))
print('\n')
print("#######################################################################################")
print('\n')
#sys.stdout.close()
#sys.stdout = open(exp_dir + 'test.txt', 'a')

model.fit_generator(generator=train_bbox_generator(),
                    steps_per_epoch=len(get_anns(coco_train)) // batch_size,
                    validation_data=val_bbox_generator(),
                    validation_steps=len(coco_val.getAnnIds()) // batch_size,
                    epochs=number_of_epoch,
                    callbacks=[Own_callback],
                    # [checkpoint, csv_log, reduce_lr, Own_callback],
                    verbose=1,
                    initial_epoch=0)

# callbacks=[checkpoint, csv_log, reduce_lr],






