import cv2
import keras
import matplotlib
import pylab as plt
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from keras.models import load_model
import os
from random import randint
import math
from random import shuffle
from gaussian import gaussian, crop, gaussian_multi_input_mp
#from visualize import showAnns
import time
import sys
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Reshape, Dropout, BatchNormalization, Activation, Conv2D
import random

# 0 --> gt                 819  739
# 1 --> kp gt bbox own     732  739
# 2 --> kp own bbox gt     541  
# 3 --> BOTH               521  530
test = 0
test_keypoint_count = 0

base_dir = '/home/salih/Repo/ceng_783/'

#model_dir = '/home/muhammed/salih/pose-grammar/train/hope_best_/hope_best_epoch_0.h5'

def test_of_model(model_dir, coeff, modelname, exp_dir, epoch, model):   # , model
    #model = load_model(model_dir)
    if(test==1):
        # GT keypoint - our bbox
        bbox_results     = json.load(open(base_dir+ 'annotations/val2017_bbox_results.json'))

        coco = COCO(base_dir+ 'annotations/person_keypoints_val2017.json')
        img_ids = coco.getImgIds(catIds=[1])

        print len(img_ids)

        peak_results = []

        for i in img_ids:
            anns = coco.loadAnns(coco.getAnnIds(imgIds=i))
            kps  = [a['keypoints'] for a in anns]

            idx = 0

            ks = []
            for i in range(17):
                t = []
                for k in kps:
                    x = k[0::3][i]
                    y = k[1::3][i]
                    v = k[2::3][i]

                    if v > 0:
                        t.append([x,y,1,idx])
                        idx+=1
                ks.append(t)
            image_id = anns[0]['image_id']
            peaks = ks

            element = {
                'image_id' : image_id,
                'peaks'    : peaks,
                'file_name': coco.loadImgs(image_id)[0]['file_name']
            }

            peak_results.append(element)

    if test == 2:
        # GT bbox - our kp
        peak_results = json.load(open(base_dir+'annotations/peak_results.json'))
        ann = json.load(open(base_dir+ 'annotations/person_keypoints_val2017.json'))
        bbox_results = ann['annotations']

    if test == 3:
        # Both our
        keypoint_results = json.load(open(base_dir + 'annotations/val2017_keypoint_results.json'))
        peak_results     = json.load(open(base_dir + 'annotations/peak_results.json'))
        bbox_results     = json.load(open(base_dir + 'annotations/val2017_bbox_results.json'))
        print len(keypoint_results)
        print len(peak_results)
        print peak_results[0]

    if test == 0:
        # Both Ground Truth
        cocodir = base_dir + 'annotations/person_keypoints_val2017.json'
        ann = json.load(open(cocodir))
        bbox_results = ann['annotations']

        coco = COCO(cocodir)
        img_ids = coco.getImgIds(catIds=[1])

        print len(img_ids)

        peak_results = []

        for i in img_ids:
            anns = coco.loadAnns(coco.getAnnIds(imgIds=i))
            kps  = [a['keypoints'] for a in anns]

            idx = 0

            ks = []
            for i in range(17):
                t = []
                for k in kps:
                    x = k[0::3][i]
                    y = k[1::3][i]
                    v = k[2::3][i]

                    if v > 0:
                        t.append([x,y,1,idx])
                        idx+=1
                ks.append(t)
            image_id = anns[0]['image_id']
            peaks = ks

            element = {
                'image_id' : image_id,
                'peaks'    : peaks,
                'file_name': coco.loadImgs(image_id)[0]['file_name']
            }

            peak_results.append(element)

    ''''
    width = int(18*coeff)
    height = int(28*coeff)
    
        # test
    last_layer_activation = 'softmax'

    input = Input(shape=(height, width, 17))
    y = Flatten()(input)
    x = Dense(1024, activation='relu')(y)
    x = Dropout(0.5)(x)
    x = Dense(width * height * 17, activation='relu')(x)
    x = keras.layers.Add()([x,y])

    out =  []
    start = 0
    end = width*height

    for i in range(17):
        # o = Activation(last_layer_activation)(x[:,start:end])
        o = keras.layers.Lambda(lambda x: x[:, start:end])(x)
        o = Activation(last_layer_activation)(o)
        out.append(o)
        start = end
        end = start + width*height

    x = keras.layers.Concatenate()(out)
    #x = keras.layers.Activation(last_layer_activation)(x)
    #x = K.concatenate(out)

    x = Reshape((height, width, 17))(x)

    model = Model(inputs=input, outputs=x)
    '''
    #adam_optimizer = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)



    model.load_weights(model_dir, by_name=True)
    
    
    
    #model = model
    
    shuffle(peak_results)

    order = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

    in_thres = 0.21
    n = 15
    my_results = []
    image_ids = []

    #joint_priors = np.load(base_dir + 'annotations/joint_priors_normalized.npy')

    coeff = coeff
    w = int(18*coeff)
    h = int(28*coeff)

    start = time.time()

    temporary_peak_res = []
    for p in peak_results:
        peaks = p['peaks']
        if len(peaks) > 17:
            if (test == 2 or test == 3):
                peaks = [peaks[i] for i in order]
                p['peaks'] = peaks
        if (sum(1 for i in p['peaks'] if i != []) >= test_keypoint_count):
            temporary_peak_res.append(p)
    peak_results = temporary_peak_res
    
    for p in peak_results:
        idx = p['image_id']
        image_ids.append(idx)

        peaks = p['peaks'] # len 18 list that holds peaks
        if len(peaks) > 17:
            if (test == 2 or test == 3):
                peaks = [peaks[i] for i in order] # change order to COCO

        bboxes = [k['bbox'] for k in bbox_results if k['image_id'] == idx]
    #     scores = [1]*len(bboxes)
        if test == 1 or test == 3:
            scores = [k['score'] for k in bbox_results if k['image_id'] == idx]
        # print len(bboxes)

        if len(bboxes) == 0 or len(peaks) == 0:
            continue

        weights_bbox = np.zeros((len(bboxes), h, w, 4, 17))

        for joint_id, peak in enumerate(peaks):

            #prior = joint_priors[joint_id]

            for instance_id, instance in enumerate(peak):

                p_x = instance[0]
                p_y = instance[1]

                for bbox_id, b in enumerate(bboxes):

                    is_inside = p_x > b[0] - b[2] * in_thres and \
                            p_y > b[1] - b[3] * in_thres and \
                            p_x < b[0] + b[2] * (1.0+in_thres) and \
                            p_y < b[1] + b[3] * (1.0+in_thres)

                    if is_inside:
                        x_scale = float(w) / math.ceil(b[2])
                        y_scale = float(h) / math.ceil(b[3])

                        x0 = int((p_x - b[0]) * x_scale)
                        y0 = int((p_y - b[1]) * y_scale)

                        if x0 >= w and y0 >= h:
                            x0 = w-1
                            y0 = h-1
                        elif x0 >= w:
                            x0 = w-1
                        elif y0 >= h:
                            y0 = h-1
                        elif x0 < 0 and y0 < 0:
                            x0 = 0
                            y0 = 0
                        elif x0 < 0:
                            x0 = 0
                        elif y0 < 0:
                            y0 = 0

                        #p = prior[y0, x0] # TODO neighborhood

                        #if p == 0:
                        p = 1e-9

                        weights_bbox[bbox_id, y0, x0, :, joint_id] = [1, instance[2], instance[3], p]

        old_weights_bbox = np.copy(weights_bbox)

        for j in range(weights_bbox.shape[0]):
            for t in range(17):
                weights_bbox[j, :, :, 0, t] = gaussian(weights_bbox[j, :, :, 0, t])
            #weights_bbox[j, :, :, 0, :]      = gaussian_multi_input_mp(weights_bbox[j, :, :, 0, :])

        output_bbox = []  #############
        for j in range(weights_bbox.shape[0]):
            inp = weights_bbox[j, :, :, 0, :]
            output = model.predict(np.expand_dims(inp, axis=0))
            output_bbox.append(output[0])

        output_bbox = np.array(output_bbox)

        keypoints_score = []

        for t in range(17):
            indexes = np.argwhere(old_weights_bbox[:, :, :, 0, t] == 1)
            keypoint = []
            for i in indexes:
                cr = crop(output_bbox[i[0],:,:,t], (i[1],i[2]), N=n)
                score = np.sum(cr)

                kp_id    = old_weights_bbox[i[0], i[1], i[2], 2, t]
                kp_score = old_weights_bbox[i[0], i[1], i[2], 1, t]
                p_score  = old_weights_bbox[i[0], i[1], i[2], 3, t]   ## ??
                bbox_id = i[0]

                score = kp_score * score ## keypoint score

                s = [kp_id, bbox_id, kp_score, score]

                keypoint.append(s)
            keypoints_score.append(keypoint)

        bbox_keypoints = np.zeros((weights_bbox.shape[0], 17, 3))

        bbox_ids = np.arange(len(bboxes)).tolist()

        # kp_id, bbox_id, kp_score, my_score
        for i in range(17):

            joint_keypoints = keypoints_score[i]

            if len(joint_keypoints) > 0:

                kp_ids = list(set([x[0] for x in joint_keypoints])) # Get unique kp_ids

                table = np.zeros((len(bbox_ids), len(kp_ids), 4))
                # print table.shape

                for b_id, bbox in enumerate(bbox_ids):
                    for k_id, kp in enumerate(kp_ids):
                        own = [x for x in joint_keypoints if x[0] == kp and x[1] == bbox]

                        if len(own) > 0:
                            table[bbox,k_id] = own[0]
                        else:
                            table[bbox,k_id] = [0]*4

                for b_id, bbox in enumerate(bbox_ids):

                    row = np.argsort(-table[bbox,:,3]) # Sort row of bbox

                    if table[bbox,row[0],3] > 0:
                        for r in row:
                            if table[bbox,r,3] > 0:
                                column = np.argsort(-table[:,r,3]) # Sort scores of keypoint in each bbox

                                if bbox == column[0]: # If max element is owned by bbox, kp is assigned to this bbox
                                    bbox_keypoints[bbox, i, :] = [x[:3] for x in peaks[i] if x[3] == table[bbox,r,0]][0]
                                    break
                                else:
                                    row2 = np.argsort(table[column[0],:,3])
                                    if row2[0] == r:
                                        bbox_keypoints[bbox, i, :] = [x[:3] for x in peaks[i] if x[3] == table[bbox,r,0]][0]
                                        break
            else:        
                for j in range(weights_bbox.shape[0]):
                    b = bboxes[j]
                    x_scale = float(w) / math.ceil(b[2])
                    y_scale = float(h) / math.ceil(b[3])

                    for t in range(17):
                        indexes = np.argwhere(old_weights_bbox[j, :, :, 0, t] == 1)
                        if len(indexes) == 0:
                            max_index = np.argwhere(output_bbox[j,:,:,t] == np.max(output_bbox[j,:,:,t]))
                            bbox_keypoints[j, t,:] = [max_index[0][1]/x_scale + b[0], max_index[0][0]/y_scale + b[1], 0] 


            #print ('number of i'  + i)


        my_keypoints = []

        for i in range(bbox_keypoints.shape[0]):
            k = np.zeros(51)
            k[0::3] = bbox_keypoints[i, :, 0]
            k[1::3] = bbox_keypoints[i, :, 1]
            k[2::3] = [2]*17      # visibility
            
            pose_rate = np.array([.415, .395, .395, .522, .522, 1.18, 1.18, 1.07, 1.07, .926, .926, 1.59, 1.59, 1.29, 1.29, 1.32, 1.32])
            pose_score = 0
            count = 0
            for f in range(17):
                if bbox_keypoints[i, f, 0] != 0 and bbox_keypoints[i, f, 1] != 0:
                    count += 1
                pose_score +=   bbox_keypoints[i, f, 2] #* pose_rate[f] #scores[i] * # scores[i] *
                #print "pose first" , str(pose_score)
            pose_score /=  17.0
            
            #print "pose_score" , str(pose_score)
            #print "count" , str(count)


            my_keypoints.append(k)

            image_data = {
                'image_id'    : idx,
                'bbox'        : bboxes[i],
                'score'       : pose_score,
                'category_id' : 1,
                'keypoints'   : k.tolist()
            }
            my_results.append(image_data)
    end = time.time()

    print 'Elapsed time %d s'%(end-start)

    print len(my_results)

    ann_filename = base_dir + 'annotations/val2017_pose_grammar_keypoint_results_{}.json'.format(modelname)
    # write output
    json.dump(my_results, open(ann_filename, 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = COCO(base_dir + 'annotations/person_keypoints_val2017.json')
    coco_pred = coco_true.loadRes(ann_filename)

    # coco_pred = generator.coco
    #sys.stdout=open(exp_dir + modelname +'.txt' , 'a')
    print('****************************** Epoch ' + str(epoch) + ' Started ***************************************')
    print('\n')
    print model_dir
    print('\n')
    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'keypoints')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    #sys.stdout.close()
    #sys.stdout=open(exp_dir +'test.txt' , 'a')
    
    
#test_of_model('/home/bertec/Repo/pose-grammar/eleventh_epoch_15.h5', 2,'eleventh_', '/home/bertec/Repo/pose-grammar/', 0)