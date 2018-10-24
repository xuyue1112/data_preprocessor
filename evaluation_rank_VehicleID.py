#coding:utf-8

from __future__ import division
import random
import scipy.io as sio
import numpy as np
import argparse
import os
parser = argparse.ArgumentParser(description="calculate rank1, rank5 for VehicleID dataset using specific split strategy.")
parser.add_argument('-cn','--class_num', type=int, help='class number of test set. options:800, 1600, 2400.')
parser.add_argument('-gn','--group_num', type=int, help='group number of evaluation to calculate average value. no less than 1.')
parser.add_argument('-f','--feature', type=str, help='path of the .mat feature file.') 
parser.add_argument('-l', '--imageList', type=str, help='path of the image list.')
args = parser.parse_args()

def compare1(a, b):
    if len(a[1]) > len(b[1]):
        return -1
    elif len(a[1]) < len(b[1]):
        return 1
    else:
        if int(a[0]) < int(b[0]):
            return -1
        elif int(a[0]) > int(b[0]):
            return 1
        else:
            return 0

result = []
for iterCount in range(0, args.group_num):
    print ("start iter {:d}".format(iterCount+1))
    print "split gallery/probe set ..." 

    testFile = open(os.path.join(args.imageList, 'test' + str(args.class_num) + "_all.lst"))
    cars = {}
    for i, line in enumerate(testFile.readlines()):
        lineList = line.split()
        if cars.has_key(lineList[1]):
            cars[lineList[1]].append(tuple((i, lineList[0])))
        else:
            cars[lineList[1]] = []
            cars[lineList[1]].append(tuple((i, lineList[0])))
    carsList = cars.items()
    carsList.sort(compare1)
    count = 0


    gallery_ids = []
    probe_ids = []
    gallery_gts = []
    probe_gts = []

    for car in carsList:
        gallery_count = random.randint(0, len(car[1])-1)
        gallery_ids.append(car[1][gallery_count][0])
        gallery_gts.append(car[0])
        for i in range(0, len(car[1])):
            if i != gallery_count:
                probe_ids.append(car[1][i][0])
                probe_gts.append(car[0])


    print 'calculate similarity matrix ...'
    #分离probe 和 gallery的feature
    features = sio.loadmat(args.feature)['feature']
    feature_query = np.zeros(shape=(len(probe_ids), features.shape[1]))
    feature_ref = np.zeros(shape=(len(gallery_ids), features.shape[1]))
    for i in range(0, len(probe_ids)):
        feature_query[i,...] = features[probe_ids[i]]
    for i in range(0, len(gallery_ids)):
        feature_ref[i, ...] = features[gallery_ids[i]]
    similar_cosine = np.zeros(shape=(feature_query.shape[0], feature_ref.shape[0]))

    #将特征向量归一化
    L2_query = np.zeros(shape=(feature_query.shape[0]))
    L2_ref = np.zeros(shape=(feature_ref.shape[0]))
    for i in range(0, len(feature_query)):
        L2_query[i] = np.linalg.norm(feature_query[i,:])
    for i in range(0, len(feature_ref)):
        L2_ref[i] = np.linalg.norm(feature_ref[i,:])    
        
    #计算query中每个元素和ref中每个元素的距离    
    for i in range(0, len(feature_query)):
        for j in range(0, len(feature_ref)):
            v1 = feature_query[i,:]
            v2 = feature_ref[j,:]
            similar_cosine[i,j] = np.dot(v1,v2) / (L2_query[i] * L2_ref[j])

    #对query中的每个元素，将它与ref中每个元素的相似度按照从大到小排序
    label_sorted_cosine = np.zeros(shape=np.shape(similar_cosine), dtype='int64')
    for i in range(0, similar_cosine.shape[0]):
        label_sorted_cosine[i,:] = np.argsort(-similar_cosine[i])

    print "calculate ran1 & rank ..."
    r1_count = 0
    r5_count = 0
    for i in range(0, len(feature_query)):
        #if int(galleryList[label_sorted_cosine[i,0] ].strip("\n").split()[1]) == int(probeList[i].strip('\n').split()[1]):
        if gallery_gts[label_sorted_cosine[i,0]] == probe_gts[i]:
            r1_count += 1
        for j in range(0, 5):
            #if int(galleryList[label_sorted_cosine[i,j] ].strip("\n").split()[1]) == int(probeList[i].strip('\n').split()[1]):
            if gallery_gts[label_sorted_cosine[i,j]] == probe_gts[i]:
                r5_count += 1
                break
    result.append((r1_count / len(feature_query),r5_count / len(feature_query)))
    print ("r1(group{:d}):{:f}".format(iterCount+1, r1_count / len(feature_query)))
    print ("r5(group{:d}):{:f}".format(iterCount+1, r5_count / len(feature_query)))

r1 = 0
r5 = 0
for i in range(0, args.group_num):
    r1 += result[i][0]
    r5 += result[i][1]
r1 /= args.group_num
r5 /= args.group_num
print ("r1(avg):{:f}".format(r1))
print ("r5(avg):{:f}".format(r5))

