# -*- coding: utf-8 -*-
# 生成PyTorch的 torchvision.datasets.ImageFolder所需的格式

# 1.生成(图片名：类别)组成的列表
with open('/media/xuyue/HDD/dataset/VeRi/annot/keypoint_image_test.txt','r') as f:
    annos = f.readlines()

annos = [(x.split(' ')[0].split('/')[-1], int(x.split(' ')[-1].strip('\n'))) for x in annos]

# 2.移动数据
import shutil
import os
PATH_INPUT = '/media/xuyue/HDD/dataset/VeRi/image_test'
PATH_OUTPUT = '/media/xuyue/HDD/dataset/VeRi/train_split_by_class/val'
if not os.path.exists(PATH_OUTPUT):
    os.mkdir(PATH_OUTPUT)
else:
    shutil.rmtree(PATH_OUTPUT)
    os.mkdir(PATH_OUTPUT)
for anno in annos:
    if not os.path.exists(os.path.join(PATH_OUTPUT, str(anno[1]))):
        os.mkdir(os.path.join(PATH_OUTPUT, str(anno[1])))
        print ('create folder {:d}', anno[1])
    shutil.copy(os.path.join(PATH_INPUT, anno[0]), \
                os.path.join(PATH_OUTPUT, str(anno[1])))



