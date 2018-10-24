
# coding: utf-8

# In[4]:


import os
import cv2


# In[16]:


path_anno_train = '/home/xuyue/workspace/VehicleKeyPointData/keypoint_train.txt'
path_anno_test = '/home/xuyue/workspace/VehicleKeyPointData/keypoint_test.txt'


# In[17]:


with open(path_anno_test) as f:
    annos = f.readlines()


# In[18]:


path_dataset_prefix = '/media/xuyue/HDD/dataset/'
for i in range(len(annos)):
    anno = annos[i]
    anno = anno.strip('\n').split(' ')
    img = cv2.imread(os.path.join(path_dataset_prefix, anno[0]))
    h, w, c = img.shape
    anno.append(str(w))
    anno.append(str(h))
    anno = ' '.join(anno) + '\n'
    annos[i] = anno
    if i % 100 == 0:
        print (i)


# In[19]:


path_anno_train_new = '/home/xuyue/workspace/VehicleKeyPointData/keypoint_train_new.txt'
path_anno_test_new = '/home/xuyue/workspace/VehicleKeyPointData/keypoint_test_new.txt'


# In[20]:


with open(path_anno_test_new,'w') as f:
    for anno in annos:
        f.write(anno)

