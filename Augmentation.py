import os
import time
import json
import random
import numpy as np
from multiprocessing import Pool, Manager
from itertools import repeat
from collections import deque

from tqdm import tqdm
from imantics import Polygons, Mask
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import cv2

'''
!pip install imantics
'''


################Setting these followings###################
json_dir = '/opt/ml/input/data/train.json' #directory to load train_all.json
data_dir = '/opt/ml/input/data' #directory to load image
saving_name = 'train.json' #directory to save obj_augmented train_all.json
multiprocessor = 8

catIds=[0,1,3,4,5,7,9,10] #category_ids to crop

save_dir = 'data' # directory to save images

n = 2 #numbers you want to patch

#prepare folders
for folder in ['batch_01_vt', 'batch_02_vt', 'batch_03']:
    path = os.path.join(save_dir, folder)
    if not os.path.isdir(path):
        os.makedirs(path)
############################################################


def load_json(json_dir):
    with open(json_dir) as f:
        json_file = json.load(f)
    return json_file


def get_masked_obj(image, mask, bbox):
    assert isinstance(image, np.ndarray)
    x, y, width, height = map(int, bbox)
    return cv2.bitwise_and(image,image, mask=mask)[y:y+height,x:x+width,:]


def load_image(image_dir):
    image = cv2.imread(image_dir).astype(np.float64)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64)
    return image 


def get_objects(image_id, catIds, obj_img, coco):

    ann_ids = coco.getAnnIds(imgIds=image_id, catIds=catIds)

    anns = coco.loadAnns(ids=ann_ids)

    img_path = os.path.join(data_dir, coco.loadImgs(image_id)[0]['file_name'])

    image = load_image(img_path)

    for ann in anns:
        mask = coco.annToMask(ann) 
        #ann ['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox', 'iscrowd']
        ann_id, category_id, segmentation, area, bbox, iscrowd = ann['id'], \
                    ann['category_id'], ann['segmentation'], ann['area'], ann['bbox'], ann['iscrowd']
        _, _, width, height = map(int, bbox)
        _obj_img = get_masked_obj(image, mask, bbox)

        obj_img.append((category_id, _obj_img, image_id, width, height, ann_id))

    print(f'image_id : {image_id} completed!')


def get_all_mask(img_id):
    image_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    mask = np.zeros((image_info["height"], image_info["width"]))

    for ann in anns:
        cat = ann['category_id'] + 1
        _mask = coco.annToMask(ann)
        mask = np.where(_mask == 0, mask, cat)
#         mask = np.maximum(coco.annToMask(ann)*cat, mask)
    mask = mask.astype(np.uint8)
    return mask


def get_accumulate(mask):
    assert isinstance(mask, np.ndarray), 'mask should be ndarray'
    mask_acc = np.add.accumulate(mask,0,dtype=np.int32)
    mask_acc = np.add.accumulate(mask_acc,1,dtype=np.int32)
    return mask_acc


def find_space(mask_acc, width, height):
    assert isinstance(mask_acc, np.ndarray), 'mask_acc should be ndarray'
    heap = []
    R, C = mask_acc.shape
    for r in range(R-height):
        for c in range(C-width):
            area = mask_acc[r+height,c+width] - mask_acc[r,c+width] \
                     - mask_acc[r+height,c] + mask_acc[r,c]
            
            heap.append((area, r, c))
    heap = sorted(heap, key=lambda x:x[0])
    
    area, r, c = random.sample(heap[:5000], 1)[0]
    return area, r, c


def patch_obj(image, mask, img_obj, category_id, r, c, height, width):
    assert isinstance(image, np.ndarray) and isinstance(img_obj, np.ndarray), 'image and img_obj should be numpy array'
    _mask = np.zeros(mask.shape)
    category_id += 1
    for i in range(r, r+height):
        for j in range(c, c+width):
            if np.any(img_obj[i-r,j-c,:]):
                image[i,j,:] = img_obj[i-r,j-c,:]
                mask[i,j] = category_id
                _mask[i,j] = category_id
    return image, mask, _mask

#['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox', 'iscrowd']
def gen_maskToseg(mask):
    return Mask(mask).polygons().segmentation


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_json(save_dir, result_dict):
    with open(save_dir,'w') as f:
        json.dump(result_dict, f, cls=NpEncoder)

        
start = time.time()

################################################

train_all_json = load_json(json_dir)

coco = COCO(json_dir)

################################################


# img_ids = list(map(lambda x:x['id'], train_all_json['images']))

img_ids = []
for img_id in list(map(lambda x:x['id'], train_all_json['images'])):
    if len(coco.getAnnIds(imgIds=img_id)) == 1:
        img_ids.append(img_id)


manager = Manager()

obj_img = manager.list()

pool = Pool(multiprocessor)
pool.starmap(get_objects, zip(img_ids, repeat(catIds), repeat(obj_img), repeat(coco)))
pool.close()
pool.join()

print('='*100)
print('Object crop completed!')
print('='*100 + '\n\n\n')

# assert sorted([obj[-1] for obj in obj_img]) == coco.getAnnIds(imgIds=img_ids, catIds=catIds)

random.shuffle(obj_img)



obj_img = deque(obj_img)
################################################

print('='*100)
print('Generating masks and annotation files')
print('='*100)



def add_patchs(img_id, annotations):
    global obj_img
# for img_id in tqdm(list(map(lambda x:x['id'], train_all_json['images']))):
    
    image_info = coco.loadImgs(img_id)[0]

    img_path = os.path.join(data_dir, coco.loadImgs(img_id)[0]['file_name'])

    image = load_image(img_path)

    mask = get_all_mask(img_id)
    
    stack = []

    while len(stack) < n:
        
        annotation = {'image_id':img_id}

        obj_img.rotate()

        image_id = obj_img[0][2]

        while image_id == img_id:
            obj_img.rotate()
            image_id = obj_img[0][2]
        
        category_id, img_obj, image_id, width, height, ann_id = obj_img[0]

        mask_acc = get_accumulate(mask)

        area, r, c = find_space(mask_acc, width, height)

        #이미지 합성, 마스크 합성
        image, mask, _mask = patch_obj(image, mask, img_obj, category_id, r, c, height, width)

        annotation['category_id'] = category_id
        annotation['segmentation'] = gen_maskToseg(_mask)

        annotation['area'] = area
        annotation['bbox'] = [c, r, c+width, r+height]
        annotation['iscrowd'] = 0

        stack.append(annotation)
    
    annotations.extend(stack)
    print(f'image id :{img_id} processed completed')

    cv2.imwrite(os.path.join(save_dir, image_info['file_name']), image)

###############################################################
manager = Manager()

annotations = manager.list()
img_ids = [value['id'] for _, value in coco.imgs.items()]
pool = Pool(multiprocessor)
pool.starmap(add_patchs, zip(img_ids, repeat(annotations)))
pool.close()
pool.join()
###############################################################

print('='*100)
print('Generated image saved')
print('='*100)

ann_ids = list(map(lambda x:x['id'], train_all_json['annotations']))

_annotations = []
for idx, annotation in enumerate(annotations, start=max(ann_ids)+1):
    tmp = {'id':idx}
    for key, value in annotation.items():
        tmp.update({key:value})

    _annotations.append(tmp)

train_all_json['annotations'].extend(_annotations)


###############save json########################
save_json(saving_name, train_all_json)
print(f'file saved at {saving_name}')
################################################

print(f'Image generation completed.\nThis work took {time.time()-start:.2f} sec')

#python Augmentation.py