import os
import random
import numpy as np
from multiprocessing import Pool, Manager
from itertools import repeat
from collections import deque

from imantics import Mask
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import torch
from torch.utils.data import DataLoader, Dataset


def train_transform():
    
    return A.Compose([
        A.Resize(512, 512),
        A.Flip(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def test_transform():
    
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def obj_transform():
    return A.Compose([
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30)
    ])


class TrashDataset(Dataset):

    def __init__(self,
                 data_path = '../../ObjectAug/data',
                 json_path='../../ObjectAug/data/train.json',
                 transforms=train_transform(),
                 isTrain=True,
                 numobj=2
                 ):
        
        super().__init__()
        self.data_path = data_path
        self.coco = COCO(json_path)
        self.isTrain=isTrain
        self.json_path=json_path
        self.transforms = transforms
        self.image_ids = sorted(set([value['image_id'] for _, value in self.coco.anns.items()]))
        self.numobj = numobj
        
        getobj = GetObjects(data_path, json_path)
        self.obj_img = getobj.getobj()
        random.shuffle(self.obj_img)
        self.obj_img = deque(self.obj_img)

        print('object load completed')


    def __getitem__(self,index):
        
        image_id = self.image_ids[index]
        
        img_info = self.coco.loadImgs(ids=image_id)[0]

        path = os.path.join(self.data_path, img_info['file_name'])
        
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        

        if self.isTrain == True:
            #####################################################################
            mask = self.get_all_mask(image_id)

            stack = []
            while len(stack) < self.numobj:

                annotation = {'image_id':image_id}

                self.obj_img.rotate()

                img_id = self.obj_img[0][2]
                while image_id == img_id:
                    self.obj_img.rotate()
                    img_id = self.obj_img[0][2]

                category_id, img_obj, img_id, height, width, ann_id = self.obj_img[0]

                mask_acc = self.get_accumulate(mask)
                
                area, r, c = self.find_space(mask_acc, height, width)

                image, mask, _mask = self.patch_obj(image, mask, img_obj, category_id, r, c, height, width)
                
                annotation['category_id'] = category_id
                annotation['segmentation'] = self.gen_maskToseg(_mask)

                annotation['area'] = area
                annotation['bbox'] = [c, r, width, height]
                annotation['iscrowd'] = 0

                stack.append(annotation)

                #bbox, area, iscrowd, segmentation
            #####################################################################


            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids) + stack
            
            # boxes: [x_min, y_min, width, height]
            boxes = np.array([x['bbox'] for x in anns])
            # boxes: [x_min, y_min, x_max, y_max]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2] # width -> x_max
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3] # height -> y_max
            labels = np.array([x['category_id'] for x in anns])
            labels = torch.as_tensor(labels, dtype=torch.int64)

            areas = np.array([x['area'] for x in anns])
            areas = torch.as_tensor(areas, dtype=torch.float32)

            is_crowds = np.array([x['iscrowd'] for x in anns])
            is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)

            segmentation = np.array([x['segmentation'] for x in anns], dtype=object)

            target = {'boxes': boxes, 
                      'labels': labels, 
                      'image_id': torch.tensor([index]), 
                      'area': areas,
                      'iscrowd': is_crowds}
            
            
            while True:
                data = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                }
            
                transformed_data = self.transforms(**data)
                
                if len(transformed_data['bboxes']) > 0:
                    image = transformed_data['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor,zip(*transformed_data['bboxes'])))).permute(1,0)
                    # Effidet -> y,x,y,x
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]
                    target['labels'] = torch.tensor(transformed_data['labels'])
                    break
            

            return image, target, image_id
        
        elif self.isTrain == False:
            transformed_data = self.transforms(image=image)
            image = transformed_data['image']
            
            return image, image_id
    


    def get_all_mask(self, img_id):
        image_info = self.coco.loadImgs(img_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((image_info["height"], image_info["width"]))

        for ann in anns:
            cat = ann['category_id'] + 1
            _mask = self.coco.annToMask(ann)
            mask = np.where(_mask == 0, mask, cat)
        mask = mask.astype(np.uint8)
        return mask


    def get_accumulate(self, mask):
        assert isinstance(mask, np.ndarray), 'mask should be ndarray'
        mask_acc = np.add.accumulate(mask,0,dtype=np.int32)
        mask_acc = np.add.accumulate(mask_acc,1,dtype=np.int32)
        return mask_acc


    def find_space(self, mask_acc, height, width):
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


    def patch_obj(self, image, mask, img_obj, category_id, r, c, height, width):
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


    def gen_maskToseg(self, mask):
        segmentation = Mask(mask).polygons().segmentation
        segmentation = sorted(segmentation, key=len, reverse=True)
        return Mask(mask).polygons().segmentation

    
    def __len__(self) -> int:
        return len(self.image_ids)


class GetObjects(object):

    def __init__(self, data_dir, json_dir, catIds=[0,1,3,4,5,7,9,10]):
        self.data_dir = data_dir
        self.catIds = catIds
        self.coco = COCO(json_dir)
        self.objtransforms = obj_transform()
    
    def getobj(self):
        multiprocessor = os.cpu_count()
        manager = Manager()
        obj_img = manager.list()
        pool = Pool(multiprocessor)

        img_ids = self.get_imgIds()


        pool.starmap(self.get_objects, zip(img_ids, repeat(obj_img)))
        pool.close()
        pool.join()
        
        return obj_img
    
    def get_imgIds(self):
        img_ids = []
        for img_id in [value['id'] for _, value in self.coco.imgs.items()]:
            if len(self.coco.getAnnIds(imgIds=img_id)) == 1:
                img_ids.append(img_id)
        return img_ids



    def get_objects(self, image_id, obj_img):

        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.catIds)

        anns = self.coco.loadAnns(ids=ann_ids)

        img_path = os.path.join(self.data_dir, self.coco.loadImgs(image_id)[0]['file_name'])

        image = self.load_image(img_path)


        for ann in anns:
            mask = self.coco.annToMask(ann) 
            ann_id, category_id, segmentation, area, bbox, iscrowd = ann['id'], \
                        ann['category_id'], ann['segmentation'], ann['area'], ann['bbox'], ann['iscrowd']
            c, r, width, height = map(int, bbox)
            
            _bbox = [c,r,c+width,r+height]

            _obj_img = self.get_masked_obj(image, mask, bbox)

            transformed = self.objtransforms(image=_obj_img, bbox=_bbox)
            _obj_img = transformed['image']

            x, y, xx, yy = map(int, transformed['bbox'])
            height = int(yy-y)
            width = int(xx-x)
            _area = width * height
            if _area > 55_000:
                ratio = np.random.randint(50_000, 55_000)/_area
                width, height = map(int, [width*ratio, height*ratio])
                _obj_img = A.resize(_obj_img, height=height, width=width)

            obj_img.append((category_id, _obj_img, image_id, height, width, ann_id))

        print(f'image_id : {image_id} completed!')

    
    def get_masked_obj(self, image, mask, bbox):
        assert isinstance(image, np.ndarray)
        x, y, width, height = map(int, bbox)
        return cv2.bitwise_and(image,image, mask=mask)[y:y+height,x:x+width,:]    
    

    def load_image(self, image_dir):
        image = cv2.imread(image_dir).astype(np.float64)
        return image/255
