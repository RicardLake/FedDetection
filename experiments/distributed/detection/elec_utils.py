import torch
import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
from xml.dom.minidom import parse

label_dict={'1':1}
def parse_ann(path,index):
    dom = parse(path)
    data = dom.documentElement
    objs = data.getElementsByTagName('object')
    classes=[]
    boxes=[]
    iscrowd=[]
    areas=[]
    for object in objs:
        cls = label_dict[object.getElementsByTagName('name')[0].childNodes[0].nodeValue]
        box_obj=object.getElementsByTagName('bndbox')[0]
        xmin = float(box_obj.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
        ymin = float(box_obj.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
        xmax = float(box_obj.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
        ymax = float(box_obj.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
        box=[xmin,ymin,xmax,ymax]
        classes.append(cls)
        boxes.append(box)
        iscrowd.append(0)
        areas.append((xmax-xmin+1)*(ymax-ymin+1))
    return {'labels':torch.LongTensor(classes),'boxes':torch.FloatTensor(boxes),'image_id':torch.LongTensor([index]),'iscrowd':torch.LongTensor(iscrowd),'area':torch.FloatTensor(areas)}

class Electric(Dataset):
    def __init__(self,img_dir,ann_dir,transforms=None,data_idxs=None):
        self.imgs=[os.path.join(img_dir,x) for x in os.listdir(img_dir)]
        self.anns=[os.path.join(ann_dir,x[:-4]+'.xml') for x in os.listdir(img_dir)]
        self.transforms=None
        if transforms:
            self.transforms=transforms
        if data_idxs is not None:
            self.imgs=[self.imgs[i] for i in data_idxs]
            self.anns=[self.anns[i] for i in data_idxs]

    def __getitem__(self, index):
        img=Image.open(self.imgs[index]).convert('RGB')
        target=parse_ann(self.anns[index],index)
        if self.transforms is not None:
            img,target=self.transforms(img,target)
        return img,target

    def __len__(self):
        return len(self.imgs)

# if __name__=='__main__':
#     parse_ann('../../dataset/elec_chip/xml_all_pad/001_0d_0_1_c.xml')
