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
    def __init__(self,path,data_idxs,transforms):
        self.imgs=[os.path.join(path[0],x) for x in os.listdir(path[0])]
        self.anns=[os.path.join(path[1],x[:-4]+'.xml') for x in os.listdir(path[0])]
        #if data_idxs:
        #    self.data_idxs=data_idxs
        if transforms:
            self.transforms=transforms
        #if self.data_idxs:
        #    self.imgs=[self.imgs[i] for i in self.data_idxs]
        #    self.anns=[self.anns[i] for i in self.data_idxs]

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
