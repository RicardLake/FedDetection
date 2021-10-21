import presets
import utils
from .coco_utils import get_coco, get_coco_kp
import torch
from elec_utils import Electric
def get_dataset(name, image_set, transform, data_path,data_idxs=None):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform,data_idxs=data_idxs)
    return ds, num_classes

def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()

def create_dataloader_coco(is_train=False,data_idxs=None):
    if is_train:
        dataset,num_classes=get_dataset('coco',"train", get_transform(True, 'hflip'),'../../../../../data/coco/',data_idxs)
    else:
        dataset,_ = get_dataset('coco', "val", get_transform(False, 'hflip'), '../../../../../data/coco/')
    transforms=get_transform(is_train,'hflip')
    #dataset = Electric(img_dir,target_dir,transforms,data_idxs)
    #dataset_test = Electric(val_img_dir,val_xml_dir,get_transform(False, 'hflip'))
    if is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
        
        train_batch_sampler = torch.utils.data.BatchSampler(
            sampler, 2, drop_last=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=8,
            collate_fn=utils.collate_fn)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1,
            sampler=sampler, num_workers=8,
            collate_fn=utils.collate_fn)
    #test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    #print('train_dataset length is :*******',len(dataset))
    #print('test_dataset length is :********,',len(dataset_test))


    return data_loader, dataset
def create_dataloader(img_dir,target_dir,is_train=False,data_idxs=None):
    #dataset,num_classes=get_dataset('coco',"train", get_transform(True, 'hflip'),'../../../../../data/coco/')
    #dataset_test, _ = get_dataset('coco', "val", get_transform(False, 'hflip'), '../../../../../data/coco/')
    #train_img_dir,train_xml_dir = '../../../../../data/electric/train_img','../../../../../data/electric/train_xml'
    #val_img_dir, val_xml_dir = '../../../../../data/electric/train_img','../../../../../data/electric/train_xml'
    transforms=get_transform(is_train,'hflip')
    dataset = Electric(img_dir,target_dir,transforms,data_idxs)
    #dataset_test = Electric(val_img_dir,val_xml_dir,get_transform(False, 'hflip'))
    if is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
        
        train_batch_sampler = torch.utils.data.BatchSampler(
            sampler, 2, drop_last=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=8,
            collate_fn=utils.collate_fn)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1,
            sampler=sampler, num_workers=8,
            collate_fn=utils.collate_fn)
    #test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    #print('train_dataset length is :*******',len(dataset))
    #print('test_dataset length is :********,',len(dataset_test))


    return data_loader, dataset
