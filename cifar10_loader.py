import pickle
import glob
import cv2
import tqdm
import os
import sys
import logging
import random
import numpy as np
import math
class Cifa10_data:
    def __init__(self,base_dir,batch_size,rotate_ratio,flip_ratio,cropSize,validate_batch_num=3):
        self.train_data_tensor,self.test_data_tensor,\
        self.train_label_tensor,self.test_label_tensor=load_cifar10(base_dir,rotate_ratio,
                                                                            flip_ratio,
                                                                                cropSize)
        self.batch_size=batch_size
        self.batchs_for_one_epoch_train=self.train_data_tensor.shape[0]//batch_size
        self.batchs_for_one_epoch_test=self.test_data_tensor.shape[0]//batch_size
        self.train_batch_counter=0
        self.test_batch_counter=0
        self.label_map=load_label_map(base_dir)
        self.valid_batches=validate_batch_num
        self.shuffle_train()

    def next_Batch_train(self):
        if(self.train_batch_counter+1)<self.batchs_for_one_epoch_train:
            start_idx=self.train_batch_counter*self.batch_size
            end_idx=(self.train_batch_counter+1)*self.batch_size
            self.train_batch_counter+=1
        else:
            self.train_batch_counter=0
            start_idx=0
            end_idx=self.batch_size
            self.shuffle_train()

        return self.train_data_tensor[start_idx:end_idx],self.train_label_tensor[start_idx:end_idx]

    def next_Batch_test(self):
        if(self.test_batch_counter+1)<self.batchs_for_one_epoch_test:
            start_idx=self.test_batch_counter*self.batch_size
            end_idx=(self.test_batch_counter+1)*self.batch_size
            self.test_batch_counter+=1
        else:
           return None

        return self.test_data_tensor[start_idx:end_idx],self.test_label_tensor[start_idx:end_idx]
    def get_validate_datas(self):
        start_idx=0
        end_idx=self.valid_batches*self.batch_size
        return self.test_data_tensor[start_idx:end_idx],self.test_label_tensor[start_idx:end_idx]
    def shuffle_train(self):
        perm=list(range(self.train_data_tensor.shape[0]))
        np.random.shuffle(perm)
        self.train_data_tensor=self.train_data_tensor[perm]
        self.train_label_tensor=self.train_label_tensor[perm]

def file_loader(file_path):
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    images=map(lambda x:rotate_image(
                                        cv2.cvtColor(
                                            np.array(x).reshape((32,32,3)
                                                                ,order="F"
                                                                ),
                                            cv2.COLOR_RGB2BGR
                                        ),
                                        270,
                                        True
                                    ),
               dict[b'data']
               )

    labels=dict[b'labels']
    return list(images),labels

def load_cifar10(base_dir:str,rotate_ratio=0.1,flip_ratio=0.1,croppedSize=None):
    train_flie_list=glob.glob(os.path.join(base_dir,"data_batch_*"))
    test_file_list=glob.glob(os.path.join(base_dir,"test_batch"))

    train_image=[]
    train_label=[]
    test_image=[]
    test_label=[]
    logging.info("train data file loading....")
    for file_path in tqdm.tqdm(train_flie_list):
        images,labels=file_loader(file_path)
        train_image.extend(images)
        train_label.extend(labels)

    logging.info("test file loading....")
    for file_path in tqdm.tqdm(test_file_list):
        images,labels=file_loader(file_path)
        test_image.extend(images)
        test_label.extend(labels)

    logging.info("data preprocessing")
    train_data_tensor,train_label_tensor=preprocess(train_image,train_label,True,rotate_ratio,flip_ratio,croppedSize)
    test_data_tensor,test_label_tensor=preprocess(test_image,test_label,False,rotate_ratio,flip_ratio,croppedSize)
    return train_data_tensor,test_data_tensor,train_label_tensor,test_label_tensor



def rotate_image(img,rotate,keep_size=False):

    height, width = img.shape[:2]
    if not keep_size:
        heightNew = int(width * math.fabs(math.sin(math.radians(rotate))) + height * math.fabs(math.cos(math.radians(rotate))))
        widthNew = int(height * math.fabs(math.sin(math.radians(rotate))) + width * math.fabs(math.cos(math.radians(rotate))))
    else:
        heightNew=height
        widthNew=width
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), rotate, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation


def preprocess(images_list,label_list,is_train=True,rotate_ratio=0.1,flip_ratio=0.1,cropSzie=None):
    rotate_angle=[30,60,90]
    flip_code=[1]
    if cropSzie==None:
        offset=0
    else:
        offset=(images_list[0].shape[0]-cropSzie)//2

    cropped_size=images_list[0].shape[0]-offset
    cropSzie=images_list[0].shape[0]-2*offset
    if not is_train:
        image_element_tensor=[item[offset:cropped_size,offset:cropped_size,:].reshape(1,cropSzie,cropSzie,3) for item in images_list]
        return np.concatenate(image_element_tensor,axis=0).astype(np.float32),build_onehot(label_list,10).astype(np.float32)
    else:
        smaple_idx_list=random.sample(range(0,len(images_list)),int(len(images_list)*rotate_ratio))
        smaple_flip_idx_list=random.sample(range(0,len(images_list)),int(len(images_list)*flip_ratio))
        rotated_images=list(map(lambda x:rotate_image(images_list[x],np.random.choice(rotate_angle),True),smaple_idx_list))
        rotate_image_labels=[label_list[item] for item in smaple_idx_list]
        fliped_images=list(map(lambda x:cv2.flip(images_list[x],np.random.choice(flip_code)),smaple_flip_idx_list))
        fliped_image_labels=[label_list[item] for item in smaple_flip_idx_list]
        images_list.extend(rotated_images)
        label_list.extend(rotate_image_labels)
        images_list.extend(fliped_images)
        label_list.extend(fliped_image_labels)

        image_element_tensor=[item[offset:cropped_size,offset:cropped_size,:].reshape(1,cropSzie,cropSzie,3) for item in images_list]
        return np.concatenate(image_element_tensor,axis=0).astype(np.float32),build_onehot(label_list,10).astype(np.float32)


def build_onehot(labels,label_num):
    label_tensor=np.zeros((len(labels),label_num),dtype=np.int)
    for i in range(len(labels)):
        label_tensor[i,labels[i]]=1
    return label_tensor

def load_label_map(base_dir):
    file_path=os.path.join(base_dir,"batches.meta")
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return [str(item, encoding = "utf-8") for item in dict[b'label_names']]

if __name__ == "__main__":
    logger = logging.getLogger()    # initialize logging class
    logger.setLevel(logging.DEBUG)  # default log level
    format = logging.Formatter("%(asctime)s - %(message)s")    # output format
    sh = logging.StreamHandler(stream=sys.stdout)    # output to standard output
    sh.setFormatter(format)
    logger.addHandler(sh)


    data_loader=Cifa10_data("C:\\Users\\rebel\\.keras\\datasets\\cifar-10-batches-py",128,0.25,0.25,28,3)
    print(data_loader.test_data_tensor.shape)
    print(data_loader.train_data_tensor.shape)
    print(data_loader.get_validate_datas()[0].shape)

