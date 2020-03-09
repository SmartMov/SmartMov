import numpy as np
import os
import glob
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self,timestep=5,taux_validation=0.2,input_shape=(480,720)):
        self.timestep = timestep
        self.taux_val = taux_validation
        self.input_shape = input_shape
    
    def load_data(self,directory,shuffle=True,nb_images=None):
        list_images_train = []
        list_gt_train = []
        list_images_val = []
        list_gt_val = []
    
        for dossier in directory:
            input_dir = os.path.join(dossier,"input/")
            gt_dir = os.path.join(dossier,"groundtruth/")
            input_images = sorted(glob.glob(input_dir+"*.jpg"))
            gt_images = sorted(glob.glob(gt_dir+"*.png"))
            for i in range(len(input_images[self.timestep:])+1):
                if i<=(1-self.taux_val)*len(input_images[self.timestep:]): # Train
                    list_images_train.append(input_images[i:i+self.timestep])
                    list_gt_train.append(gt_images[i+self.timestep-1])
                else:
                    list_images_val.append(input_images[i:i+self.timestep])
                    list_gt_val.append(gt_images[i+self.timestep-1])
        
        self.im_train = np.array(list_images_train)
        self.gt_train = np.array(list_gt_train)
        self.im_val = np.array(list_images_val)
        self.gt_val = np.array(list_gt_val)
        
        if shuffle:
            p = np.random.permutation(len(list_gt_train))
            self.im_train = self.im_train[p]
            self.gt_train = self.gt_train[p]
            p_val = np.random.permutation(len(list_gt_val))
            self.im_val = self.im_val[p_val]
            self.gt_val = self.gt_val[p_val]
        
        if nb_images!=None:
            seuil_train = int(np.round((1-self.taux_val)*nb_images))
            seuil_val = int(np.round(self.taux_val*nb_images))
            self.im_train = self.im_train[:seuil_train]
            self.gt_train = self.gt_train[:seuil_train]
            self.im_val = self.im_val[:seuil_val]
            self.gt_val = self.gt_val[:seuil_val]
    
    def load_data_by_files(self,dir_save):
        self.im_train = np.load(os.path.join(dir_save,"im_train.npy"))
        self.gt_train = np.load(os.path.join(dir_save,"gt_train.npy"))
        self.im_val = np.load(os.path.join(dir_save,"im_val.npy"))
        self.gt_val = np.load(os.path.join(dir_save,"gt_val.npy"))
    
    def get_nb_train(self):
        return self.im_train.shape[0]
    
    def get_nb_val(self):
        return self.im_val.shape[0]
    
    def get_nb_images(self):
        return self.im_train.shape[0]+self.im_val.shape[0]
    
    def save_data(self,dir_save):
        np.save(os.path.join(dir_save,"im_train.npy"),self.im_train)
        np.save(os.path.join(dir_save,"gt_train.npy"),self.gt_train)
        np.save(os.path.join(dir_save,"im_val.npy"),self.im_val)
        np.save(os.path.join(dir_save,"gt_val.npy"),self.gt_val)
    
    def generate_train(self,batch_size):
        assert self.get_nb_train()>=(self.timestep*batch_size), "Pas assez d'images de train pour ce batch_size et ce timestep, il en faut au moins {} et il y en a {}".format(self.timestep*batch_size,
                                                                                                              self.get_nb_train())
        i=0
        nb = self.get_nb_train()
        while i<=(nb//batch_size):
            batch_input = np.zeros((batch_size,self.timestep,self.input_shape[0],self.input_shape[1],3),dtype=np.float32)
            batch_target = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],1),dtype=np.uint8)
            for j in range(batch_size):
                images_to_use = self.im_train[i*batch_size+j]
                for index,im in enumerate(images_to_use):
                    img = Image.open(im)
                    img = img.resize((self.input_shape[1],self.input_shape[0]))
                    batch_input[j,index] = ((np.array(img)-127.5)/127.5).astype(np.float32)
                gt = Image.open(self.gt_train[i*batch_size+j])
                gt = gt.resize((self.input_shape[1],self.input_shape[0]))
                gt = np.array(gt,dtype=np.uint8)
                gt = np.expand_dims(gt,axis=-1)
                gt[gt<=90] = 0
                gt[gt>0] = 1
                batch_target[j] = gt
            
            i+=1
            if i>=(nb//batch_size):
                i=0
            
            yield batch_input,batch_target
    
    def generate_val(self,batch_size):
        assert self.get_nb_val()>=(self.timestep*batch_size), "Pas assez d'images de validation pour ce batch_size et ce timestep, il en faut au moins {} et il y en a {}".format(self.timestep*batch_size,
                                                                                                              self.get_nb_val())
        i=0
        nb = self.get_nb_val()
        while i<=(nb//batch_size):
            batch_input = np.zeros((batch_size,self.timestep,self.input_shape[0],self.input_shape[1],3),dtype=np.float32)
            batch_target = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],1),dtype=np.uint8)
            for j in range(batch_size):
                images_to_use = self.im_val[i*batch_size+j]
                for index,im in enumerate(images_to_use):
                    img = Image.open(im)
                    img = img.resize((self.input_shape[1],self.input_shape[0]))
                    batch_input[j,index] = ((np.array(img)-127.5)/127.5).astype(np.float32)
                gt = Image.open(self.gt_val[i*batch_size+j])
                gt = gt.resize((self.input_shape[1],self.input_shape[0]))
                gt = np.array(gt,dtype=np.uint8)
                gt = np.expand_dims(gt,axis=-1)
                gt[gt<=90] = 0
                gt[gt>0] = 1
                batch_target[j] = gt
            
            i+=1
            if i>=(nb//batch_size):
                i=0
            
            yield batch_input,batch_target
    
    def test_batch(self, indice):
        assert indice<self.get_nb_val(),"Cet indice est trop grand (max {})".format(self.get_nb_val()-1)
        batch_input = np.zeros(shape=(1,self.timestep,self.input_shape[0],self.input_shape[1],3),dtype=np.uint8)
        batch_target = np.zeros(shape=(1,self.input_shape[0],self.input_shape[1],1),dtype=np.uint8)
        images_to_use = self.im_val[indice]
        for index,im in enumerate(images_to_use):
            img = Image.open(im)
            img = img.resize((self.input_shape[1],self.input_shape[0]))
            img = np.array(img)
            batch_input[0,index] = img
        gt = Image.open(self.gt_val[indice])
        gt = gt.resize((self.input_shape[1],self.input_shape[0]))
        gt = np.array(gt,dtype=np.uint8)
        gt = np.expand_dims(gt,axis=-1)
        gt[gt<=90] = 0
        gt[gt>0] = 1
        batch_target[0] = gt
        return batch_input, batch_target