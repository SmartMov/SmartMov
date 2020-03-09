import os
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
from PIL import Image

from mrcnn import utils

class DavisDataset(utils.Dataset):
    def load_davis(self, direc, subset='all', select_class='all', personal_ids=None):
        assert subset in ['all','train','val'], "subset non valide, doit être 'train', 'val' ou 'all'"
        
        self.direc = direc
        self.direc_images = os.path.join(self.direc,"JPEGImages")
        self.direc_annot = os.path.join(self.direc,"Annotations")
        if subset!='all':
            check_path = os.path.join(self.direc,"ImageSets/2017/"+subset+".txt")
            check_file = []
            with open(check_path) as file:
                for line in file:
                    line = line.split("\n")[0]
                    check_file.append(line)
        else:
            check_path1 = os.path.join(self.direc,"ImageSets/2017/train.txt")
            check_path2 = os.path.join(self.direc,"ImageSets/2017/val.txt")
            check_file = []
            with open(check_path1) as file:
                for line in file:
                    line = line.split("\n")[0]
                    check_file.append(line)
            with open(check_path2) as file:
                for line in file:
                    line = line.split("\n")[0]
                    check_file.append(line)
                
        with open(os.path.join(direc,'davis_semantics.json')) as json_file:
            self.json_data = json.load(json_file)
        
        if select_class=='all':
            self.classes = []
            self.possible_dir =[]
            
            for dire in os.listdir(os.path.join(direc,"JPEGImages")):
                if dire in check_file:
                    self.possible_dir.append(dire)
            
            with open(os.path.join(direc,'categories.json')) as json_file:
                data = json.load(json_file)
                for p in data:
                    self.classes.append(p)
        else:
            self.classes = select_class
            self.possible_dir =[]
            for dire in os.listdir(os.path.join(direc,"JPEGImages")):
                if dire not in check_file:
                    continue
                for p in self.json_data[dire].values():
                    if p in select_class:
                        self.possible_dir.append(dire)
                        break
                    
        if personal_ids!=None:
            if select_class!='all':
                assert len(select_class)==len(personal_ids), "Il doit y avoir autant d'ids que de classes sélectionnées"
                self.pers_ids = personal_ids
            else: # Si on veut toutes les classes
                assert len(self.classes)==len(personal_ids), "Il doit y avoir autant d'ids que de classes sélectionnées (ici {})".format(len(self.classes))
                self.pers_ids = personal_ids
        else: # Si les ids ne sont pas perso
            self.pers_ids = list(np.arange(len(self.classes))+1)
        
        for index,i in enumerate(self.pers_ids): # Pour chaque classe
            self.add_class("davis2017", i, select_class[index])
        
        image_id = 0
        for d in self.possible_dir: # Pour chaque dossier
            dir_path = os.path.join(self.direc_images,d)
            dir_annot_path = os.path.join(self.direc_annot,d)
            annot_dict = self.json_data[d]
            for im in os.listdir(dir_path):
                im_path = os.path.join(dir_path,im)
                img = plt.imread(im_path)
                annot_path = os.path.join(dir_annot_path,im.split(".")[0]+".png")
                self.add_image(
                    "davis2017",image_id=image_id,
                    path=im_path,
                    width=img.shape[1],
                    height=img.shape[0],
                    annotations_path=annot_path,
                    annotations_dict=annot_dict)
                image_id+=1
        
        self.nb_images = image_id
        
    def load_mask(self,image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "davis2017":
            return super(DavisDataset, self).load_mask(image_id)
        annot = image_info['annotations_path']
        annot = np.array(Image.open(annot))
        indices_right_class = []
        class_masks = []
        for index,p in enumerate(image_info['annotations_dict'].values()):
            if p in self.classes:
                indices_right_class.append(index+1)
                class_masks.append(p)
        nb_masks = len(indices_right_class)
        masks = np.zeros((annot.shape[0],annot.shape[1],nb_masks))
        for i,p in enumerate(indices_right_class):
            temp = get_object(p,annot)
            masks[...,i] = temp
        classes = []
        for e in class_masks:
            ind = np.where(np.array(self.classes)==e)[0]
            ind = ind[0]
            classes.append(self.pers_ids[ind])
        return masks.astype(np.bool),np.array(classes)
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "davis2017":
            return info['path']
        else:
            super(DavisDataset, self).image_reference(image_id)
    
    def get_nb_images(self):
        return self.nb_images

def get_object(indice,annotations):
    return annotations==indice



