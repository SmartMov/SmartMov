import os
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
from PIL import Image
from mrcnn import utils

################################################
# Vérifier les classes dans l'union
################################################

class UnionDataset(utils.Dataset):
    def load_union(self,nb_images_per_dataset,*args):
        self.acc_nb_im_datasets = []
        self.base_datasets = args
        index = 0
        index2 = 0
        for dataset in args: # Pour chaque dataset
            class_info = dataset.class_info
            for cl in class_info:
                source = 'union'
                class_id = cl['id']
                name = cl['name']
                # self.add_class(*(cl.values()))
                self.add_class(source,class_id,name)
            for im in dataset.image_info:
                if index2>=nb_images_per_dataset:
                    break
                image_id = index
                # source = im['source']
                source = 'union'
                path = im['path']
                arg = dict()
                for k,val in zip(im,im.values()):
                    if k in ['id','source','path']:
                        continue
                    else:
                        arg[k] = val
                arg['original_index'] = index2
                self.add_image(source=source,image_id=image_id,
                               path=path,**arg)
                index+=1
                index2+=1
            self.acc_nb_im_datasets.append(index)
            index2 = 0
    
    def load_mask(self,image_id):
        image_info = self.image_info[image_id]
        tmp_acc = np.array(self.acc_nb_im_datasets)
        ind_data = np.sum(tmp_acc <= image_id)
        right_dataset = self.base_datasets[ind_data]
        return right_dataset.load_mask(image_info['original_index'])
    
    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for i,info in enumerate(self.class_info):
            if info["id"] == class_id: # Si l'id de la nouvelle classe est déjà dans la base
                if info["source"] == source:
                    # source.class_id combination already available, skip
                    return
                else: # Si même id mais pas même source, vérifier si les noms correspondent
                    # Si même id mais pas même nom : soit erreur soit construire les dictionnaires de remplacement (ici erreur)
                    assert info["name"]==class_name, "Mêmes id ({}) mais pas mêmes noms de classes ('{}' et '{}')".format(class_id,info["name"],class_name)
                    if info["name"]==class_name:
                        return
            else: # Si id différents, être sur que les noms sont différents
                assert info["name"]!=class_name, "Mêmes noms de classe ('{}') mais ids différents ({} et {})".format(class_name,info["id"],class_id)
        
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })