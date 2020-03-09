import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
import tensorflow as tf
import tensorflow.keras.models as KM
import os
import sys
import time
import cv2
import glob
import metrics

ROOT_DIR = os.path.abspath("")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import coco
import unetlib

class SmartMov:
    def load_models(self, models_to_load='all', **kwargs):
        """
        Charger les modèles spécifiés

        Parameters
        ----------
        models_to_load : str qui représente les modèles à charger. Doit être 'all', 'unet' ou 'rcnn'. Par défaut 'all'
        **kwargs : shape_unet : tuple (H,W,3) qui représente la taille des images en entrée du U-Net. Si 'all' ou 'unet'\n
                   model_unet : str qui représente le chemin vers le fichier .h5 à charger. Si 'all' ou 'unet'\n
                   timestep : int qui représente le timestep. Si 'all' ou 'unet'\n
                   model_rcnn : str qui représente le chemin vers le fichier .h5 des poids à charger. Si 'all' ou 'rcnn'\n
                   config : object de configuration du Mask-RCNN. Si 'all' ou 'rcnn'\n
                   class_names : list des noms des classes que le Mask-RCNN doit trouver (incluant 'BG'). Si 'all' ou 'rcnn'

        Returns
        -------
        None.

        """
        assert models_to_load in ['all','unet','rcnn'], "models_to_load doit être soit 'all', soit 'unet', soit 'rcnn'"
        if models_to_load=='all':
            self.load_unet(**kwargs)
            self.load_rcnn(**kwargs)
        elif models_to_load=='unet':
            self.load_unet(**kwargs)
        elif models_to_load=='rcnn':
            self.load_rcnn(**kwargs)
    
    def load_unet(self,**kwargs):
        """
        Charger le modèle U-Net depuis un fichier .h5

        Parameters
        ----------
        **kwargs : shape_unet : tuple (H,W,3) qui représente la taille des images en entrée du U-Net\n
                   model_unet : str qui représente le chemin vers le fichier .h5 à charger\n
                   timestep : int qui représente le timestep

        Returns
        -------
        None.
        """
        self.shape = kwargs['shape_unet']
        self.timestep = kwargs['timestep']
        self.unet = KM.load_model(kwargs['model_unet'])
    
    def load_rcnn(self,**kwargs):
        """
        Charger le modèle du Mask-RCNN depuis le fichier .h5 des poids

        Parameters
        ----------
        **kwargs : model_rcnn : str qui représente le chemin vers le fichier .h5 des poids à charger\n
                   config : object de configuration du Mask-RCNN\n
                   class_names : list des noms des classes que le Mask-RCNN doit trouver (incluant 'BG')

        Returns
        -------
        None.
        """
        MODEL_DIR = os.path.abspath(kwargs['model_rcnn']+"/../logs/")
        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)
        self.model_dir = MODEL_DIR
        self.rcnn = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=kwargs['config'])
        self.rcnn.load_weights(kwargs['model_rcnn'], by_name=True)
        assert kwargs['config'].NUM_CLASSES == len(kwargs['class_names']), "Il n'y a pas le même nombre de classes que de noms de classes"
        self.class_names = kwargs['class_names']
        self.rcnn_mode = 'inference'
    
    def predict(self,images,models_to_use='all',priority='rcnn'):
        """
        Réalise la prédiction à partir des réseaux spécifiés

        Parameters
        ----------
        images : tableau d'images (shape=(1,TIMESTEP,H,W,3) ou (TIMESTEP,H,W,3) ou (H,W,3) si seulement Mask-RCNN) de type np.uint8\n
        models_to_use : str qui représente les modèles à utiliser. Doit être 'all', 'unet' ou 'rcnn'. Par défaut 'all'
        priority : str qui représente le modèle à prioriser pour générer le masque final. Doit être 'rcnn' ou 'unet'. Utile seulement si models_to_use='all'. Par défaut 'rcnn'.

        Returns
        -------
        Tuple (prediction, nombre de personnes) si 'all'
        prediction si 'rcnn'
        masque du mouvement si 'unet'
        """
        assert models_to_use in ['all','unet','rcnn'],"models_to_use doit être soit 'all', soit 'unet', soit 'rcnn'"
        assert priority in ['rcnn','unet'],"priority doit être soit 'rcnn' soit 'unet'"
        if models_to_use=='unet':
            assert 'unet' in self.__dict__.keys(), "Pas de modèle U-Net chargé"
            assert len(images.shape) in [4,5], "images doit être de quatre ou cinq dimensions"
            if len(images.shape) == 5:
                assert images.shape[0]==1, "La première dimension doit être à 1 (batch_size)"
            else:
                images = np.expand_dims(images,axis=0)
            assert images.shape[1]==self.timestep, "Le timestep (deuxième dimension) doit être de 10"
            return self.predict_unet(images)
        elif models_to_use=='rcnn':
            assert 'rcnn' in self.__dict__.keys(), "Pas de modèle Mask-RCNN chargé"
            assert self.rcnn_mode=='inference',"Le Mask-RCNN n'est pas en inférence"
            assert len(images.shape) in [3,4,5], "images doit être de trois,quatre ou cinq dimensions"
            if len(images.shape) == 5:
                assert images.shape[0]==1, "La première dimension doit être à 1 (batch_size)"
                im_mask = np.copy(images[0,-1])
            elif len(images.shape) == 4:
                im_mask = np.copy(images[-1])
            else:
                im_mask = np.copy(images)
            return self.predict_rcnn(im_mask)
        elif models_to_use=='all':
            assert ('unet' in self.__dict__.keys() and 'rcnn' in self.__dict__.keys()),"Aucun modèle chargé"
            assert self.rcnn_mode=='inference',"Le Mask-RCNN n'est pas en inférence"
            assert len(images.shape) in [4,5], "images doit être de quatre ou cinq dimensions"
            if len(images.shape) == 5:
                assert images.shape[0]==1, "La première dimension doit être à 1 (batch_size)"
            else:
                images = np.expand_dims(images,axis=0)
            assert images.shape[1]==self.timestep, "Le timestep (deuxième dimension) n'est pas bon (doit être à {})".format(self.timestep)
            
            nb_pers = 0
            
            motion_mask = self.predict_unet(images)
            
            if np.sum(motion_mask)<=200: # Si pas de mouvement, pas d'estimation
                return {'rois':[], 'masks':[], 'class_ids':[], 'scores':[]}, nb_pers
            
            im_mask = images[0,-1]
            
            r = self.predict_rcnn(im_mask)
            
            seg_motion_mask = np.zeros(shape=(motion_mask.shape[0],motion_mask.shape[1],r['rois'].shape[0]),dtype=np.bool)
        
            for i,rec in enumerate(r['rois']): # Chaque ligne => un rectangle (y1,x1,y2,x2)
                y1,x1,y2,x2 = rec
                mask_temp = np.zeros_like(motion_mask,dtype=np.bool)
                for j in range(x1,x2):
                    for k in range(y1,y2):
                        mask_temp[k,j] = True
                seg_motion_mask[:,:,i] = mask_temp & motion_mask
            
            for i in range(r['masks'].shape[-1]):
                mask = r['masks'][...,i]
                motion = seg_motion_mask[...,i]
                if np.sum(mask & motion) >= 0.2*np.max([np.sum(mask),np.sum(motion)]):
                    if priority=='rcnn':
                        r['masks'][...,i] = mask
                    else:
                        r['masks'][...,i] = motion
                    nb_pers+=1
                else:
                    r['rois'][i] = [0,0,0,0]
                    r['masks'][...,i] = np.zeros((mask.shape[0],mask.shape[1]),dtype=np.bool)
                    
            return r,nb_pers
    
    def predict_unet(self, images):
        """
        Réaliser la prédiction à partir du U-Net

        Parameters
        ----------
        images : tableau d'images (shape=(1,TIMESTEP,H,W,3) ou (TIMESTEP,H,W,3)) de type np.uint8\n

        Returns
        -------
        motion_mask : masque de mouvement booléen
        """
        images_shape = images[0,0].shape # Shape initiale des images
        im_unet = np.zeros((1,self.timestep,self.shape[0],self.shape[1],self.shape[2]))
        for i,im in enumerate(images[0]):
            im_unet[0,i] = (resize(im,(self.shape[0],self.shape[1]))-0.5)*2.
        motion_mask = np.argmax(self.unet.predict(im_unet)[0],axis=-1).astype(np.bool)
        motion_mask = np.round(resize(motion_mask,(images_shape[0],images_shape[1]))).astype(np.bool)
        return motion_mask
    
    def predict_rcnn(self,image):
        """
        Réaliser la prédiction avec le Mask-RCNN

        Parameters
        ----------
        image : tableau d'images (shape=(H,W,3)) de type np.uint8\n

        Returns
        -------
        prediction du Mask-RCNN (dict)

        """
        results = self.rcnn.detect([image],verbose=0)
        return results[0]
    
    def create_model(self, models_to_create='all',**kwargs):
        """
        Créé les modèles spécifiés

        Parameters
        ----------
        models_to_create : Défini le modèle à créer. Doit être 'all', 'unet' ou 'rcnn'. Par défaut 'all'
        **kwargs : shape_unet : tuple (H,W,3) donnant la taille en entrée du U-Net. Nécéssaire si 'all' ou 'unet'\n
                   timestep : int représentant le timestep du U-Net. Nécéssaire si 'all' ou 'unet'
                   model_dir : str qui représente le dossier ou enregistrer les checkpoints lors du train du modèle\n
                   config : objet qui représente la configuration du Mask-RCNN à créer\n
                   reuse : bool qui correspond à réutiliser d'anciens poids (si True) ou non (si False)\n
                   model_rcnn : str qui représente le fichier .h5 à charger qui contient les poids si reuse=True
                   class_names : list qui représente les classes à chercher

        Returns
        -------
        None.
        """
        assert models_to_create in ['all','unet','rcnn'],"models_to_create doit être 'all','unet' ou 'rcnn'"
        if models_to_create=='unet':
            self.unet = self.create_unet(kwargs['shape_unet'],kwargs['timestep'])
        elif models_to_create=='rcnn':
            self.rcnn = self.create_rcnn(**kwargs)
        elif models_to_create=='all':
            self.unet = self.create_unet(kwargs['shape_unet'])
            self.rcnn = self.create_rcnn(**kwargs)
    
    def create_unet(self, shape, timestep):
        self.shape = shape
        self.timestep = timestep
        return unetlib.create(shape,timestep)
    
    def create_rcnn(self, **kwargs):
        """
        Créé un modèle Mask-RCNN à entrainer par la suite

        Parameters
        ----------
        **kwargs : model_dir : str qui représente le dossier ou enregistrer les checkpoints lors du train du modèle\n
                   config : objet qui représente la configuration du Mask-RCNN à créer\n
                   reuse : bool qui correspond à réutiliser d'anciens poids (si True) ou non (si False)\n
                   model_rcnn : str qui représente le fichier à charger qui contient les poids si reuse=True
                   class_names : list qui représente les classes à chercher

        Returns
        -------
        model : model Mask-RCNN prêt à être entrainé
        """
        MODEL_DIR = kwargs['model_dir']
        config=kwargs['config']
        model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)
        self.config = config
        self.rcnn_mode = 'training'
        self.model_dir = MODEL_DIR
        assert config.NUM_CLASSES == len(kwargs['class_names']), "Il n'y a pas le même nombre de classes que de noms de classes"
        self.class_names = kwargs['class_names']
        if kwargs['reuse']==True: # Si on réutilise un modèle déjà entrainé
            model.load_weights(kwargs['model_rcnn'],by_name=True, exclude=[
                                    "mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
        return model
    
    def train(self, models_to_train='all', **kwargs):
        """
        Réalise l'entrainement des modèles spécifiés

        Parameters
        ----------
        models_to_train : str qui représente les modèles à entrainer. Doit être 'all', 'unet' ou 'rcnn'. Par défaut 'all'
        **kwargs : dir_train_unet : str qui représente le dossier contenant les images pour entrainer le U-Net (et faire la validation). Si 'all' ou 'unet'.\n
                   generator_unet : objet data_generator contenant les images pour entrainer le U-Net. Si 'all' ou 'unet'.\n
                   batch_size_unet : int qui représente le batch_size pour entrainer le U-Net. Si 'all' ou 'unet'.\n
                   epochs_unet : int qui représente le nombre d'epochs pour entrainer le U-Net. Si 'all' ou 'unet'.\n
                   dir_checkpoint_unet : str qui représente le dossier ou faire les checkpoints lors du train du U-Net. Si 'all' ou 'unet'.\n
                   dataset_train_rcnn : objet dataset à utiliser pour entrainer le Mask-RCNN. Si 'all' ou 'rcnn'.\n
                   dataset_val_rcnn : objet dataset à utiliser pour faire la validation du Mask-RCNN. Si 'all' ou 'rcnn'.\n
                   epochs_rcnn : int qui représente le nombre d'epochs pour entrainer le Mask-RCNN. Si 'all' ou 'rcnn'.\n
                   layers_rcnn : spécifie les couches à entrainer dans le Mask-RCNN. Doit être 'heads' ou 'all'. Mettre 'heads' dans le cas de transfer learning. Nécéssaire si 'all' ou 'rcnn'.

        Returns
        -------
        None.

        """
        assert models_to_train in ['all','unet','rcnn'], "models_to_train doit être 'all', 'unet' ou 'rcnn'"
        if models_to_train=='unet':
            batch_size = kwargs['batch_size_unet']
            timestep = self.timestep
            epochs = kwargs['epochs_unet']
            dir_ckpt = kwargs['dir_checkpoint_unet']
            if 'generator_unet' in kwargs.keys():
                gen = kwargs['generator_unet']
                self.unet = self.train_unet(generator=gen, batch_size=batch_size, timestep=timestep,
                                            epochs=epochs, dir_checkpoint=dir_ckpt)
            elif 'dir_train_unet' in kwargs.keys():
                direc_train = kwargs['dir_train_unet']
                self.unet = self.train_unet(dir_train=direc_train, batch_size=batch_size, timestep=timestep,
                                            epochs=epochs, dir_checkpoint=dir_ckpt)
        
        elif models_to_train=='rcnn':
            self.train_rcnn(dataset_train=kwargs['dataset_train_rcnn'],
                            dataset_val=kwargs['dataset_val_rcnn'],
                            epochs=kwargs['epochs_rcnn'],
                            layers=kwargs['layers_rcnn'])
        elif models_to_train=='all':
            batch_size = kwargs['batch_size_unet']
            timestep = self.timestep
            epochs = kwargs['epochs_unet']
            dir_ckpt = kwargs['dir_checkpoint_unet']
            if 'generator_unet' in kwargs.keys():
                gen = kwargs['generator_unet']
                self.unet = self.train_unet(generator=gen, batch_size=batch_size, timestep=timestep,
                                            epochs=epochs, dir_checkpoint=dir_ckpt)
            elif 'dir_train_unet' in kwargs.keys():
                direc_train = kwargs['dir_train_unet']
                self.unet = self.train_unet(dir_train=direc_train, batch_size=batch_size, timestep=timestep,
                                            epochs=epochs, dir_checkpoint=dir_ckpt)
            self.train_rcnn(dataset_train=kwargs['dataset_train_rcnn'],
                            dataset_val=kwargs['dataset_val_rcnn'],
                            epochs=kwargs['epochs_rcnn'],
                            layers=kwargs['layers_rcnn'])
    
    def train_unet(self,**kwargs):
        """
        Réalise l'entrainement du U-Net

        Parameters
        ----------
        **kwargs : dir_train : str qui représente le dossier contenant les images pour entrainer le U-Net (et faire la validation).\n
                   generator : object data_generator contenant les images (écrase dir_train)\n
                   dir_checkpoint : str qui représente le dossier ou faire les checkpoints lors du train.\n
                   batch_size : int qui représente le batch_size pour entrainer le U-Net.\n
                   epochs : int qui représente le nombre d'epochs pour entrainer le U-Net.

        Returns
        -------
        model U-Net entrainé

        """
        shape = self.shape
        batch_size = kwargs['batch_size']
        timestep = self.timestep
        epochs = kwargs['epochs']
        ckpt_dir = kwargs['dir_checkpoint']
        if 'generator' in kwargs.keys():
            gen = kwargs['generator']
            return unetlib.train(batch_size, timestep, epochs, shape, self.unet, ckpt_dir, generator=gen)
        elif 'dir_train' in kwargs.keys():
            direc_train = kwargs['dir_train']
            return unetlib.train(batch_size, timestep, epochs, shape, self.unet, ckpt_dir, nom_fichier=direc_train)
    
    def train_rcnn(self, **kwargs):
        """
        Réalise l'entrainement du Mask-RCNN

        Parameters
        ----------
        **kwargs : dataset_train : objet dataset à utiliser pour entrainer le Mask-RCNN.\n
                   dataset_val : objet dataset à utiliser pour faire la validation du Mask-RCNN.\n
                   epochs : int qui représente le nombre d'epochs pour entrainer le Mask-RCNN.\n
                   layers : spécifie les couches à entrainer dans le Mask-RCNN. Doit être 'heads' ou 'all'. Mettre 'heads' dans le cas de transfer learning.

        Returns
        -------
        None.

        """
        dataset_train = kwargs['dataset_train']
        dataset_val = kwargs['dataset_val']
        epochs = kwargs['epochs']
        layers = kwargs['layers']
        self.rcnn.train(dataset_train, dataset_val,
                        learning_rate=self.config.LEARNING_RATE,
                        epochs=epochs,
                        layers=layers)
    
    def convert_rcnn(self):
        """
        Convertit le Mask-RCNN du mode 'training' ou mode 'inference'

        Returns
        -------
        None.

        """
        assert self.rcnn_mode=='training', "Le Mask-RCNN est déjà en inférence"
        self.rcnn.keras_model.save_weights("weight_save_temp.h5")
        self.load_rcnn(model_rcnn="weight_save_temp.h5",config=self.config,class_names=self.class_names)
        os.remove("weight_save_temp.h5")
    
    def save(self,models_to_save='all',**kwargs):
        """
        Sauvegarde les modèles choisis

        Parameters
        ----------
        models_to_save : Spécifie le modèle à sauvegarder. Soit 'all', 'unet' ou 'rcnn'. Par défaut 'all'\n
        **kwargs : dir_unet : str à spécifier si models_to_save='all' ou 'unet'. Fichier destination de la sauvegarde du U-Net
                   dir_rcnn : str à spécifier si models_to_save='all' ou 'rcnn'. Fichier destination de la sauvegarde du Mask-RCNN

        Returns
        -------
        None.
        """
        assert models_to_save in ['all','unet','rcnn'],"models_to_save doit être 'all', 'unet' ou 'rcnn'"
        if models_to_save=='unet':
            self.unet.save(kwargs['dir_unet'])
        elif models_to_save=='rcnn':
            self.rcnn.keras_model.save_weights(kwargs['dir_rcnn'])
        elif models_to_save=='all':
            self.unet.save(kwargs['dir_unet'])
            self.rcnn.keras_model.save_weights(kwargs['dir_rcnn'])
    
    def evaluate(self, pred, gt, to_evaluate=['rect'], gt_type='instance'):
        """
        Fonction qui retourne la métrique choisie

        Parameters
        ----------
        to_evaluate : list des choses à évaluer. Parmi ['rect', 'masks', 'motion_mask', 'nb_pers']. Par défaut ['rect'].\n
        gt_type : type des vérités terrains. Soit 'instance' soit 'bool'. Par défaut 'instance'.\n
        pred : tuple (predictions, nombre de personnes). Résultat de la fonction predict.\n
        gt : tuple (gt, nombre de personnes).

        Returns
        -------
        None.

        """
        s = gt[0].shape        
        if np.array(pred[0]['masks']).shape[0]==0: # Si liste vide
            pred[0]['masks'] = np.expand_dims(np.zeros(s,dtype=np.bool),axis=-1)
        
        if len(pred[0]['masks'].shape)==3 and pred[0]['masks'].shape[-1]==0:
            pred[0]['masks'] = np.expand_dims(np.zeros(s,dtype=np.bool),axis=-1)

        if gt_type=='instance':
            gt_rois = metrics.compute_rect_with_masks(gt[0])
        else:
            gt_rois = None
        self.last_metrics = []
        for to_ev in to_evaluate:
            if to_ev=='rect':
                assert gt_type=='instance', "Pour l'évaluation sur les {}, il faut des gt de type 'instance'".format(to_ev)
                self.last_metrics.append(metrics.score_box(pred[0]['rois'],gt_rois,s))
            elif to_ev=='masks':
                assert gt_type=='instance', "Pour l'évaluation sur les {}, il faut des gt de type 'instance'".format(to_ev)
                self.last_metrics.append(metrics.score_mask(pred[0]['masks'],gt[0],pred[0]['rois'],gt_rois,s))
            elif to_ev=='motion_mask':
                pred2 = pred[0]['masks'][...,0]
                for i in range(pred[0]['masks'].shape[-1]):
                    pred2 = pred2 | pred[0]['masks'][...,i]
                if gt_type=='instance':
                    print('instance')
                    gt2 = gt[0][...,0]
                    for i in range(gt[0].shape[-1]):
                        gt2 = gt2 | gt[0][...,i]
                elif gt_type=='bool':
                    assert len(gt[0].shape)==2, "La GT doit avoir deux dimensions dans le cas de gt_type='bool'"
                    gt2 = gt[0]
                self.last_metrics.append((metrics.compute_mask_iou(gt2,pred2),metrics.conf(pred2,gt2)))
            elif to_ev=='nb_pers':
                self.last_metrics.append(gt[1]-pred[1])
        return self.last_metrics
    
    def visualize(self, im_orig, pred, models_used='all', viz=True):
        """
        Superpose la prédiction des deux réseaux à l'image originale et l'affiche

        Parameters
        ----------
        im_orig : image originale sur laquelle la prédiction a été faite (la dernière du TIMESTEP).\n
        pred : tuple (prediction, nombre de personnes)
        models_used : str qui représente les models à partir desquels la prédiction a été faite. Doit être 'all', 'unet' ou 'rcnn'. Par défaut 'all'.\n

        Returns
        -------
        im_masked : image avec la prédiction

        """
        
        assert models_used in ['all','unet','rcnn'],"models_used doit être 'all', 'unet' ou 'rcnn'."
        
        if models_used=='all':
            r = pred[0]
            list_colors = [(255,0,0),(255,255,0),(23,255,0),(0,0,255),
                           (255,0,255),(0,0,0),(255,255,255),
                           (162,0,255),(255,143,0),(0,255,201),(100,100,100)]
            im_masked = display_instances(im_orig, r['rois'], r['masks'], r['class_ids'],
                                          self.class_names, r['scores'], pred[1], list_colors)
            if viz:
                plt.figure()
                plt.imshow(im_masked)
            return im_masked
        elif models_used=='rcnn':
            r = pred
            list_colors = [(255,0,0),(255,255,0),(23,255,0),(0,0,255),
                           (255,0,255),(0,0,0),(255,255,255),
                           (162,0,255),(255,143,0),(0,255,201),(100,100,100)]
            im_masked = display_instances(im_orig, r['rois'], r['masks'], r['class_ids'],
                                          self.class_names, r['scores'], None, list_colors)
            if viz:
                plt.figure()
                plt.imshow(im_masked)
            return im_masked
        elif models_used=='unet':
            plt.figure()
            plt.subplot(121)
            plt.imshow(im_orig)
            plt.title('Image originale')
            plt.subplot(122)
            plt.imshow(pred)
            plt.title('Prediction')
            
            return pred

def random_colors(N,list_colors):
    """
    Renvoie N couleurs aléatoires

    Parameters
    ----------
    N : nombre de couleurs à sélectionner
    list_colors : list des couleurs disponibles

    Returns
    -------
    colors : list des couleurs sélectionnées

    """
    np.random.seed(1)
    colors = []
    for i in range(N):
        # colors.append(list_colors[np.random.randint(0,len(list_colors))])
        colors.append(list_colors[i%len(list_colors)])
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    Applique les masques à l'image

    Parameters
    ----------
    image : image originale
    mask : masques
    color : couleurs
    alpha : Gère la transparence. Par défaut 0.5.

    Returns
    -------
    image : image avec les masques

    """
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores, nb_pers, list_colors):
    """
    Retourne l'image finale
    """
    image = np.copy(image)
    if nb_pers==0:
        image = cv2.putText(
            image, "Nb objects detected: {}".format(nb_pers), (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1
        )
        return image
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances, list_colors)

    if not n_instances:
        a=None
        # print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    if nb_pers!=None:
        image = cv2.putText(image, "Nb objects detected: {}".format(nb_pers),
                            (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)

    return image