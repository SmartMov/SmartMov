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
from utils import constants_image, get_new_colors, barre
import PIL

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
    
    def evaluate(self, pred, gt, models_used='all', metrics_to_compute=['f1'], gt_type='bool'):
        """
        Fonction qui retourne la métrique choisie

        Parameters
        ----------
        metrics_to_compute : list des choses à évaluer. Parmi ['iou','conf','f1','rect','masks']. Par défaut ['f1'].\n
        gt_type : type des vérités terrains. Soit 'instance' soit 'bool'. Par défaut 'bool'.\n
        pred : tuple (predictions, nombre de personnes). Résultat de la fonction predict.\n
        gt : groundtruth
        models_used : models utilisés pour faire la prédiction

        Returns
        -------
        last_metrics : résultat des métriques demandées

        """
        
        assert models_used in ['all','rcnn'],"models_used doit être 'all' ou 'rcnn'"
        for met in metrics_to_compute:
            assert met in ['iou','conf','f1','rect','masks'], "La métrique {} n'existe pas".format(met)
        
        if models_used=='rcnn':
            if pred['masks'] == []:
                nb_pers_aff = 0
            else:
                nb_pers_aff = pred['masks'].shape[-1]
            pred3 = (pred,nb_pers_aff)
        else:
            pred3=pred
        
        s = gt.shape     
        if np.array(pred3[0]['masks']).shape[0]==0: # Si liste vide
            pred3[0]['masks'] = np.expand_dims(np.zeros(s,dtype=np.bool),axis=-1)
        
        if len(pred3[0]['masks'].shape)==3 and pred3[0]['masks'].shape[-1]==0: # Cas particulier (bug du Mask-RCNN)
            pred3[0]['masks'] = np.expand_dims(np.zeros(s,dtype=np.bool),axis=-1)

        if gt_type=='instance':
            gt_rois = metrics.compute_rect_with_masks(gt)
        else:
            gt_rois = None
            
        self.last_metrics = []
        for met in metrics_to_compute:
            if met=='iou':
                pred2 = pred3[0]['masks'][...,0]
                for i in range(pred3[0]['masks'].shape[-1]):
                    pred2 = pred2 | pred3[0]['masks'][...,i]
                if gt_type=='bool':
                    assert len(gt.shape)==2, "La GT doit avoir deux dimensions dans le cas de gt_type='bool'"
                    gt2 = gt
                elif gt_type=='instance':
                    gt2 = gt[...,0]
                    for i in range(gt.shape[-1]):
                        gt2 = gt2 | gt[...,i]
                self.last_metrics.append(metrics.compute_mask_iou(gt2,pred2))
            elif met=='conf':
                pred2 = pred3[0]['masks'][...,0]
                for i in range(pred3[0]['masks'].shape[-1]):
                    pred2 = pred2 | pred3[0]['masks'][...,i]
                if gt_type=='bool':
                    assert len(gt.shape)==2, "La GT doit avoir deux dimensions dans le cas de gt_type='bool'"
                    gt2 = gt
                elif gt_type=='instance':
                    gt2 = gt[...,0]
                    for i in range(gt.shape[-1]):
                        gt2 = gt2 | gt[...,i]
                self.last_metrics.append(metrics.conf(pred2,gt2))
            elif met=='f1':
                pred2 = pred3[0]['masks'][...,0]
                for i in range(pred3[0]['masks'].shape[-1]):
                    pred2 = pred2 | pred3[0]['masks'][...,i]
                if gt_type=='bool':
                    assert len(gt.shape)==2, "La GT doit avoir deux dimensions dans le cas de gt_type='bool'"
                    gt2 = gt
                elif gt_type=='instance':
                    gt2 = gt[...,0]
                    for i in range(gt.shape[-1]):
                        gt2 = gt2 | gt[...,i]
                mat_conf=metrics.conf(pred2,gt2)
                self.last_metrics.append(metrics.f1_score(mat_conf))
            elif met=='rect':
                assert gt_type=='instance',"Pour l'évaluation sur les {}, il faut des gt de type 'instance'".format(met)
                self.last_metrics.append(metrics.score_box(pred3[0]['rois'],gt_rois,s))
            elif met=='masks':
                assert gt_type=='instance', "Pour l'évaluation sur les {}, il faut des gt de type 'instance'".format(met)
                self.last_metrics.append(metrics.score_mask(pred3[0]['masks'],gt,pred3[0]['rois'],gt_rois,s))
                
        return self.last_metrics
    
    def single_display(self,inp,predic=None,gt=None,temps_predic=0,display=['nb_obj'],metrics_to_display=None,gt_type=None,models_used='all',num_im=0,colors=None,couleur_texte=(255,0,0),return_colors=False,return_scores=False,return_pred=False):
        """
        Renvoie l'image traitée à partir d'une séquence de 5 images d'entrée ou de la prédiction déjà effectuée.        

        Paramètres
        ----------
        inp : numpy array de taille (TIMESTEP,H,W,3) de type uint8
              Contient les images à utiliser pour la prédiction. Sont nécéssaires même si la prédiction a été faite avant l'appel à cette fonction.
        predic : prédiction du réseau spécifié dans la variable models_used, optionnel
                 Nécéssaire si le traitement de l'image doit être fait à partir d'une prédiction déjà réalisée.
                 Par défaut None (signifie qu'elle ne doit pas être utilisée)
        gt : groundtruth utilisée pour faire l'évaluation, optionnel
             Nécéssaire si metrics_to_display n'est pas None.
             Par défaut None (pas besoin de groundtruth)
        temps_predic : int, optionnel (nécéssaire si predic!=None et si 'temps' est dans display)
                       Dans le cas ou la prédiciton a été faite auparavant (predic != None) et que l'on souhaite afficher le temps d'inférence sur l'image
                       Par défaut 0
        display : list, optionnel
                  Liste des informations à afficher sur l'image. Doit être parmi ['scores','temps','nb_obj','num_im'].
                  Les scores seront affichés en bas à droite et les autres informations en gaut à gauche
                  Si 'scores' est à afficher, alors metrics_to_display doit être différent de None
                  Par défaut ['nb_obj']
        metrics_to_display : list, optionnel
                             Liste des métriques à calculer et à afficher sur l'image. La matrice de confusion ne sera jamais affichée sur l'image mais peut être tout de même calculée pour la récupérer en sortie.
                             Doit être parmi ['iou','f1','conf','rect','masks']. Cette liste correspond à celle de la méthode evaluate. Les mêmes restrictions s'y appliquent.
                             Pour calculer 'rect' et 'masks', les groundtruth doivent être de type 'instance' (gt_type='instance').
                             Si metrics_to_display!=None, gt_type doit être spécifié
                             Par défaut None (signifie qu'aucun score n'est calculé)
        gt_type : string, optionnel
                  Nécéssaire si metrics_to_display!=None. Doit être parmi ['bool','instance'].
                  Par défaut None (aucun score n'est donc calculé)
        models_used : string, optionnel
                      Spécifié le modèle utilisé ou à utiliser pour réaliser la prédiciton. Doit être parmi ['all','rcnn'].
                      Par défaut 'all'
        num_im : int, optionnel
                 Nécéssaire si 'num_im' est dans display. Numéro de l'image à afficher
                 Par défaut 0
        colors : list, optionnel
                 Liste des couleurs à utiliser pour chacunes des prédictions. Si None, les couleurs seront aléatoires. Ne pas changer sauf si tracking.
                 Par défaut None
        couleur_texte : tuple de taille 3, optionnel
                        Couleur dans laquelle seront écrites les informations sur l'image
                        Par défaut (255,0,0)
        return_colors : bool, optionnel
                        Renvoie les couleurs utilisées pour l'affichage de l'image. Nécéssaire pour le tracking.
                        Par déafut False
        return_scores : bool, optionnel
                        Renvoie les scores calculés et spécifiés dans metrics_to_display. Les scores ne sont calculés que si 'scores' est dans display
                        Par défaut False
        return_pred : bool, optionnel
                      Renvoie la prédiction utilisée pour le traitement de l'image.
                      Par défaut False

        Sorties
        -------
        Dépend des entrées. L'image traitée sera toujours retournée. Dans l'ordre viennent ensuite les couelurs, les scores et la prédiction.

        """
        
        nb_obj,scores=False,False
        for disp in display:
            assert disp in ['scores','temps','num_im','nb_obj'], "{} n'est pas un paramètre accepté".format(disp)
            if disp=='nb_obj':
                nb_obj=True
            elif disp=='scores':
                scores=True
        assert models_used in ['all','rcnn'],"models_used doit être 'all' ou 'rcnn'"
        
        couleur=couleur_texte
        taille,epaisseur,offset_texte,debut_gauche,debut_bas = constants_image(inp[-1])
        
        if nb_obj:
            current_display = 1
        else:
            current_display = 0
        offset_horiz_scores = inp[-1].shape[1]-debut_bas
                
        if predic is None:
            t0=time.time()
            pred = self.predict(inp,models_to_use=models_used)
            tf=time.time()-t0
        else:
            pred=predic
            tf=temps_predic
        
        if return_colors:
            im_output,choosen_colors = self.visualize(inp[-1],pred,viz=False,models_used=models_used,
                                                      colors=colors,return_colors=return_colors,
                                                      disp_nb_obj=nb_obj,offset=debut_gauche,taille=taille,
                                                      epaisseur=epaisseur,couleur=couleur)
        else:
            im_output = self.visualize(inp[-1],pred,viz=False,models_used=models_used,
                                        colors=colors,return_colors=return_colors,
                                        disp_nb_obj=nb_obj,offset=debut_gauche,taille=taille,
                                        epaisseur=epaisseur,couleur=couleur)
        
        for disp in display:
            if disp=='scores':
                assert gt is not None, "Pas de groundtruth,donc les scores ne peuvent pas être calculés"
                assert metrics_to_display is not None, "Une liste des métriques à afficher doit être donnée"
                assert gt_type is not None, "Le type de groundtruth doit être spécifié (bool ou instance)"
                evaluat = self.evaluate(pred,gt,models_used,metrics_to_display, gt_type)
                nb_scores=0
                for i,met in enumerate(evaluat):
                    if metrics_to_display[i]=='iou':
                        im_output = cv2.putText(im_output, "IoU  {:.2f}".format(met),
                    	                        (offset_horiz_scores, im_output.shape[0]-nb_scores*offset_texte-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, epaisseur)
                        nb_scores+=1
                    elif metrics_to_display[i]=='f1':
                        im_output = cv2.putText(im_output, "F1   {:.2f}".format(met),
                    	                        (offset_horiz_scores, im_output.shape[0]-nb_scores*offset_texte-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, epaisseur)
                        nb_scores+=1
                    elif metrics_to_display[i]=='rect':
                        im_output = cv2.putText(im_output, "Rect {:.2f}".format(met),
                    	                        (offset_horiz_scores, im_output.shape[0]-nb_scores*offset_texte-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, epaisseur)
                        nb_scores+=1
                    elif metrics_to_display[i]=='masks':
                        im_output = cv2.putText(im_output, "Mask {:.2f}".format(met),
                    	                        (offset_horiz_scores, im_output.shape[0]-nb_scores*offset_texte-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, epaisseur)
                        nb_scores+=1
            elif disp=='temps':
                im_output = cv2.putText(im_output, "Inference time: {:.2f}s".format(tf),
                                        (10, current_display*offset_texte+debut_gauche), cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, epaisseur)
                current_display+=1
            elif disp=='num_im':
                im_output = cv2.putText(im_output, "Number: {}".format(num_im),
                                        (10, current_display*offset_texte+debut_gauche), cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, epaisseur)
                current_display+=1
    	
        list_return = [im_output]
        for i,e in enumerate([return_colors,return_scores,return_pred]):
            if e:
                if i==0:
                    list_return.append(choosen_colors)
                elif i==1:
                    if scores is False:
                        list_return.append(None)
                    else:
                        list_return.append(evaluat)
                elif i==2:
                    list_return.append(pred)
        return tuple(list_return)
    
    def multi_display(self,images,gt=None,display=['nb_obj'],metrics_to_display=None,gt_type=None,couleur_texte=(255,0,0),video_location=None,name_video='no_name',fps_video=20.0,photo_location=None,models_used='all',return_scores=True):
        """
        Renvoie les métriques demandées et crée une vidéo ou enregistre une série de photos traitées.

        Parameters
        ----------
        images : list
                 Liste contenant les chemins de toutes les images à traiter.
        gt : list, optionnel
             Liste contenant les chemins de toutes les groundtruth des images à traiter.
             TODO : Ne fonctionne que pour des groundtruth de type 'bool'. Mettre en place pour les groundtruth de type 'instance'
             L'ordre des éléments de cette liste doit correspondre à celui des éléments de la liste images et doit être de la même longueur.
             Par défaut None (signifie que les scores ne seront pas calculés)
        display : list, optionnel
                  Liste des informations à afficher sur l'image. Doit être parmi ['scores','temps','nb_obj','num_im'].
                  Les scores seront affichés en bas à droite et les autres informations en gaut à gauche
                  Si 'scores' est à afficher, alors metrics_to_display doit être différent de None
                  Par défaut ['nb_obj']
        metrics_to_display : list, optionnel
                             Liste des métriques à calculer et à afficher sur l'image. La matrice de confusion ne sera jamais affichée sur l'image mais peut être tout de même calculée pour la récupérer en sortie.
                             Doit être parmi ['iou','f1','conf','rect','masks']. Cette liste correspond à celle de la méthode evaluate. Les mêmes restrictions s'y appliquent.
                             Pour calculer 'rect' et 'masks', les groundtruth doivent être de type 'instance' (gt_type='instance').
                             Si metrics_to_display!=None, gt_type doit être spécifié
                             Par défaut None (signifie qu'aucun score n'est calculé)
        gt_type : string, optionnel
                  Nécéssaire si metrics_to_display!=None. Doit être parmi ['bool','instance'].
                  Par défaut None (aucun score n'est donc calculé)
        couleur_texte : tuple de taille 3, optionnel
                        Couleur dans laquelle seront écrites les informations sur l'image
                        Par défaut (255,0,0)
        video_location : string, optionnel
                         Nécéssaire si une vidéo est souhaitée. Coorespond au dossier dans lequel enregsitrer la vidéo.
                         Si None, aucune vidéo n'est crée.
                         Si à la fois video_location==None et photo_location==None, aucune image ne sera enregistrée en sortie.
                         Par défaut None
        name_video : string, optionnel
                     Nom du fichier de la vidéo. Par défaut "no_name.avi"
        fps_video : float, optionnel
                    Nombre de fps dans le vidéo crée. Par défaut 20.0.
        photo_location : string, optionnel
                         Nécéssaire si les photos sont souhaitées en .jpg en sortie. Coorespond au dossier dans lequel enregsitrer les photos.
                         Si None, aucune photo n'est enregistrée.
                         Si à la fois video_location==None et photo_location==None, aucune image ne sera enregistrée en sortie.
                         Par défaut None
        models_used : string, optionnel
                      Spécifié le modèle utilisé ou à utiliser pour réaliser la prédiciton. Doit être parmi ['all','rcnn'].
                      Par défaut 'all'
        return_scores : bool, optionnel
                        Renvoie les scores calculés et spécifiés dans metrics_to_display. Les scores ne sont calculés que si 'scores' est dans display
                        Par défaut True

        Returns
        -------
        met_final : None su return_scores==False. Sinon renvoie une liste des toutes les métriques demandées pour chacune des images traitées.

        """
        
        vid = []
        
        first_image = True
        new_color=None
        
        list_pred=[]
        list_color=[]
        
        nb_pred_tracking=10
        
        met_final = []
        for i in range(len(metrics_to_display)):
                met_final.append([])
        
        for i,im in enumerate(images):
            if i==len(images)-self.timestep:
                break
            inp = []
            for im2 in images[i:i+self.timestep]:
                inp.append(plt.imread(im2))
            inp = np.array(inp)
            if gt is not None:
                gt0 = np.array(PIL.Image.open(gt[i+self.timestep-1]))
                if gt0.dtype!=np.bool:
                    gt0[(gt0<255) & (gt0>1)] = 0
                    gt0[gt0>0] = 1
                    gt0 = gt0.astype(np.bool)
            
            t0=time.time()
            pred = self.predict(inp,models_to_use=models_used)
            tf=time.time()-t0
            
            if first_image==False:
                if len(list_pred)==nb_pred_tracking:
                    del list_pred[0]
                    del list_color[0]
            list_pred.append(pred)
            
            if first_image==False:
                new_color = get_new_colors(list_pred,list_color,models_used=models_used)
            
            im_vid,new_former_color,evaluat,pred = self.single_display(inp,predic=pred,gt=gt0, temps_predic=tf, display=display,
                                                                           metrics_to_display=metrics_to_display,gt_type=gt_type,
                                                                           models_used=models_used,num_im=i+1,colors=new_color,couleur_texte=couleur_texte,
                                                                           return_colors=True, return_scores=True,return_pred=True)
            for j in range(len(metrics_to_display)):
                met_final[j].append(evaluat[j])
                    
            list_color.append(new_former_color)
            
            vid.append(im_vid)
            barre(i,len(images)-self.timestep)
            first_image=False
        
        print()
        
        if video_location is not None: # Si on veut créer une vidéo
            shape_vid=vid[0].shape
            if not os.path.exists(video_location):
                os.mkdir(video_location)
                print("Dossier video_location non existant. Création réussie.")
            if name_video.split('.')[-1]!='avi':
                FILE_NAME = os.path.join(video_location,name_video+".avi")
            else:
                FILE_NAME = os.path.join(video_location,name_video)
            
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(FILE_NAME,fourcc, fps_video, (shape_vid[1],shape_vid[0]))
            
            for i in range(len(vid)):
                out.write(cv2.cvtColor(vid[i],cv2.COLOR_RGB2BGR))
                barre(i,len(vid))
            out.release()
        
        if photo_location is not None:
            if not os.path.exists(photo_location):
                os.mkdir(photo_location)
                print("Dossier photo_location non existant. Création réussie.")
            for i in range(len(vid)):
                image_temp=PIL.Image.fromarray(vid[i])
                image_temp.save(os.path.join(photo_location,"{}.jpg".format(i)))
                barre(i,len(vid))
        
        if return_scores:
            return met_final
        
            
            
    def visualize(self, im_orig, pred, models_used='all', colors=None, return_colors=False, viz=True, disp_nb_obj=True, offset=20, taille=0.5, epaisseur=1,couleur=(255,0,0)):
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
            im_masked,choosen_colors = display_instances(im_orig, r['rois'], r['masks'], r['class_ids'],
                                          self.class_names, r['scores'], pred[1], list_colors, input_colors=colors,
                                          disp_nb_obj=disp_nb_obj,offset=offset,taille=taille,epaisseur=epaisseur,couleur=couleur)
            if viz:
                plt.figure()
                plt.imshow(im_masked)
            if return_colors:
                return im_masked,choosen_colors
            else:
                return im_masked
        elif models_used=='rcnn':
            r = pred
            list_colors = [(255,0,0),(255,255,0),(23,255,0),(0,0,255),
                           (255,0,255),(0,0,0),(255,255,255),
                           (162,0,255),(255,143,0),(0,255,201),(100,100,100)]
            if r['masks'] == []:
                nb_pers_aff = 0
            else:
                nb_pers_aff = r['masks'].shape[-1]
            im_masked,choosen_colors = display_instances(im_orig, r['rois'], r['masks'], r['class_ids'],
                                          self.class_names, r['scores'], nb_pers_aff, list_colors, input_colors=colors,
                                          disp_nb_obj=disp_nb_obj,offset=offset,taille=taille,epaisseur=epaisseur,couleur=couleur)
            if viz:
                plt.figure()
                plt.imshow(im_masked)
            if return_colors:
                return im_masked,choosen_colors
            else:
                return im_masked
        elif models_used=='unet':
            if viz:
                plt.figure()
                plt.subplot(121)
                plt.imshow(im_orig)
                plt.title('Image originale')
                plt.subplot(122)
                plt.imshow(pred)
                plt.title('Prediction')
            
            return pred

def random_colors(N,list_colors,really_random=False):
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
    colors = []
    if really_random:
        colors= [tuple(255 * np.random.rand(3)) for _ in range(N)]
    else:
        colors=[]
        for i in range(N):
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

def display_instances(image, boxes, masks, ids, names, scores, nb_pers, list_colors, input_colors=None, disp_nb_obj=True, offset=20,taille=0.5,epaisseur=1,couleur=(255,0,0)):
    """
    Retourne l'image finale
    """
    image = np.copy(image)
    if nb_pers==0:
        if disp_nb_obj:
            image = cv2.putText(
                image, "Nb objects detected: {}".format(nb_pers), (10, offset), cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, epaisseur
            )
        return image,[]
    n_instances = boxes.shape[0]
    if input_colors is not None:
        colors = input_colors
    else:
        colors = random_colors(n_instances, list_colors)

    if not n_instances:
        a=None
        # print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    
    for i, color in enumerate(colors):
        if color==-1:
            color=random_colors(1, list_colors, True)[0]
            colors[i]=color
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
            image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, taille, color, epaisseur
        )
    if nb_pers!=None:
        if disp_nb_obj:
            image = cv2.putText(image, "Nb objects detected: {}".format(nb_pers),
                                (10, offset), cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, epaisseur)

    return image,colors