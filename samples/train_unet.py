import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import PIL
import time
import cv2

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from SmartMov import SmartMov
from data_generator import DataGenerator
from utils import sorted_nicely

class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85
    NUM_CLASSES = 1 + 2 # Background + Person + Cars
    NAME = 'coco_person_car'
    STEPS_PER_EPOCH = 2000 # Nombre d'images dans la datset à utiliser pour l'entrainement

config = InferenceConfig()
config.display()

class_names = ['BG','person','car']

s=(480,720,3)

MODELS_DIR = os.path.join(ROOT_DIR,"models/")
MODELS_UNET_DIR = os.path.join(MODELS_DIR,"U-Net/")
MODELS_MASKRCNN_DIR = os.path.join(MODELS_DIR,"Mask-RCNN/")
DATASET_DIR = os.path.join(ROOT_DIR,"../Datasets/")

TIMESTEP = 5

#%% Création du détecteur
smartmov = SmartMov()

#%% Chargement du Mask-RCNN
smartmov.load_models('rcnn', model_rcnn=MODELS_MASKRCNN_DIR+'mask_rcnn_person_car_v2.h5', config=config, class_names=class_names)

#%% Création du U-Net
smartmov.create_model('unet',shape_unet=s,timestep=TIMESTEP)

#%% Première manière d'entrainer (avec un objet DataGenerator qui peut être appliqué à plusieurs dossiers)

CDNET_DIR = os.path.join(DATASET_DIR,"CD-NET_2014/")
dirs = glob.glob(CDNET_DIR+"*/") # Liste de tous les dossiers à utiliser pour l'entrainement

gen = DataGenerator(timestep=TIMESTEP) # Création du DataGenerator
gen.load_data(dirs,nb_images=1500) # Chargement des images à utiliser

smartmov.train('unet', generator_unet=gen, dir_checkpoint_unet=os.path.join(MODELS_UNET_DIR,"checkpoint/"),
               batch_size_unet=1,epochs_unet=2) # Entrainement

smartmov.save(models_to_save='unet',dir_unet=os.path.join(MODELS_UNET_DIR,'unet_example_generator.h5')) # Sauvegarde du modèle

#%% Seconde manière d'entrainer (avec un seul dossier)

train_dir = os.path.join(ROOT_DIR,"dataset_train/") # Dossier contenant les images et les groundtruth pour l'entrainement

smartmov.train('unet',dir_train_unet=train_dir, dir_checkpoint_unet=os.path.join(MODELS_UNET_DIR,"checkpoint/"),
                    batch_size_unet=1,epochs_unet=2) # Entrainement

smartmov.save(models_to_save='unet',dir_unet=os.path.join(MODELS_UNET_DIR,'unet_example_directory.h5')) # Sauvegarde du modèle

#%% Prédiction

IMAGES_DIR = os.path.join(ROOT_DIR,"dataset_test/input/") # Dossier des images à prédire (doivent correspondre au U-Net entrainé avant)

input_list = sorted_nicely(glob.glob(IMAGES_DIR+"*.jpg")) # Liste ordonnée des images à évaluer

im = []
for f in input_list:
    im.append(plt.imread(f)) # Chargement des images
im = np.array(im)

pred = smartmov.predict(im,models_to_use='all') # pred contiendra (prediction globale, nombre d'objets détectés)
pred_unet = smartmov.predict(im,models_to_use='unet') # pred_unet contiendra uniquement un masque binaire
pred_rcnn = smartmov.predict(im,models_to_use='rcnn') # pred_rcnn contiendra la prediction du Mask-RCNN

#%% Visualisation
smartmov.visualize(im[-1],pred)
smartmov.visualize(im[-1],pred_rcnn,models_used='rcnn')
smartmov.visualize(im[-1],pred_unet,models_used='unet')
