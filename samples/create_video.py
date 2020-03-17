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
import coco
from utils import barre, to_excel, sorted_nicely, get_new_colors
from metrics import f1_score

class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9
    NUM_CLASSES = 1 + 2 # Background + Person + Cars
    NAME = 'coco_person_car'
    STEPS_PER_EPOCH = 2000

config = InferenceConfig()
config.display()

class_names = ['BG','person','car']

s=(480,720,3) # Shape d'entrée du U-Net

MODELS_DIR = os.path.join(ROOT_DIR,"models/")
MODELS_UNET_DIR = os.path.join(MODELS_DIR,"U-Net/")
MODELS_MASKRCNN_DIR = os.path.join(MODELS_DIR,"Mask-RCNN/")
DATASET_DIR = os.path.join(ROOT_DIR,"../Datasets/")

TIMESTEP = 5

#%% Création du détecteur
smartmov = SmartMov()

#%% Chargement des deux réseaux
smartmov.load_models('rcnn', model_rcnn=MODELS_MASKRCNN_DIR+'mask_rcnn_person_car_v2.h5',
                     config=config, class_names=class_names)
smartmov.load_models('unet', model_unet=MODELS_UNET_DIR+"unet_hall.h5", shape_unet=s, timestep=TIMESTEP)

#%% Load images

IMAGES_DIR = os.path.join(DATASET_DIR,"PETS2006_organized2/input/") # Dossier des images à évaluer (doivent correspondre au U-Net chargé avant)
GT_DIR = os.path.join(DATASET_DIR,"PETS2006_organized2/groundtruth/") # Dossier de la groundtruth correspondante à ces images

input_list = sorted_nicely(glob.glob(IMAGES_DIR+"*.jpg")) # Liste ordonnée des images à évaluer
gt_list = sorted_nicely(glob.glob(GT_DIR+"*.png")) # Liste ordonnée des groundtruth

#%% Création groundtruth pour évaluer la correspondance des classes
pets_img=[15,113,16,7]
pets_nb=[1,3,2,1]

nb_occurence_gt_pets=[]
for i in range (len(pets_img)):
    for j in range (pets_img[i]):
        nb_occurence_gt_pets.append(([1,2],[pets_nb[i],0]))

gt_classes = nb_occurence_gt_pets

#%% Création de la vidéo

RESULTS_DIR = os.path.join(ROOT_DIR,"results/")
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
    print("Dossier RESULTS non existant. Création réussie.")

VIDEOS_DIR = os.path.join(RESULTS_DIR,"videos/") # Dossier pour contenir les vidéos
if not os.path.exists(VIDEOS_DIR):
    os.mkdir(VIDEOS_DIR)
    print("Dossier VIDEOS non existant. Création réussie.")

PHOTOS_DIR = os.path.join(RESULTS_DIR,"photos/") # Dossier pour enregsitrer les images séparément (inutile ici, argument photo_location de multi_display)

# Création d'une vidéo à partir des images et des groudntruth données, avec affichées les métriques spécifiées
# Cette vidéo sera enregistrée à l'emplacement video_location, aura pour nom name_video et les fps sont spécifiés
metric = smartmov.multi_display(input_list,gt_list,gt_classes=gt_classes,display=['nb_obj','temps','scores','num_im'],
                                metrics_to_display=['iou','f1'], metrics_to_compute=['iou','conf','f1','class'],
                                gt_type='bool',couleur_texte=(255,255,0),video_location=VIDEOS_DIR,name_video="PETS",
                                fps_video=[15.0,20.0])

#%% Excel creation
RESULTS_DIR = os.path.join(ROOT_DIR,"results/")
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
    print("Dossier RESULTS non existant. Création réussie.")

EXCEL_DIR = os.path.join(RESULTS_DIR,"excel/")
if not os.path.exists(EXCEL_DIR):
    os.mkdir(EXCEL_DIR)
    print("Dossier EXCEL non existant. Création réussie.")

EXCEL_FILE = os.path.join(EXCEL_DIR,"res_example.xlsx")

smartmov.create_excel(metric,metrics_to_display=['iou','conf','f1','class'],nom_fichier=EXCEL_FILE,nom_feuille="Results PETS")