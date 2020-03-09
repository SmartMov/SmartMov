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
from utils import barre, to_excel
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

#%% Load Mask-RCNN
smartmov.load_models('rcnn', model_rcnn=MODELS_MASKRCNN_DIR+'mask_rcnn_person_car_v2.h5',
                     config=config, class_names=class_names)
smartmov.load_models('unet', model_unet=MODELS_UNET_DIR+"unet_hall.h5", shape_unet=s, timestep=TIMESTEP)

#%% Evaluation

list_inp = glob.glob(os.path.join(DATASET_DIR,"PETS2006_organized/Validation/input/*.jpg")) # Dossier des inputs à évaluer
list_gt = glob.glob(os.path.join(DATASET_DIR,"PETS2006_organized/Validation/groundtruth/*.png")) # Dossier des GT correspondantes

mat_conf = []
iou = []
f1 = []
vid = []

for i,im in enumerate(list_inp):
    if i==len(list_inp)-TIMESTEP:
        break
    inp = []
    for im2 in list_inp[i:i+TIMESTEP]:
        inp.append(plt.imread(im2))
    inp = np.array(inp)
    gt0 = np.array(PIL.Image.open(list_gt[i+TIMESTEP-1]))
    gt0[gt0<=90] = 0
    gt0[gt0>0] = 1
    gt0 = gt0.astype(np.bool)
    t0=time.time()
    pred = smartmov.predict(inp)
    tf=time.time()-t0
    gt = (gt0,0)
    evaluat = smartmov.evaluate(pred,gt,to_evaluate=['motion_mask'], gt_type='bool')
    mat_conf.append(evaluat[0][1])
    f1.append(f1_score(mat_conf[-1]))
    iou.append(evaluat[0][0])
    im_vid = smartmov.visualize(inp[-1],pred,viz=False)
    im_vid = cv2.putText(im_vid, "Inference time: {:.2f}s".format(tf),
                        (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
    im_vid = cv2.putText(im_vid, "Number: {}".format(i+1),
                        (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
    im_vid = cv2.putText(im_vid, "Score: IoU {:.2f}   F1 {:.2f}".format(iou[-1],f1[-1]),
                        (im_vid.shape[1]-150, im_vid.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0), 1)
    vid.append(im_vid)
    barre(i,len(list_inp)-TIMESTEP,np.mean(iou))

print()
mat_conf_moy = np.mean(mat_conf,axis=0)

#%% Excel creation
RESULTS_DIR = os.path.join(ROOT_DIR,"results/")
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
    print("Dossier RESULTS non existant. Création réussie.")

EXCEL_DIR = os.path.join(RESULTS_DIR,"excel/")
if not os.path.exists(EXCEL_DIR):
    os.mkdir(EXCEL_DIR)
    print("Dossier EXCEL non existant. Création réussie.")

EXCEL_FILE = os.path.join(EXCEL_DIR,"res.xlsx")

to_excel(EXCEL_FILE,"Results hall",mat_conf_moy,iou,f1)

#%% Vidéo

shape_vid=vid[0].shape

VIDEOS_DIR = os.path.join(RESULTS_DIR,"videos/")
if not os.path.exists(VIDEOS_DIR):
    os.mkdir(VIDEOS_DIR)
    print("Dossier VIDEOS non existant. Création réussie.")

FILE_NAME = os.path.join(VIDEOS_DIR,"hall.avi")

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(FILE_NAME,fourcc, 25.0, (shape_vid[1],shape_vid[0]))

for cpt,i in enumerate(range(len(vid))):
    out.write(cv2.cvtColor(vid[i],cv2.COLOR_RGB2BGR))
out.release()