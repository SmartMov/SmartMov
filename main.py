import numpy as np
import matplotlib.pyplot as plt
from SmartMov import SmartMov
import os
import sys
import PIL
import tensorflow as tf
from data_generator import *
import cv2
import glob
import time

ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
import coco
import davis

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

# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                    'bus', 'train', 'truck', 'boat', 'traffic light',
#                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                    'teddy bear', 'hair drier', 'toothbrush']

s=(480,720,3)

MODELS_DIR = os.path.abspath("models/")
MODELS_UNET_DIR = os.path.join(MODELS_DIR,"U-Net/")
MODELS_MASKRCNN_DIR = os.path.join(MODELS_DIR,"Mask-RCNN/")

#%% Création du détecteur
smartmov = SmartMov()

#%% Load Mask-RCNN
# smartmov.load_models('all',model_unet="temp_unet.h5",shape_unet=s, timestep=5,
#                          model_rcnn='mask_rcnn_only_person.h5',config=config,class_names=class_names)
# smartmov.load_models('unet',model_unet="temp_unet.h5",shape_unet=s, timestep=5)
smartmov.load_models('rcnn', model_rcnn=MODELS_MASKRCNN_DIR+'mask_rcnn_person_car_v2.h5', config=config, class_names=class_names)
smartmov.load_models('unet', model_unet=MODELS_UNET_DIR+"unet_skating.h5", shape_unet=s, timestep=5)

#%% Create Mask-RCNN
smartmov.create_model('rcnn',model_dir="logs/",config=config,reuse=True,
                          model_rcnn=MODELS_MASKRCNN_DIR+"mask_rcnn_coco.h5",class_names=class_names)

#%% Load dataset train

DATASET_DIR = os.path.abspath("../Datasets/")

CDNET_DIR = os.path.join(DATASET_DIR,"CD-NET_2014/")
dirs = glob.glob(CDNET_DIR+"*/")

gen = DataGenerator(5)
# gen.load_data(dirs,nb_images=5000)
gen.load_data_by_files('save_unet_all_v2')

COCO_DIR = os.path.join(DATASET_DIR,"coco2014/")

coco_train = coco.CocoDataset()
coco_train.load_coco(COCO_DIR, "train", year='2014', auto_download=True, class_ids=[1,3])
coco_train.prepare()

coco_val = coco.CocoDataset()
cc = coco_val.load_coco(COCO_DIR, "minival", year='2014', auto_download=True, return_coco=True, class_ids=[1,3])
coco_val.prepare()

# DAVIS_DIR = os.path.join(DATASET_DIR,"Davis 2017/")

# davis_train = davis.DavisDataset()
# davis_train.load_davis(DAVIS_DIR,subset='train',select_class=['person'],personal_ids=[1])
# davis_train.prepare()

# davis_val = davis.DavisDataset()
# davis_val.load_davis(DAVIS_DIR,subset='val',select_class=['person'],personal_ids=[1])
# davis_val.prepare()

#%% Train Mask-RCNN
smartmov.train('rcnn',dataset_train_rcnn=coco_train,dataset_val_rcnn=coco_val,
                    epochs_rcnn=3, layers_rcnn='heads')

#%% Convert to inference mode
smartmov.convert_rcnn()

#%% Create U-Net
smartmov.create_model('unet',shape_unet=s,timestep=5)

#%% Train U-Net
# smartmov.train('unet',dir_train_unet=dirs, generator_unet=gen,
#                     batch_size_unet=1,epochs_unet=3)

# smartmov.save(models_to_save='unet',dir_unet='unet_all_v2.h5')

train_dir = os.path.join(DATASET_DIR,"winterDriveway/")

smartmov.train('unet',dir_train_unet=train_dir, dir_checkpoint_unet="save_unet_winterDriveway/",
                    batch_size_unet=1,epochs_unet=3)

smartmov.save(models_to_save='unet',dir_unet='unet_winterDriveway.h5')

#%% Model show convolution maps

displayer = tf.keras.models.Model(inputs=smartmov.unet.inputs,outputs=smartmov.unet.layers[23].output)

im,tar = gen.test_batch(50)

conv_map = displayer.predict(im)

for i in range(conv_map.shape[-1]):
    plt.subplot(4,3,i+1)
    plt.imshow(conv_map[0,...,i])
    plt.title(f"Map {i+1}")
    plt.colorbar()


#%% Prédiction

# smartmov.load_models('all',model_unet="unet1.h5",shape_unet=s,
#                          timestep=5, model_rcnn='mask1.h5',config=config,class_names=class_names)

im = []
for f in glob.glob("test_images/skating/*.jpg"):
    im.append(plt.imread(f))

im = np.array(im)

im,tar = gen.test_batch(50)

img = plt.imread("test_images/winterDriveway.jpg")

# im = []
# for i in range(5):
#     im.append(davis_val.load_image(45+i))
    
# im = np.array(im)

pred = smartmov.predict(im,models_to_use='all')
pred_unet = smartmov.predict(im,models_to_use='unet')
pred_rcnn = smartmov.predict(im,models_to_use='rcnn')

plt.figure()
plt.subplot(131)
plt.imshow(im[-1])
plt.title('Entrée')
# plt.subplot(132)
# plt.imshow(tar[0,...,0])
# plt.title("Target")
plt.subplot(133)
plt.imshow(pred_unet)
plt.title('Prédiction')

gt0 = np.array(PIL.Image.open("test_images/skating/gt001939.png"))
gt0[gt0>0] = 1
gt0 = gt0.astype(np.bool)

# gt0 = davis_val.load_mask(49)[0]

gt = (gt0,1)

#%% Visualize
smartmov.visualize(im[-1],pred)
smartmov.visualize(im[-1],(pred_rcnn,2))
smartmov.visualize(img,(pred_rcnn,2))

#%% Save
# smartmov.save(models_to_save='all',dir_unet="unet1.h5",dir_rcnn="mask1.h5")

#%% Evaluation
evaluat = smartmov.evaluate(pred,gt,to_evaluate=['motion_mask','nb_pers'], gt_type='bool')
print(evaluat)

#%% Trucs de Mael
# list_inp = glob.glob(os.path.join(DATASET_DIR,"PETS2006_organized/Train/input/*.jpg"))
list_inp = glob.glob(os.path.join(DATASET_DIR,"skating_organized/input/*.jpg"))
# list_gt = glob.glob(os.path.join(DATASET_DIR,"PETS2006_organized/Train/groundtruth/*.png"))
list_gt = glob.glob(os.path.join(DATASET_DIR,"skating_organized/groundtruth/*.png"))

def barre(i,nb_images,score):
    nb_barres_visu = 25
    prctage = i/nb_images
    nb_barres = np.floor(prctage * nb_barres_visu).astype(np.int32)
    chaine = ""
    for j in range(nb_barres_visu):
        if j<=nb_barres:
            chaine+="#"
        else:
            chaine+='-'
    print("\rImage {}/{} : [{}] Score = {:.3f}".format(i+1,nb_images,chaine,np.mean(score)),end='')

def f1_score(mat):
    if 2*mat[0,0]+mat[1,0]+mat[0,1]==0:
        return 1.0
    return 2*mat[0,0]/(2*mat[0,0]+mat[1,0]+mat[0,1])

mat_conf = []
iou = []
f1 = []
nb_pers = []
vid = []

for i,im in enumerate(list_inp):
    if i==len(list_inp)-5:
        break
    inp = []
    for im2 in list_inp[i:i+5]:
        inp.append(plt.imread(im2))
    inp = np.array(inp)
    gt0 = np.array(PIL.Image.open(list_gt[i+4]))
    gt0[gt0<=90] = 0
    gt0[gt0>0] = 1
    gt0 = gt0.astype(np.bool)
    t0=time.time()
    pred = smartmov.predict(inp)
    tf=time.time()-t0
    gt = (gt0,0)
    evaluat = smartmov.evaluate(pred,gt,to_evaluate=['motion_mask','nb_pers'], gt_type='bool')
    mat_conf.append(evaluat[0][1])
    f1.append(f1_score(mat_conf[-1]))
    iou.append(evaluat[0][0])
    nb_pers.append(-evaluat[1])
    im_vid = smartmov.visualize(inp[-1],pred,False)
    im_vid = cv2.putText(im_vid, "Inference time: {:.2f}s".format(tf),
                        (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
    im_vid = cv2.putText(im_vid, "Number: {}".format(i+1),
                        (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
    im_vid = cv2.putText(im_vid, "Score: IoU {:.2f}   F1 {:.2f}".format(iou[-1],f1[-1]),
                        (im_vid.shape[1]-150, im_vid.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0), 1)
    vid.append(im_vid)
    # plt.imshow(im_vid)
    barre(i,len(list_inp)-5,np.mean(iou))
    # break

#%% Vidéo

s=vid[0].shape
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter('skating2.avi',fourcc, 30.0, (s[1],s[0]))

for cpt,i in enumerate(range(len(vid))):
    out.write(vid[i])
out.release()

#%% Nombre de pers
nb = [1,3]
ind_nb_pers_gt = [102,199]
nb_pers_gt = []
for i in range(len(list_inp)-5):
    nb_pers_gt.append(nb[np.sum(np.array(ind_nb_pers_gt)<i)])

nb_pers2 = np.array(nb_pers)
nb_pers_gt = np.array(nb_pers_gt)

conf_nb_pers = np.zeros(shape=(np.max(nb_pers2)+1,np.max(nb_pers_gt)+1))
for a,b in zip(nb_pers2,nb_pers_gt):
    conf_nb_pers[a,b]+=1