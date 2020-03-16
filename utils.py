import numpy as np
import xlwings as wg
import xlwings.constants as c
from xlwings.utils import rgb_to_int as rgb
import matplotlib.pyplot as plt
import os
from metrics import f1_score
import re

def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

def inser_titre(cell,nom,sh):
    
    sh.range(cell).value = nom
    sh.range(cell).api.Font.Color = rgb((0,0,255))
    sh.range(cell).api.Font.Size = 20
    sh.range(cell).api.Font.Bold = 1

def inser_soustitre(cell,nom,sh):
    
    sh.range(cell).value = nom
    sh.range(cell).api.Font.Color = rgb((0,0,0))
    sh.range(cell).api.Font.Size = 15
    sh.range(cell).api.Font.Bold = 1
    
def inser_val_matrice(cell,val,sh):
    
    sh.range(cell).value = val
    sh.range(cell).api.Font.Color = rgb((0,0,0))
    sh.range(cell).api.Font.Size = 13
    sh.range(cell).api.Font.Bold = 0
    sh.range(cell).api.Borders.Color=rgb((0,0,255))
    sh.range(cell).api.HorizontalAlignment = c.Constants.xlCenter
    sh.range(cell).api.VerticalAlignment = c.Constants.xlCenter
    
def inser_val(cell,val,sh):
    
    sh.range(cell).value = val
    sh.range(cell).api.Font.Color = rgb((0,0,0))
    sh.range(cell).api.Font.Size = 13
    sh.range(cell).api.Font.Bold = 0
    sh.range(cell).api.HorizontalAlignment = c.Constants.xlCenter
    sh.range(cell).api.VerticalAlignment = c.Constants.xlCenter

def inser_val_it(cell,val,sh):
    
    sh.range(cell).value = val
    sh.range(cell).api.Font.Color = rgb((0,0,0))
    sh.range(cell).api.Font.Size = 10
    sh.range(cell).api.Font.Bold = 0
    sh.range(cell).api.Font.Italic = 1
    sh.range(cell).api.HorizontalAlignment = c.Constants.xlCenter
    sh.range(cell).api.VerticalAlignment = c.Constants.xlCenter
    
def inser_fig(fig,name,left,top,width,height,sh):

    sh.pictures.add(fig,name=name,left=left,top=top,width=width,height=height)
   
def to_excel(nom_fichier,nom_sheet,confu,iou_inp,f1_inp):
    
    if not(isinstance(iou_inp,list)):
        iou = [iou_inp]
    else:
        iou = iou_inp
    
    if not(isinstance(f1_inp,list)):
        f1 = [f1_inp]
    else:
        f1 = f1_inp
    
    iou_mean=np.mean(iou)
    f1_mean=f1_score(confu)
    
    fig_IOU=plt.figure()
    plt.title("IoU")
    plt.plot(iou)
    plt.xlabel("Frames")
    plt.ylabel("IoU")
    
    fig_F1=plt.figure()
    plt.title("F1-score")
    plt.plot(f1)
    plt.xlabel("Frames")
    plt.ylabel("F1-score")
    
    plt.close('all')
    
    # Création
    if os.path.isfile(nom_fichier):
        bk = wg.Book(nom_fichier)
        bool_new=False
    else:
        bk = wg.Book()
        bool_new = True
    
    # création de la feuille 
    s1 = bk.sheets.add(nom_sheet)
        
    inser_titre('A1','Confusion Matrix',s1)
    inser_titre('A9','F1 Score mean',s1)
    inser_titre('A12','IoU Score mean',s1)
    
    inser_val_matrice('B3',confu[0,0],s1)
    inser_val_matrice('B4',confu[1,0],s1)
    inser_val_matrice('C3',confu[0,1],s1)
    inser_val_matrice('C4',confu[1,1],s1)
    
    inser_soustitre('E2','Accuracy',s1)
    inser_val('E3',confu[0,0]/(confu[0,0]+confu[0,1]),s1)
    inser_val('E4',confu[1,1]/(confu[1,0]+confu[1,1]),s1)
    
    inser_soustitre('A6','Recall',s1)
    inser_val('B6',confu[0,0]/(confu[0,0]+confu[1,0]),s1)
    inser_val('C6',confu[1,1]/(confu[0,1]+confu[1,1]),s1)
    
    eps = 1.96*np.sqrt(f1_mean*(1-f1_mean)/len(f1))
    
    inser_val('E9',f1_mean,s1)
    inser_soustitre('C10','Interval',s1)
    inser_val_it('D10',f1_mean-eps,s1)
    inser_val_it('E10',f1_mean+eps,s1)
    
    eps = 1.96*np.sqrt(iou_mean*(1-iou_mean)/len(iou))
    
    inser_val('E12',iou_mean,s1)
    inser_soustitre('C13','Interval',s1)
    inser_val_it('D13',iou_mean-eps,s1)
    inser_val_it('E13',iou_mean+eps,s1)
    
    inser_fig(fig_IOU,'ACC',350,10,200,150,s1)
    inser_fig(fig_F1,'F1',350,160,200,150,s1)
    
    if bool_new:
        bk.save(nom_fichier)
    else:
        bk.save()

def barre(i,nb_images,score=None):
    nb_barres_visu = 25
    prctage = i/nb_images
    nb_barres = np.floor(prctage * nb_barres_visu).astype(np.int32)
    chaine = ""
    for j in range(nb_barres_visu):
        if j<=nb_barres:
            chaine+="#"
        else:
            chaine+='-'
    if score is not None:
        print("\rImage {}/{} : [{}] Score = {:.3f}".format(i+1,nb_images,chaine,np.mean(score)),end='')
    else:
        print("\rImage {}/{} : [{}]".format(i+1,nb_images,chaine),end='')

def gravity_center(rect):
    return ((rect[3]-rect[1])/2.),((rect[2]-rect[0])/2.)

def compute_distance(rect,former_rect):
    grav_rect = gravity_center(rect)
    dist_tab = []
    for i,form_rect in enumerate(former_rect):
        grav_rect_form = gravity_center(form_rect)
        dist = np.sqrt((grav_rect[0]-grav_rect_form[0])**2+(grav_rect[1]-grav_rect_form[1])**2) # Centre
        dist1 = np.sqrt((rect[0]-form_rect[0])**2+(rect[1]-form_rect[1])**2)
        dist2 = np.sqrt((rect[0]-form_rect[0])**2+(rect[3]-form_rect[3])**2)
        dist3 = np.sqrt((rect[2]-form_rect[2])**2+(rect[1]-form_rect[1])**2)
        dist4 = np.sqrt((rect[2]-form_rect[2])**2+(rect[3]-form_rect[3])**2)
        dist_tab.append(0.5*dist+0.5*(dist1+dist2+dist3+dist4))
    sorted_index = np.argsort(dist_tab)
    sorted_dist = np.sort(dist_tab)
    return sorted_index,sorted_dist

def get_nearest_rectangles(new_pred,former_pred,models_used):
    assert models_used in ['all','rcnn'], "models_used doit être 'all' ou 'rcnn'."
    
    if models_used=='all':
        former_rect = former_pred[0]['rois']
        new_rect = new_pred[0]['rois']
    else:
        former_rect = former_pred['rois']
        new_rect = new_pred['rois']
    
    if new_rect != [] and former_rect != []:
        index_distances = []
        distances = []
        for index,rect in enumerate(new_rect): # Pour chaque nouveau rectangle
            ind,dis = compute_distance(rect,former_rect)
            index_distances.append(ind)
            distances.append(dis)
        # On élimine les rectangles trop loin
        index_distances2 = []
        distances2 = []
        for ind,dis in zip(index_distances,distances):
            index_distances2.append(ind[:np.min([len(index_distances),2])])
            distances2.append(dis[:np.min([len(index_distances),2])])
        
        return index_distances2,distances2
    return None,None

def get_new_colors(last_preds,last_colors,models_used='all'):
    """
    Renvoie les couleurs pour la vidéo

    Parameters
    ----------
    last_preds : dernières prédictions du réseau [t-N,t-N+1,...,t]
    last_colors : dernières couleurs utilisées pour l'affichage des anciennes prédictions [t-N,...t-1]
    models_used : 'all' ou 'rcnn'. Indique le modèle utilisé pour les prédictions

    Returns
    -------
    color_final : couleurs à utiliser pour l'affichage de la prédiction

    """
    assert models_used in ['all','rcnn'], "models_used doit être 'all' ou 'rcnn'."
    new_pred=last_preds[-1]
    if models_used=='all':
        new_rect=new_pred[0]['rois']
    else:
        new_rect=new_pred['rois']
    former_preds=last_preds[:-1]
    nb_pred_new=np.array(new_rect).shape[0]
    nb_predictions_used = len(former_preds)
    assert nb_predictions_used==len(last_colors), "Nombre de couleurs et de prédictions antérieures différents"
    color_final = np.zeros((nb_pred_new))-1
    color_final = list(color_final)
    
    if nb_pred_new==0:
        return color_final
    
    already_used_color=[]
    
    for index,rect in enumerate(new_rect):
        for former_pred,former_color in zip(reversed(former_preds),reversed(last_colors)):
            if models_used=='all':
                lenu=len(former_pred[0]['rois'])
            else:
                lenu=len(former_pred['rois'])
                
            if lenu==0:
                find_color=False
                continue
            nearest_rect,distances = get_nearest_rectangles(new_pred,former_pred,models_used)
            if nearest_rect is None:
                find_color=False
                continue
            
            plus_proche = nearest_rect[index]
            if plus_proche.shape[0]==0: # Si rectangles trop loins
                color_final[index]=-1
                find_color=True
            else:
                for k,plus_proche_rect in enumerate(plus_proche):
                    if former_color[plus_proche_rect] in already_used_color: # Si le plus proche est déjà pris
                        find_color=False
                        continue
                    else:
                        if index_is_min(nearest_rect,distances,(index,k)): # Si distance min
                            color_final[index] = former_color[plus_proche_rect]
                            find_color=True
                            already_used_color.append(former_color[plus_proche_rect])
                            break
                        else:
                            find_color=False
                            continue
                if find_color==False: # Si pas de couleur pour ce rectangle, on cherche dans les prédictions d'avant
                    continue
                else: # Si couleur trouvée on arrête d'aller dans les autres prédictions
                    break
        if find_color==False: # Si même avec toutes les prédictions pas de couleur
            color_final[index] = -1
    return color_final
    
def index_is_min(index_distances,distances,location):
    indice = index_distances[location[0]][location[1]]
    position_indice=[]
    for i,ind in enumerate(index_distances):
        for j,e in enumerate(ind):
            if e==indice:
                position_indice.append((i,j)) # On a les coordonnées de l'indice qu'on cherche
    dist=[]
    for a,b in position_indice:
        dist.append(distances[a][b])
    if distances[location[0]][location[1]]<=np.min(dist):
        return True
    else:
        return False

def constants_image(img):
    s=img.shape
    taille=s[0]/600
    epaisseur=1
           
    if ((s[1]/s[0])>1.6) and (s[0]>300) :
        epaisseur = 3
       
    if (s[1]<250):
        taille=s[0]/400
    
    espace_ligne=int(taille*40)
    debut_gauche=int(taille*35)
    debut_bas=int(taille*35*5)
        
    return taille,epaisseur,espace_ligne,debut_gauche,debut_bas
    
    
    