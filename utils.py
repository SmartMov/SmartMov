import numpy as np
import xlwings as wg
import xlwings.constants as c
from xlwings.utils import rgb_to_int as rgb
import matplotlib.pyplot as plt
import os
from metrics import f1_score

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
   
def to_excel(nom_fichier,nom_sheet,confu,iou,f1):

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