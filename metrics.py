import numpy as np
import matplotlib.pyplot as plt
from mrcnn import visualize

def res_gt(dataset,indice_im,class_names,viz=False):
    """
    Renvoie le dictionnaire pour l'affichage des vérités terrains

    Parameters
    ----------
    dataset : dataset dans laquelle l'image est à trouver
    indice_im : Indice de l'image dans la dataset
    class_names : Nom des classes pour l'affichage
    viz : bool, True si les masques doivent être affichés

    Returns
    -------
    gt : dictionnaire des vérités terrains {'rois','masks','class_ids','scores'}
    """
    
    image = dataset.load_image(indice_im)
    gt_temp = dataset.load_mask(indice_im)
    gt = dict()
    gt['masks'] = gt_temp[0]
    gt['class_ids'] = gt_temp[1]
    gt['scores'] = np.ones_like(gt['class_ids']).astype(np.float32)
    gt['rois'] = np.zeros(shape=(gt['scores'].shape[0],4),dtype=np.int32)
    for i in range(gt['masks'].shape[-1]):
        mask = gt['masks'][:,:,i]
        if mask.max()==True:
            ones = np.where(mask)
            y1 = np.min(ones[0])
            y2 = np.max(ones[0])
            x1 = np.min(ones[1])
            x2 = np.max(ones[1])
            gt['rois'][i] = np.array([y1,x1,y2,x2])
        else:
            gt['rois'][i] = -1
    
    if(viz):
        plt.figure()
        plt.subplot(121)
        plt.axis('off')
        plt.imshow(image)
        plt.title('Image de base')
        ax = plt.subplot(122)
        visualize.display_instances(image, gt['rois'], gt['masks'], gt['class_ids'], 
                                    class_names, gt['scores'],ax=ax)
        plt.title('Groundtruth')
    return gt

def compare_gt_pred(model,dataset,indice_im,class_names,viz=False,mode="box"):
    """
    Comparaison visuelle et/ou métrique de la prédiciton et de la vérité terrain

    Parameters
    ----------
    model : modèle à utiliser pour la prédiction
    dataset : dataset dans laquelle prendre l'image à traiter
    indice_im : indice de l'image dans la dataset
    class_names : classes à afficher
    viz : bool, affiche ou non la comparaison visuelle
    mode : mode de comparaison pour la métrique (box pour les rectangles et mask pour les masques)

    Returns
    -------
    score : score de la métrique choisie
    """
    
    assert mode in ['box','mask'], "mode doit être soit 'box' soit 'mask'"
    
    gt = res_gt(dataset,indice_im,False)
    image = dataset.load_image(indice_im)
    pred = model.detect([image],verbose=0)[0]

    if viz:
        plt.figure()
        ax1 = plt.subplot(121)
        visualize.display_instances(image, pred['rois'], pred['masks'], pred['class_ids'], 
                                    class_names, pred['scores'],ax=ax1)
        plt.title('Prédiction')
        ax = plt.subplot(122)
        visualize.display_instances(image, gt['rois'], gt['masks'], gt['class_ids'], 
                                    class_names, gt['scores'],ax=ax)
        plt.title('Groundtruth')
    
    if mode=='box':
        return score_box(pred['rois'],gt['rois'],(image.shape[0],image.shape[1]))
    elif mode=='mask':
        return score_mask(pred['masks'],gt['masks'],pred['rois'],gt['rois'],(image.shape[0],image.shape[1]))

def score_box(rect_pred, rect_gt, s):
    iou = 0
    if np.sum(rect_gt[0]==-1)==4: # Si il n'y a pas de gt
        if rect_pred.shape[0]==0: # Si il n'y a pas de prédictions tout est bien
            return 1.0
        else:
            return compute_area_list_rect(list(rect_pred))/(s[0]*s[1])

    diff_nb_pred = rect_pred.shape[0] - rect_gt.shape[0]
    list_rect_pred = list(rect_pred)
    list_rect_gt = list(rect_gt)
    if diff_nb_pred==0: # Si il y a autant d'objets détectés que dans la GT
        i=0
        for i,pred in enumerate(rect_pred):
            y1,x1,y2,x2 = pred # Coordonnées du rectangle dans la prediction
            G = (np.mean([y1,y2]),np.mean([x1,x2]))
            dist_min = float('inf')
            for gt in rect_gt:
                y1_gt,x1_gt,y2_gt,x2_gt = gt
                G_gt = (np.mean([y1_gt,y2_gt]),np.mean([x1_gt,x2_gt]))
                dist = np.sqrt((G[0]-G_gt[0])**2+(G[1]-G_gt[1])**2)
                if dist<dist_min:
                    dist_min = dist
                    nearest_gt = gt
            iou += compute_rect_iou(pred,nearest_gt)
        return iou/(i+1)
    elif diff_nb_pred>0: # Si il y a plus de prédictions que de gt
        i=0
        for i,gt in enumerate(rect_gt):
            y1_gt,x1_gt,y2_gt,x2_gt = gt # Coordonnées du rectangle dans la prediction
            G_gt = (np.mean([y1_gt,y2_gt]),np.mean([x1_gt,x2_gt]))
            dist_min = float('inf')
            for index,pred in enumerate(list_rect_pred):
                y1,x1,y2,x2 = pred
                G = (np.mean([y1,y2]),np.mean([x1,x2]))
                dist = np.sqrt((G[0]-G_gt[0])**2+(G[1]-G_gt[1])**2)
                if dist<dist_min:
                    rem_index = index
                    dist_min = dist
                    nearest_pred = pred
            del list_rect_pred[rem_index]
            iou += compute_rect_iou(nearest_pred,gt)
        coef = compute_area_list_rect(list_rect_pred)/(s[0]*s[1])
        return iou/(i+1+diff_nb_pred*coef)
    else: # Si il y a plus de gt que de prédictions
        i=0
        for i,pred in enumerate(rect_pred):
            y1,x1,y2,x2 = pred # Coordonnées du rectangle dans la prediction
            G = (np.mean([y1,y2]),np.mean([x1,x2]))
            dist_min = float('inf')
            for index,gt in enumerate(list_rect_gt):
                y1_gt,x1_gt,y2_gt,x2_gt = gt
                G_gt = (np.mean([y1_gt,y2_gt]),np.mean([x1_gt,x2_gt]))
                dist = np.sqrt((G[0]-G_gt[0])**2+(G[1]-G_gt[1])**2)
                if dist<dist_min:
                    rem_index = index
                    dist_min = dist
                    nearest_gt = gt
            del list_rect_gt[rem_index]
            iou += compute_rect_iou(pred,nearest_gt)
        coef = compute_area_list_rect(list_rect_gt)/(s[0]*s[1])
        return iou/(i+1-diff_nb_pred*coef)

"""
def score_mask(mask_pred,mask_gt,s):
    # On utilise les rectangles entourant les masques de la prédiction ainsi que les rectangles "parfaits" de la GT
    rect_gen = np.zeros(shape=(mask_pred.shape[-1],4))
    for i in range(mask_pred.shape[-1]):
        mask = mask_pred[:,:,i]
        ones = np.where(mask)
        y1 = np.min(ones[0])
        y2 = np.max(ones[0])
        x1 = np.min(ones[1])
        x2 = np.max(ones[1])
        rect_gen[i] = np.array([y1,x1,y2,x2])
    return score_box(rect_gen,mask_gt,s)
"""

def score_mask(mask_pred,mask_gt,rect_pred,rect_gt,s):
    iou = 0
    if np.sum(rect_gt[0]==-1)==4: # Si il n'y a pas de gt
        if rect_pred.shape[0]==0: # Si il n'y a pas de prédictions tout est bien
            return 1.0
        else:
            return compute_area_list_rect(list(rect_pred))/(s[0]*s[1])
    
    diff_nb_pred = rect_pred.shape[0] - rect_gt.shape[0]
    list_rect_pred = list(rect_pred)
    list_rect_gt = list(rect_gt)
    list_mask_pred = list(np.swapaxes(np.swapaxes(mask_pred,2,1),1,0))
    list_mask_gt = list(np.swapaxes(np.swapaxes(mask_gt,2,1),1,0))
    if diff_nb_pred==0: # Si il y a autant d'objets détectés que dans la GT
        i=0
        for i,pred in enumerate(rect_pred):
            y1,x1,y2,x2 = pred # Coordonnées du rectangle dans la prediction
            G = (np.mean([y1,y2]),np.mean([x1,x2]))
            dist_min = float('inf')
            for index,gt in enumerate(rect_gt):
                mask = mask_gt[:,:,index]
                y1_gt,x1_gt,y2_gt,x2_gt = gt
                G_gt = (np.mean([y1_gt,y2_gt]),np.mean([x1_gt,x2_gt]))
                dist = np.sqrt((G[0]-G_gt[0])**2+(G[1]-G_gt[1])**2)
                if dist<dist_min:
                    dist_min = dist
                    nearest_gt = gt
                    nearest_mask_gt = mask
            iou += compute_mask_iou(mask_pred[:,:,i],nearest_mask_gt)
        return iou/(i+1)
    elif diff_nb_pred>0: # Si il y a plus de prédictions que de gt
        i=0
        for i,gt in enumerate(rect_gt):
            y1_gt,x1_gt,y2_gt,x2_gt = gt # Coordonnées du rectangle dans la prediction
            G_gt = (np.mean([y1_gt,y2_gt]),np.mean([x1_gt,x2_gt]))
            dist_min = float('inf')
            for index,(pred,mask) in enumerate(zip(list_rect_pred,list_mask_pred)):
                y1,x1,y2,x2 = pred
                G = (np.mean([y1,y2]),np.mean([x1,x2]))
                dist = np.sqrt((G[0]-G_gt[0])**2+(G[1]-G_gt[1])**2)
                if dist<dist_min:
                    rem_index = index
                    dist_min = dist
                    nearest_pred = pred
                    nearest_mask_pred = mask
            del list_rect_pred[rem_index]
            del list_mask_pred[rem_index]
            iou += compute_mask_iou(nearest_mask_pred,mask_gt[:,:,i])
        coef = compute_area_list_rect(list_rect_pred)/(s[0]*s[1])
        return iou/(i+1+diff_nb_pred*coef)
    else: # Si il y a plus de gt que de prédictions
        i=0
        for i,pred in enumerate(rect_pred):
            y1,x1,y2,x2 = pred # Coordonnées du rectangle dans la prediction
            G = (np.mean([y1,y2]),np.mean([x1,x2]))
            dist_min = float('inf')
            for index,(gt,mask) in enumerate(zip(list_rect_gt,list_mask_gt)):
                y1_gt,x1_gt,y2_gt,x2_gt = gt
                G_gt = (np.mean([y1_gt,y2_gt]),np.mean([x1_gt,x2_gt]))
                dist = np.sqrt((G[0]-G_gt[0])**2+(G[1]-G_gt[1])**2)
                if dist<dist_min:
                    rem_index = index
                    dist_min = dist
                    nearest_gt = gt
                    nearest_mask_gt = mask
            del list_rect_gt[rem_index]
            del list_mask_gt[rem_index]
            iou += compute_mask_iou(mask_pred[:,:,i],nearest_mask_gt)
        coef = compute_area_list_rect(list_rect_gt)/(s[0]*s[1])
        return iou/(i+1-diff_nb_pred*coef)
    
def compute_rect_iou(rect1,rect2): # Compute IoU over 2 rectangles
    rect1_area = (rect1[2]-rect1[0])*(rect1[3]-rect1[1])
    rect2_area = (rect2[2]-rect2[0])*(rect2[3]-rect2[1])
    y1 = np.maximum(rect1[0], rect2[0])
    y2 = np.minimum(rect1[2], rect2[2])
    x1 = np.maximum(rect1[1], rect2[1])
    x2 = np.minimum(rect1[3], rect2[3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = rect1_area + rect2_area - intersection
    iou = intersection / union
    return iou

def compute_mask_iou(mask1,mask2): # mask1 = pred et mask2 = gt
    union = np.sum(mask1 | mask2)
    intersection = np.sum(mask1 & mask2)
    if union==0:
        return 1.0
    return intersection / union

def compute_area_list_rect(list_rect):
    taille = 0
    for e in list_rect:
        taille += (e[2]-e[0])*(e[3]-e[1])
    return taille-compute_inter_list_rect(list_rect)

def compute_inter_list_rect(list_rect):
    intersection = 0
    for i,rect1 in enumerate(list_rect):
        del list_rect[i]
        for rect2 in list_rect:
            y1 = np.maximum(rect1[0], rect2[0])
            y2 = np.minimum(rect1[2], rect2[2])
            x1 = np.maximum(rect1[1], rect2[1])
            x2 = np.minimum(rect1[3], rect2[3])
            intersection += np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    return intersection

def compute_rect_with_masks(masks):
    res = np.zeros((masks.shape[-1],4))
    for i in range(masks.shape[-1]):
        wh = np.where(masks[...,i])
        y2 = np.min(wh[0])
        y1 = np.max(wh[0])
        x2 = np.min(wh[1])
        x1 = np.max(wh[1])
        res[i] = np.array([y2,x2,y1,x1])
    return res

def conf(b,gt):
    VP = b & gt
    FN = gt & (~b)
    FP = b & (~gt)
    VN = (~b) & (~gt)

    vp=np.sum(VP)
    fn=np.sum(FN)
    fp=np.sum(FP)
    vn=np.sum(VN)

    return np.array([[vp,fn],[fp,vn]])

def f1_score(mat):
    if 2*mat[0,0]+mat[1,0]+mat[0,1]==0:
        return 1.0
    return 2*mat[0,0]/(2*mat[0,0]+mat[1,0]+mat[0,1])