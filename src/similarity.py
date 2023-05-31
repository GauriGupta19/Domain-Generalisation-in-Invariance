from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sn

from src.data import *



def get_mismatched_data(
    test_data1, test_data2, test_labels1, test_labels2, 
    same_label = False, 
    diff_image = False
    ):
    """
    Given two datasets (1),(2) and their labels, generates a dataset
    pulling from (2). Constraints enforced to make the labels same/different
    at each index. Addiitional constraint may be enforced to make the images different.
    
    Args:
        same_label:
            True -> requires labels to be same
            False -> requires labels to be different
        diff_image:
            True -> requires images to be different 
                (only matters if same_label = True)
            False -> doesn't care
    """
    
    mismatched_data = copy.deepcopy(test_data1)
    mismatched_labels = copy.deepcopy(test_labels1)
    
    for i in range(0, len(test_data1)):
        
        while(True):
            
            temp_idx = random.randint(0, len(test_data2)-1)
            
            if (same_label == True and
                test_labels1[i] == test_labels2[temp_idx]):
                
                if (not diff_image or i != temp_idx):
                    
                    mismatched_data[i] = test_data2[temp_idx]
                    mismatched_labels[i] = test_labels2[temp_idx]
                    break
            
            if (same_label == False and
                test_labels1[i] != test_labels2[temp_idx]):
                    
                mismatched_data[i] = test_data2[temp_idx]
                mismatched_labels[i] = test_labels2[temp_idx]
                break

    for i in range(0, len(test_data1)):
        if same_label:
            assert(test_labels1[i] == mismatched_labels[i])
        else:
            assert(test_labels1[i] != mismatched_labels[i])
  
    return mismatched_data

def pair_matching(testset, test_labels, model, coord):
    """
    Evaluates classification of a pair of a transformed and normal image 
    in the same class as the same image or different images. Effectively:
    
    "Here are two 5s. Are they the same image, or different handwritings?"
    
    The classification is done by thresholds
    on cosine similarity of the model's latent encodings of the images.
    
    Args:
        testset:
            Raw dataset to pull test images from
        test_labels:
            Labels to test on
        model:
            Model which we are evaluating
        coord: 
            Specifies the transformation we apply to the testset.
            - 0: no transformation
            - 1: rotation only
            - 2: translation only
            - 3: rotation + translation
    """
    
    # create untransformed and transformed data
    test_data1, test_labels1, test_angles1, test_translations1 = get_mnist_data(
        testset, 
        digits = test_labels)
    
    test_data2, test_labels2, test_angles2, test_translations2 = get_mnist_data(
        testset, 
        digits = test_labels, 
        coord = coord)
    
    # create same-class different-image data
    mismatched = get_mismatched_data(
        test_data1, test_data2, test_labels1, test_labels2, 
        same_label = True, 
        diff_image = True)
    
    # labels of ones and zeros
    positive_labels = torch.ones((test_labels1.size()[0]))
    negative_labels = torch.zeros((test_labels1.size()[0]))
    
    # we are only interested in checking if a pair of two images in the same class (rotated and non-rotated) are same image or not. 
    
    
    # create our total data by concatenation.
    # test_data1 and test_data2 are same (label ones)
    # test_data1 and mismatched are different (label zeros)
    point1 = torch.cat([test_data1, test_data1])
    point2 = torch.cat([test_data2, mismatched])
    label = torch.cat([positive_labels, negative_labels])
    
    # obtain latent representations of points
    z1 = model.encode(point1)
    z_mu1, z_sigma1 = z1[0], z1[1]
    z_sample1 = z_mu1 + torch.randn_like(z_mu1)*z_sigma1

    z2 = model.encode(point2)
    z_mu2, z_sigma2 = z2[0], z2[1]
    z_sample2 = z_mu2 + torch.randn_like(z_mu2)*z_sigma2
    
    point1 = z_mu1[:, model.coord:]
    point2 = z_mu2[:, model.coord:]
    
    # cosine similarity & roc curve
    cos = torch.nn.CosineSimilarity(dim=1)
    scores = cos(point1, point2)
    fpr, tpr, thresholds = roc_curve(label, scores, pos_label=1)
    auc = roc_auc_score(label, scores)
    
    return auc, fpr, tpr, thresholds

def class_matching(testset, test_labels, model, coord):
    """
    Evaluates classification of a pair of a transformed and normal image 
    as the same class or different classes. Effectively:
    
    "Here are two digit images. Are they the same digit, or different digits?"
    
    The classification is done by thresholds
    on cosine similarity of the model's latent encodings of the images.
    
    Args:
        testset:
            Raw dataset to pull test images from
        test_labels:
            Labels to test on
        model:
            Model which we are evaluating
        coord: 
            Specifies the transformation we apply to the testset.
            - 0: no transformation
            - 1: rotation only
            - 2: translation only
            - 3: rotation + translation
    """
    # create untransformed and transformed data
    test_data1, test_labels1, test_angles1, test_translations1 = get_mnist_data(
        testset, 
        digits = test_labels)
    test_data2, test_labels2, test_angles2, test_translations2 = get_mnist_data(
        testset, 
        digits = test_labels, 
        coord = coord)
    
    # create same-class and different-class data
    same_class = get_mismatched_data(
        test_data1, test_data2, test_labels1, test_labels2, 
        same_label = True)
    diff_class = get_mismatched_data(
        test_data1, test_data2, test_labels1, test_labels2, 
        same_label = False)
    
    # labels of ones and zeros
    positive_labels = torch.ones((test_labels1.size()[0]))
    negative_labels = torch.zeros((test_labels1.size()[0]))
 

    # we are only interested in checking if a pair of two images(rotated and non-rotated) are same class or not
    
    # create our total data by concatenation.
    # test_data1 and same_class are same class (label ones)
    # test_data1 and diff_calss are different class (label zeros)
    point1 = torch.cat([test_data1, test_data1])
    point2 = torch.cat([same_class, diff_class])
    label = torch.cat([positive_labels, negative_labels])

    
    # obtain latent representations of points
    z1 = model.encode(point1)
    z_mu1, z_sigma1 = z1[0], z1[1]
    z_sample1 = z_mu1 + torch.randn_like(z_mu1)*z_sigma1

    z2 = model.encode(point2)
    z_mu2, z_sigma2 = z2[0], z2[1]
    z_sample2 = z_mu2 + torch.randn_like(z_mu2)*z_sigma2

    point1 = z_mu1[:, model.coord:]
    point2 = z_mu2[:, model.coord:]
    
    # cosine similarity & roc curve
    cos = torch.nn.CosineSimilarity(dim=1)
    scores = cos(point1, point2)
    fpr, tpr, thresholds = roc_curve(label, scores, pos_label=1)
    auc = roc_auc_score(label, scores)
    
    return auc, fpr, tpr, thresholds

def plot_sample_heatmap(testset, test_labels, model, coord, n_samples = 20, **kwargs):
    """
    Plots heatmap of cosine similarity of latent encodings of pairs of 
    transformed and normal images.
    
    Args:
        testset:
            Raw dataset to pull images from
        test_labels:
            Labels to sample from.
        model:
            Model which we are using.
        coord: 
            Specifies the transformation we apply to the testset.
            - 0: no transformation
            - 1: rotation only
            - 2: translation only
            - 3: rotation + translation
        n_samples:
            Number of samples to use from each class. Default is 20.
        **kwargs:
            Passed to seaborn.heatmap()
    """
    # create normal and transformed data
    test_data1, test_labels1, test_angles1, test_translations1 = get_mnist_data(
        testset, 
        digits = test_labels,
        n_samples = n_samples)
    test_data2, test_labels2, test_angles2, test_translations2 = get_mnist_data(
        testset, 
        digits = test_labels,
        n_samples = n_samples,
        coord = coord)
    
    test_data1 = torch.index_select(test_data1, 0, torch.argsort(test_labels1))
    test_data2 = torch.index_select(test_data2, 0, torch.argsort(test_labels2))
    
    z1 = model.encode(test_data1)
    z_mu1, z_sigma1 = z1[0], z1[1]
    z_sample1 = z_mu1 + torch.randn_like(z_mu1)*z_sigma1

    z2 = model.encode(test_data2)
    z_mu2, z_sigma2 = z2[0], z2[1]
    z_sample2 = z_mu2 + torch.randn_like(z_mu2)*z_sigma2

    point1 = z_mu1[:, model.coord:]
    point2 = z_mu2[:, model.coord:]
    
    scores = cosine_similarity(point1, point2)
    ax = heatmap = sn.heatmap(scores, **kwargs)
    ticks = (np.array(range(len(test_labels)))+0.5)*n_samples
    ax.set_xticks(ticks, test_labels)
    ax.set_yticks(ticks, test_labels)
    
    # fig = heatmap.get_figure()
    


