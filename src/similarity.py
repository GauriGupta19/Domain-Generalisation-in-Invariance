from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sn

from src.data import *



def get_mismatched_data(test_data1, test_data2, test_labels1, test_labels2):
    """
    Given two datasets (1),(2) and their labels, generates a mismatched dataset
    pulling from (2) that differs in label from (1) at each index.
    """
    
    mismatched_data = copy.deepcopy(test_data1)
    mismatched_labels = copy.deepcopy(test_labels1)
    
    for i in range(0, len(test_data1)):
        
        while(True):
            
            temp_idx = random.randint(0, len(test_data2)-1)
            
            if((test_labels1[i] != test_labels2[temp_idx]).item()):
                mismatched_data[i] = test_data2[temp_idx]
                mismatched_labels[i] = test_labels2[temp_idx]
                break

    for i in range(0, len(test_data1)):
        assert(test_labels1[i].item() != mismatched_labels[i].item())
  
    return mismatched_data

def pair_matching(testset, test_labels, model, coord):
    """
    Evaluates classification of a pair of a transformed and normal image as
    the same label or different labels. The classification is done by thresholds
    on cosine similarity of the model's latent encodings of the images.
    
    Args:
        testset:
            Raw dataset to pull test images from
        test_labels:
            Correct labels of testset
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
    
    # create mismatched transformed data that differs from the original.
    mismatched_data = get_mismatched_data(test_data1, test_data2, test_labels1, test_labels2)
    
    # labels of ones and zeros
    same_label = torch.ones((test_labels1.size()[0]))
    mismatched_label = torch.zeros((test_labels1.size()[0]))
    
    # create our total data by concatenation.
    # data1 and data2 are same
    # data1 and mismatched are different
    point1 = torch.cat([test_data1, test_data1])
    point2 = torch.cat([test_data2, mismatched_data])
    label = torch.cat([same_label, mismatched_label])
    
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
            Correct labels of testset.
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
    


def class_matching(testset, test_labels, model, coord):
    
    #pair of corresponding transformed images
    test_data1, test_labels1, test_angles1, test_translations1 = get_mnist_data(
        testset, 
        digits = test_labels)
    test_data2, test_labels2, test_angles2, test_translations2 = get_mnist_data(
        testset, 
        digits = test_labels, 
        coord = coord)
    
    idx = torch.arange(test_labels2.size()[0])
    same_label = torch.ones((test_labels1.size()[0]))
 
    #shuffled pairs
    indices = torch.randperm(test_labels2.size()[0])
    test_data2_shuffled = test_data2[indices]
    test_labels2_shuffled = test_labels2[indices]
    same_label_shuffled = (test_labels2_shuffled==test_labels1).long()
    
#     indices = test_data2_shuffled[test_labels1!=test_labels2_shuffled]
#     same_label_shuffled = torch.zeros((test_data2_shuffled.size()[0]))

    # we are only interested in checking if a pair of two images(rotated and non-rotated) are same or not (for same class label-> should be same)
    point1 = torch.cat([test_data1, test_data1])
    point2 = torch.cat([test_data2, test_data2_shuffled])
    label = torch.cat([same_label, same_label_shuffled])

    z1 = model.encode(point1)
    z_mu1, z_sigma1 = z1[0], z1[1]
    z_sample1 = z_mu1 + torch.randn_like(z_mu1)*z_sigma1

    z2 = model.encode(point2)
    z_mu2, z_sigma2 = z2[0], z2[1]
    z_sample2 = z_mu2 + torch.randn_like(z_mu2)*z_sigma2

    point1 = z_mu1[:, model.coord:]
    point2 = z_mu2[:, model.coord:]
    
    cos = torch.nn.CosineSimilarity(dim=1)
    scores = cos(point1, point2)
    fpr, tpr, thresholds = roc_curve(label, scores, pos_label=1)
    auc = roc_auc_score(label, scores)
    return auc, fpr, tpr, thresholds