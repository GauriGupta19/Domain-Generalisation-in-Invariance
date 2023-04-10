from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score
from scikitplot.metrics import plot_roc_curve
import matplotlib.pyplot as plt
import random
import copy

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sn

from src.data import *

# Your mismatched data is the mismatched_data which you'll use along with test_data1
def get_mismatched_data(test_data1, test_data2, test_labels1, test_labels2):
    mismatched_data = copy.deepcopy(test_data1)
    mismatched_labels = copy.deepcopy(test_labels1)
    for i in range(0, len(test_data1)):
        while(True):
            temp_idx = random.randint(0, len(test_data2)-1)
            if((test_labels1[i] != test_labels2[temp_idx]).item()):
                mismatched_data[i] = test_data2[temp_idx]
                mismatched_labels[i] = test_labels2[temp_idx]
                break

    # Checking function to make sure every pair is mismatched
    for i in range(0, len(test_data1)):
        assert(test_labels1[i].item() != mismatched_labels[i].item())
  
    return mismatched_data

def pair_matching(testset, test_labels, model, coord, model_coord):
    
    if coord not in [0, 1, 2, 3]:
            raise ValueError("'coord' argument must be 0, 1, 2 or 3")
    
    if coord == 1 or coord == 3:
        rotation_range = [-60, 61]
    else:
        rotation_range = [0,1]
    if coord == 2 or coord == 3:
        translation_range = [-10, 11]
    else:
        translation_range = [0,1]
    
    #pair of corresponding transformed images
    test_data1, test_labels1, test_angles1, test_translations1 = get_mnist_data(
        testset, 
        digits = test_labels, 
        rotation_range=[0,1], 
        translation_range = [0,1])
    test_data2, test_labels2, test_angles2, test_translations2 = get_mnist_data(
        testset, 
        digits = test_labels, 
        rotation_range = rotation_range, 
        translation_range = translation_range)
    
    idx = torch.arange(test_labels2.size()[0])
    same_label = torch.ones((test_labels1.size()[0]))

    #shuffled pairs
#     indices = torch.randperm(test_labels2.size()[0])
#     test_data2_shuffled = test_data2[indices]
#     test_labels2_shuffled = test_labels2[indices]
#     same_label_shuffled = (indices==idx).long()

    # we are only interested in checking if a pair of two images(rotated and non-rotated) are same or not
    
    mismatched_data = get_mismatched_data(test_data1, test_data2, test_labels1, test_labels2)
    mismatched_label = torch.zeros((test_labels1.size()[0]))
    
    point1 = torch.cat([test_data1, test_data1])
    point2 = torch.cat([test_data2, mismatched_data])
    label = torch.cat([same_label, mismatched_label])
    
    z1 = model.encode(point1)
    z_mu1, z_sigma1 = z1[0], z1[1]
    z_sample1 = z_mu1 + torch.randn_like(z_mu1)*z_sigma1

    z2 = model.encode(point2)
    z_mu2, z_sigma2 = z2[0], z2[1]
    z_sample2 = z_mu2 + torch.randn_like(z_mu2)*z_sigma2
    
    # 1 for rotation, 2 for translation, etc
    point1 = z_mu1[:, model_coord:]
    point2 = z_mu2[:, model_coord:]
    
    cos = torch.nn.CosineSimilarity(dim=1)
    scores = cos(point1, point2)
    fpr, tpr, thresholds = roc_curve(label, scores, pos_label=1)
    # plot_roc_curve(label, scores)
    auc = roc_auc_score(label, scores)
    return auc, fpr, tpr, thresholds

def class_matching(testset, test_labels, model, coord, model_coord):
    
    if coord not in [0, 1, 2, 3]:
        raise ValueError("'coord' argument must be 0, 1, 2 or 3")
    
    if coord == 1 or coord == 3:
        rotation_range = [-60, 61]
    else:
        rotation_range = [0,1]
    if coord == 2 or coord == 3:
        translation_range = [-10, 11]
    else:
        translation_range = [0,1]
    
    #pair of corresponding transformed images
    test_data1, test_labels1, test_angles1, test_translations1 = get_mnist_data(
        testset, 
        digits = test_labels, 
        rotation_range=[0,1], 
        translation_range = [0,1])
    test_data2, test_labels2, test_angles2, test_translations2 = get_mnist_data(
        testset, 
        digits = test_labels, 
        rotation_range = rotation_range, 
        translation_range = translation_range)
    
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

    point1 = z_mu1[:, model_coord:]
    point2 = z_mu2[:, model_coord:]
    
    cos = torch.nn.CosineSimilarity(dim=1)
    scores = cos(point1, point2)
    fpr, tpr, thresholds = roc_curve(label, scores, pos_label=1)
    # plot_roc_curve(label, scores)
    auc = roc_auc_score(label, scores)
    return auc, fpr, tpr, thresholds

def plot_sample_heatmap(testset, test_labels, model, coord, model_coord):
    if coord not in [0, 1, 2, 3]:
        raise ValueError("'coord' argument must be 0, 1, 2 or 3")
    
    if coord == 1 or coord == 3:
        rotation_range = [-60, 61]
    else:
        rotation_range = [0,1]
    if coord == 2 or coord == 3:
        translation_range = [-10, 11]
    else:
        translation_range = [0,1]
    
    #pair of corresponding transformed images
    test_data1, test_labels1, test_angles1, test_translations1 = get_mnist_data(
        testset, 
        digits = test_labels, 
        rotation_range=[0,1], 
        translation_range = [0,1])
    test_data2, test_labels2, test_angles2, test_translations2 = get_mnist_data(
        testset, 
        digits = test_labels, 
        rotation_range = rotation_range, 
        translation_range = translation_range)
    
    z1 = model.encode(test_data1[:10])
    z_mu1, z_sigma1 = z1[0], z1[1]
    z_sample1 = z_mu1 + torch.randn_like(z_mu1)*z_sigma1

    z2 = model.encode(test_data2[:10])
    z_mu2, z_sigma2 = z2[0], z2[1]
    z_sample2 = z_mu2 + torch.randn_like(z_mu2)*z_sigma2

    point1 = z_mu1[:, model_coord:]
    point2 = z_mu2[:, model_coord:]
    
    scores = cosine_similarity(point1, point2)
    heatmap = sn.heatmap(scores, annot=True)
    fig = heatmap.get_figure()
    # fig.savefig("heatmap.svg") 
