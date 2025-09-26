import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.nn.modules.batchnorm import _BatchNorm
import pandas as pd
import torch.nn as nn
import os
import uuid
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class PrototypicalLoss:
  # This is just to initilize it
    def __init__(self, flag='neg'):
        self.flag = flag


# The centroid Computing
    def _compute_per_class_centroid(self, i, label, data):
        label1d = label.squeeze()
        data_class = data[label1d == i, :]
        return torch.mean(data_class, 0, True)

    def _compute_class_centroid(self, label, data):
        unique_labels = label.unique().squeeze()
        centroids = self._compute_per_class_centroid(unique_labels[0], label, data)
        for i in range(1, len(unique_labels)):
            index = unique_labels[i]
            centroids = torch.cat((centroids, self._compute_per_class_centroid(index, label, data)), dim=0)
        return centroids

# The similarity or distance computaions
    def _cosine_similarity(self, data, centroid):
        data_norm = torch.nn.functional.normalize(data, dim=1)
        centroid_norm = torch.nn.functional.normalize(centroid, dim=1)
        similarity = torch.mm(data_norm, centroid_norm.t())
        return similarity

    def _similarity_matrix(self, data, centroid):
        epsilon = 1e-6
        similarity = 1 / (torch.cdist(data, centroid, p=2) + epsilon)
        return similarity

    def _distance_matrix(self, data, centroid):
        distance = torch.cdist(data, centroid, p=2)
        return distance
# Using the computations to get loss
    def _prototypical_loss_sim(self, S, labels, alpha=0.01):
        softmax = torch.nn.Softmax(dim=1)
        o = softmax(S / alpha)
        labels = labels.squeeze().long()
        loss = F.cross_entropy(o, labels, reduction='mean')
        return loss

    def _prototypical_loss_neg(self, D, labels):
        softmax = torch.nn.Softmax(dim=1)
        o = softmax(-D)
        labels = labels.squeeze().long()
        loss = F.cross_entropy(o, labels, reduction='mean')
        return loss

    def _prototypical_loss_negexp(self, D, labels, alpha = 0.1):
        softmax = torch.nn.Softmax(dim=1)
        o = softmax(-alpha*torch.exp(D))
        labels = labels.squeeze().long()
        loss = F.cross_entropy(o, labels, reduction='mean')
        return loss

#The call function
    def __call__(self, data,label):
        if self.flag == 'neg':
            centroids = self._compute_class_centroid(label, data)
            distance = self._distance_matrix(data, centroids)
            return self._prototypical_loss_neg(distance, label)
        elif self.flag == 'sim':
            centroids = self._compute_class_centroid(label, data)
            similarity = self._similarity_matrix(data, centroids)
            return self._prototypical_loss_sim(similarity, label)
        elif self.flag == 'cos':
            centroids = self._compute_class_centroid(label, data)
            similarity = self._cosine_similarity(data.detach(), centroids)
            return self._prototypical_loss_sim(similarity, label)
        elif self.flag == 'negexp':
            centroids = self._compute_class_centroid(label, data)
            distance = self._distance_matrix(data, centroids)
            return self._prototypical_loss_negexp(distance, label)


def prototypical_testing(test_embed, train_centroids):

  # Making them both be on Cpu
    test_embed = test_embed.cpu()
    train_centroids = train_centroids.cpu()

# Computing the distabce
    cdist = torch.cdist(test_embed, train_centroids)

#Pick the most similar aka the one with the smallest distance

    test_label = torch.argmin(cdist,dim=1)

    return test_label #the testing label based on training centroid
