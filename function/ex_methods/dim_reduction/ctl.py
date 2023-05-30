import os
import random
import argparse
from tqdm import tqdm

import numpy as np


import torch
from torch import nn
from torchvision import transforms
from function.ex_methods.module.module import Module
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Controller():

    def __init__(self, vis_type_list, device):
        self.clear()
        self.feats = None
        self.device = device
        self.model:nn.Module = None
        self.vis_type_list = vis_type_list
        self.visualizers = {vis_type:self.get_visualizer(vis_type)
                            for vis_type in vis_type_list}

    def clear(self):
        self.model_flag = False
        self.layer_flag = False
        self.data_flag = False
        self.feats_flag = False

    def set_model(self, model:nn.Module):
        self.model = model
        self.model_flag = True

    def hook_func(self, module, feat_in, feat_out):
        self.feats = feat_out.detach()

    def set_layer(self, layer_name):

        if self.model == None:
            raise NotImplementedError('No model fixed, cannot select layer')
        else:
            finished_flag = False
            for name, params in self.model.named_children():
                if layer_name == name:
                    params.register_forward_hook(self.hook_func)
                    finished_flag = True
                    break
            if finished_flag == False:
                # Set default layer to get latent feature(the last conv2d layer)
                for layer_name2, param in reversed(list(self.model.named_children())):
                    if isinstance(param, nn.Conv2d) or isinstance(param, nn.Sequential):
                        print(f'[Info]No target layer {layer_name} found and instead {layer_name2} is selected')
                        self.set_layer(layer_name2)
                        break
            self.layer_flag = True

    def set_data_loaders(self, benign_dataloader, poisoned_dataloader):
        if isinstance(benign_dataloader, torch.utils.data.DataLoader):
            self.benign_data_loader = benign_dataloader
        else:
            print('Please provide a benign dataloader with type "torch.utils.data.DataLoader"')
            return
        if isinstance(poisoned_dataloader, torch.utils.data.DataLoader):
            self.poisoned_data_loader = poisoned_dataloader
        else:
            print('Please provide a adversarial dataloader with type "torch.utils.data.DataLoader"')
            return
        self.data_flag = True

    def prepare_feats(self):
        if not self.model_flag:
            print('No Model Ready')
        elif not self.layer_flag:
            print('No Layer Selected')
        elif not self.data_flag:
            print('No Data Loaded')
        else:
            _, self.benign_feats = self.get_latent_feature(self.benign_data_loader)
            _, self.poisoned_feats = self.get_latent_feature(self.poisoned_data_loader)
            self.feats_flag = True

    def get_reduced_feats(self, vis_type):
        if not self.feats_flag:
            raise NotImplementedError(
                'No feats available, please call cls.prepare_feats function first')
        if vis_type in ('pca', 'tsne'):
            reduced_benign_feats = self._get_reduced_latent_feature(self.benign_feats, vis_type)
            reduced_poisoned_feats = self._get_reduced_latent_feature(self.poisoned_feats, vis_type)
        else:
            reduced_benign_feats, reduced_poisoned_feats = \
                self._get_contrast_reduced_latent_feature(self.benign_feats,
                                                         self.poisoned_feats, vis_type)
        return reduced_benign_feats, reduced_poisoned_feats

    def get_visualizer(self, type):
        if type.lower() == 'pca':
            visualizer = PCA(n_components=2)
        elif type.lower() == 'tsne':
            visualizer = TSNE(n_components=2)
        elif type.lower() in ('oracle', 'svm'):
            visualizer = oracle_visualizer()
        elif type.lower() == 'mean_diff':
            visualizer = mean_diff_visualizer()
        elif type.lower() == 'ss':
            visualizer = spectral_visualizer()
        else:
            raise NotImplementedError(
                'Visualization Method %s is Not Implemented!' % type)
        return visualizer

    def get_latent_feature(self, data_loader):
        targets, features = [], []
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(data_loader), desc='Get Latent Feature'):
                data, target = data.to(self.device), target.to(self.device)  # train set batch
                targets.append(target)
                self.model.forward(data)
                features.append(self.feats.cpu().detach())
        targets = torch.cat(targets, dim=0)
        features = torch.cat(features, dim=0)

        return targets.cpu(), features.reshape(features.shape[0], -1).cpu()

    def _get_reduced_latent_feature(self, feats, vis_type):
        return self.visualizers[vis_type].fit_transform(feats)

    def _get_contrast_reduced_latent_feature(self, feats1, feats2, vis_type):
        return self.visualizers[vis_type].fit_transform(feats1, feats2)

    def show_layer_list(self):
        if self.model == None:
            raise NotImplementedError('No model fixed, no info provided')
        print('The whole model arch')
        for name, params in self.model.named_parameters():
            print(name, params.data.size())
        print('--------------------------------------')

class mean_diff_visualizer:

    def fit_transform(self, clean, poison):
        clean_mean = clean.mean(dim=0)
        poison_mean = poison.mean(dim=0)
        mean_diff = poison_mean - clean_mean
        print("Mean L2 distance between poison and clean:",
              torch.norm(mean_diff, p=2).item())

        proj_clean_mean = torch.matmul(clean, mean_diff)
        proj_poison_mean = torch.matmul(poison, mean_diff)

        return proj_clean_mean, proj_poison_mean


class oracle_visualizer:

    def __init__(self):
        self.clf = svm.LinearSVC()

    def fit_transform(self, clean, poison):

        clean = clean.numpy()
        num_clean = len(clean)

        poison = poison.numpy()
        num_poison = len(poison)

        # print(clean.shape, poison.shape)

        X = np.concatenate([clean, poison], axis=0)
        y = []

        for _ in range(num_clean):
            y.append(0)
        for _ in range(num_poison):
            y.append(1)

        self.clf.fit(X, y)

        norm = np.linalg.norm(self.clf.coef_)
        self.clf.coef_ = self.clf.coef_ / norm
        self.clf.intercept_ = self.clf.intercept_ / norm

        projection = self.clf.decision_function(X)

        return projection[:num_clean], projection[num_clean:]


class spectral_visualizer:

    def fit_transform(self, clean, poison):
        all_features = torch.cat((clean, poison), dim=0)
        all_features -= all_features.mean(dim=0)
        _, _, V = torch.svd(all_features, compute_uv=True, some=False)
        vec = V[:, 0]  # the top right singular vector is the first column of V
        vals = []
        for j in range(all_features.shape[0]):
            vals.append(torch.dot(all_features[j], vec).pow(2))
        vals = torch.tensor(vals)


        return vals[:clean.shape[0]], vals[clean.shape[0]:]
