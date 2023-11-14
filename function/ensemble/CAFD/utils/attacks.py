import numpy as np
import torch
# from torchvision import transforms
#import sys, os
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt

import config as cf


class Attacks:
    
    def __init__(self, model, eps, N_train, N_test, momentum=None, is_normalized=False, retain=False):
        self.adv_examples = {'train': None, 'test': None}
        self.adv_labels = {'train': None, 'test': None}
        self.adv_stored = {'train': False, 'test': False}
        self.count = {'train': 0, 'test': 0}
        
        self.model = model.cuda()
        self.eps = eps
        self.momentum = momentum
        self.is_normalized = is_normalized
        self.retain = retain
        
        self.freeze()
        if retain:
            self.reset_imgs(N_train, N_test)
        
            
    def freeze(self):
        """Freeze weights."""
        for param in self.model.parameters():
            param.requires_grad = False
            
    
    def reset_imgs(self, N_train, N_test):
        """ Resets new variable to store adversarial examples."""
        self.adv_examples['train'] = torch.zeros(N_train,3,32,32)
        self.adv_examples['test'] = torch.zeros(N_test,3,32,32)
        self.adv_labels['train'] = torch.zeros((N_train), dtype=torch.long)
        self.adv_labels['test'] = torch.zeros((N_test), dtype=torch.long)
        self.adv_stored['train'] = False
        self.adv_stored['test'] = False
        self.count['train'] = 0
        self.count['test'] = 0
        
    
    def set_stored(self, mode, is_stored):
        """ Defines whether adversarial examples have been stored."""
        # TODO: Health-check to test whether the last element in 'adv_examples' contains and adv
        self.adv_stored[mode] = is_stored
        
    
    def get_adversarial(self, batch_size, mode):
        """ Retrieve adversarial examples if they have already been generated."""
        # Restart at beggining if iteration if complete
        if self.count[mode] >= self.adv_examples[mode].size(0):
            self.count[mode] = 0
        
        x_batch = self.adv_examples[mode][(self.count[mode]):(self.count[mode] + batch_size)]
        y_batch = self.adv_labels[mode][(self.count[mode]):(self.count[mode] + batch_size)]
        self.count[mode] = self.count[mode] + batch_size
        
        return x_batch.cuda(), y_batch.cuda()
 
    # =================================== ATTACK ALGORITHMS HELPER ===================================
    
    def compute_alpha(self, eps, max_iter, is_normalized):
        """ 
        Computes alpha for scaled or normalized inputs. If PGD with
        multiple iterations is performed, alpha is computed by dividing
        by a fixed constant (4.) as opposed to the number of iterations
        (max_iter), which is stated in MIM paper.
        
        Output:
            - alpha:
                - (is_normalized : True)  := returns a cuda tensor of shape [1,C,1,1], containing alpha values for each channel C
                - (is_normalized : False) := returns a scalar alpha
        """
        alpha = None
        if is_normalized:
            # Epsilon is in the range of possible inputs.
            # Reshape to  [1,C,1,1] to enable broadcasting
            alpha = (eps * cf.eps_size)[np.newaxis,:,np.newaxis,np.newaxis]
            
            if max_iter > 1:
                alpha = ( alpha / 4.)
            
            alpha = torch.FloatTensor(alpha).cuda()
            
        else:
            alpha = eps
            if max_iter > 1:
                alpha = eps / 4.
        
        return alpha
    
    
    def clamp_tensor(self, x, is_normalized):
        """ Clamps tensor x between valid ranges for the image (normalized or scaled range)"""
        if is_normalized:
            x.data[:,0,:,:].clamp_(min=cf.min_val[0], max=cf.max_val[0])
            x.data[:,1,:,:].clamp_(min=cf.min_val[1], max=cf.max_val[1])
            x.data[:,2,:,:].clamp_(min=cf.min_val[2], max=cf.max_val[2])
        else:
            x.data.clamp_(min=0.0, max=1.0)
            
        return x.data
        
    
    # =================================== ATTACK ALGORITHMS ===================================

    def fast_pgd(self, x_batch, y_batch, max_iter, mode):
        """
        Generates adversarial examples using  projected gradient descent (PGD).
        If adversaries have been generated, retrieve them.
        
        Input:
            - x_batch : batch images to compute adversaries 
            - y_batch : labels of the batch
            - max_iter : # of iterations to generate adversarial example (FGSM=1)
            - mode : batch from 'train' or 'test' set
            - is_normalized :type of input normalization (0: no normalization, 1: zero-mean per-channel normalization)
        
        Output:
            - x : batch containing adversarial examples
        """
        # Retrieve adversaries
        if self.adv_stored[mode]:
            return self.get_adversarial(x_batch.size(0), mode)
        
        x = x_batch.clone().detach().requires_grad_(True).cuda()
        
        # Compute alpha. Alpha might vary depending on the type of normalization.
        alpha = self.compute_alpha(self.eps, max_iter, self.is_normalized)
        
        # Set velocity for momentum
        if self.momentum:
            g = torch.zeros(x_batch.size(0), 1, 1, 1).cuda()
        
        for _ in range(max_iter):
            
            logits = self.model(x)
            loss = nn.CrossEntropyLoss()(logits, y_batch)
            
            loss.backward()
            
            # Get gradient
            noise = x.grad.data
            
            # Momentum : You should not be using the mean here...
            if self.momentum:
                g = self.momentum * g.data + noise / torch.mean(torch.abs(noise), dim=(1,2,3), keepdim=True)
                noise = g.clone().detach()
            
            # Compute Adversary
            x.data = x.data + alpha * torch.sign(noise)
            
            # Clamp data between valid ranges
            x.data = self.clamp_tensor(x, self.is_normalized)
            
            x.grad.zero_()
        
        # Store adversarial images to array to retrieve on later iterations. Maybe refactor?
        if self.retain:
            self.adv_examples[mode][(self.count[mode]):(self.count[mode] + x.size(0))] = x.clone().detach()
            self.adv_labels[mode][(self.count[mode]):(self.count[mode] + x.size(0))] = y_batch.clone().detach()
            self.count[mode] = self.count[mode] + x.size(0)

        return x, y_batch