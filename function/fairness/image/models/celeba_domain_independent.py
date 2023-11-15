import os
import pickle
import h5py
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from function.fairness.image.models import basenet
from function.fairness.image.models import dataloader
from function.fairness.image.models.celeba_core import CelebaModel
from function.fairness.image import utils

class CelebaDomainIndependent(CelebaModel):
    def __init__(self, opt):
        super(CelebaDomainIndependent, self).__init__(opt)
        self.best_dev_mAP_conditional = 0.
        self.best_dev_mAP_max = 0.
        self.best_dev_mAP_sum_prob = 0.
        self.best_dev_mAP_sum_out = 0.
        
    def _criterion(self, output, target):
        domain_label = target[:, -1:]
        class_num = output.size(1) // 2
        loss = F.binary_cross_entropy_with_logits(
                   domain_label*output[:, :class_num]
                       + (1-domain_label)*output[:, class_num:],
                   target[:, :-1])
        return loss
    
    def inference_conditional(self, output, target):
        """Inference method: condition on the known domain"""
        
        domain_label = target[:, -1:]
        predict_prob = torch.sigmoid(output).cpu().numpy()
        class_num = predict_prob.shape[1] // 2
        predict_prob = domain_label*predict_prob[:, :class_num] \
                       + (1-domain_label)*predict_prob[:, class_num:]
        return predict_prob
    
    def inference_max(self, output):
        """Inference method: choose the max of the two domains"""
        
        predict_prob = torch.sigmoid(output).cpu().numpy()
        class_num = predict_prob.shape[1] // 2
        predict_prob = np.maximum(predict_prob[:, :class_num],
                                  predict_prob[:, class_num:])
        return predict_prob
    
    def inference_sum_prob(self, output):
        """Inference method: sum the probability from two domains"""
        
        predict_prob = torch.sigmoid(output).cpu().numpy()
        class_num = predict_prob.shape[1] // 2
        predict_prob = predict_prob[:, :class_num] + predict_prob[:, class_num:]
        return predict_prob
    
    def inference_sum_out(self, output):
        """Inference method: sum the output from two domains"""
        
        class_num = output.size(1) // 2
        return (output[:, :class_num] + output[:, class_num:]).cpu().numpy()
    
    def train(self):
        """Train the model for one epoch, evaluate on validation set and 
        save the best model for each inference method
        """
        
        start_time = datetime.now()
        self._train(self.train_loader)
        utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'ckpt.pth'))
        dev_loss, dev_output, _ = self._test(self.dev_loader)
        
        dev_predict_conditional = self.inference_conditional(dev_output, self.dev_target)
        dev_per_class_AP_conditional = utils.compute_weighted_AP(self.dev_target,
                                            dev_predict_conditional, self.dev_class_weight)
        dev_mAP_conditional = utils.compute_mAP(dev_per_class_AP_conditional, self.subclass_idx)
        if dev_mAP_conditional > self.best_dev_mAP_conditional:
            self.best_dev_mAP_conditional = dev_mAP_conditional
            utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'best-conditional.pth'))
            
        dev_predict_max = self.inference_max(dev_output)
        dev_per_class_AP_max = utils.compute_weighted_AP(self.dev_target, 
                                    dev_predict_max, self.dev_class_weight)
        dev_mAP_max = utils.compute_mAP(dev_per_class_AP_max, self.subclass_idx)
        if dev_mAP_max > self.best_dev_mAP_max:
            self.best_dev_mAP_max = dev_mAP_max
            utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'best-max.pth'))
            
        dev_predict_sum_prob = self.inference_sum_prob(dev_output)
        dev_per_class_AP_sum_prob = utils.compute_weighted_AP(self.dev_target,
                                         dev_predict_sum_prob, self.dev_class_weight)
        dev_mAP_sum_prob = utils.compute_mAP(dev_per_class_AP_sum_prob, self.subclass_idx)
        if dev_mAP_sum_prob > self.best_dev_mAP_sum_prob:
            self.best_dev_mAP_sum_prob = dev_mAP_sum_prob
            utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'best-sum_prob.pth'))
            
        dev_predict_sum_out = self.inference_sum_out(dev_output)
        dev_per_class_AP_sum_out = utils.compute_weighted_AP(self.dev_target,
                                         dev_predict_sum_out, self.dev_class_weight)
        dev_mAP_sum_out = utils.compute_mAP(dev_per_class_AP_sum_out, self.subclass_idx)
        if dev_mAP_sum_out > self.best_dev_mAP_sum_out:
            self.best_dev_mAP_sum_out = dev_mAP_sum_out
            utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'best-sum_out.pth'))
        
        self.log_result('Dev epoch', 
                        {
                            'loss': dev_loss/len(self.dev_loader), 
                            'mAP_conditional': dev_mAP_conditional,
                            'mAP_max': dev_mAP_max,
                            'mAP_sum_prob': dev_mAP_sum_prob,
                            'mAP_sum_out': dev_mAP_sum_out,
                        },
                        self.epoch)
        
        duration = datetime.now() - start_time
        print(('Finish training epoch {}, dev mAP conditional: {}'
               'dev mAP max: {}, dev mAP sum prob: {}, '
               'dev mAP sum out: {}, time used: {}').format(self.epoch, dev_mAP_conditional,
                    dev_mAP_max, dev_mAP_sum_prob, dev_mAP_sum_out, duration))
        
    def _compute_result(self, model_name, data_loader, target, class_weight,
                          inference_fn, save_name, conditional=False):
        """Load model and compute performance with given inference method"""
        
        state_dict = torch.load(os.path.join(self.save_path, model_name))
        self.network.load_state_dict(state_dict['model'])
        loss, output, feature = self._test(data_loader)
        if conditional:
            predict = inference_fn(output, target)
        else:
            predict = inference_fn(output)
        per_class_AP = utils.compute_weighted_AP(target, predict, 
                                                     class_weight)
        mAP = utils.compute_mAP(per_class_AP, self.subclass_idx)
        result = {'output': output.cpu().numpy(), 
                  'feature': feature.cpu().numpy(),
                  'per_class_AP': per_class_AP,
                  'mAP': mAP}
        utils.save_pkl(result, os.path.join(self.save_path, save_name))
        
        res = {
            "y_pred": predict[:, self.subclass_idx],
            "y_true": self.test_target[:, self.subclass_idx],
            "z": self.test_target[:, -1]
        }
        return mAP, res
        
    def test(self):
        # Test and save the result for different inference methods
        dev_mAP_conditional, result = self._compute_result('best-conditional.pth', self.dev_loader,
                                  self.dev_target, self.dev_class_weight,
                                  self.inference_conditional,
                                  'dev_conditional_result.pkl', conditional=True)
        test_mAP_conditional, _ = self._compute_result('best-conditional.pth', self.test_loader,
                                   self.test_target, self.test_class_weight,
                                   self.inference_conditional,
                                   'test_conditional_result.pkl', conditional=True)
        
        dev_mAP_max, _ = self._compute_result('best-max.pth', self.dev_loader,
                                  self.dev_target, self.dev_class_weight,
                                  self.inference_max,
                                  'dev_max_result.pkl')
        test_mAP_max, _ = self._compute_result('best-max.pth', self.test_loader,
                                   self.test_target, self.test_class_weight,
                                   self.inference_max,
                                   'test_max_result.pkl')
        
        dev_mAP_sum_prob, _ = self._compute_result('best-sum_prob.pth', self.dev_loader,
                                  self.dev_target, self.dev_class_weight,
                                  self.inference_sum_prob,
                                  'dev_sum_prob_result.pkl')
        test_mAP_sum_prob, _ = self._compute_result('best-sum_prob.pth', self.test_loader,
                                   self.test_target, self.test_class_weight,
                                   self.inference_sum_prob,
                                   'test_sum_prob_result.pkl')

        dev_mAP_sum_out, _ = self._compute_result('best-sum_out.pth', self.dev_loader,
                                  self.dev_target, self.dev_class_weight,
                                  self.inference_sum_out,
                                  'dev_sum_out_result.pkl')
        test_mAP_sum_out, _ = self._compute_result('best-sum_out.pth', self.test_loader,
                                   self.test_target, self.test_class_weight,
                                   self.inference_sum_out,
                                   'test_sum_out_result.pkl')
        
        # Output the mean AP for the best model on dev and test set
        info = (('Dev conditional mAP: {}, max mAP: {}, sum prob mAP: {}, sum out mAP: {}\n'
                 'Test conditional mAP: {}, max mAP: {}, sum prob mAP: {}, sum out mAP: {}'
                 ).format(dev_mAP_conditional, dev_mAP_max, dev_mAP_sum_prob, dev_mAP_sum_out,
                          test_mAP_conditional, test_mAP_max, test_mAP_sum_prob, test_mAP_sum_out))
        utils.write_info(os.path.join(self.save_path, 'result.txt'), info)
        return result
        
        
        
        