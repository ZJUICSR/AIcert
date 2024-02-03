from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Optional, Tuple, Any, TYPE_CHECKING
import numpy as np
from scipy.ndimage import zoom
from tqdm.auto import trange
from function.attack.attacks.config import MY_NUMPY_DTYPE
from function.attack.attacks.attack import EvasionAttack
from function.attack.estimators.estimator import BaseEstimator
from function.attack.estimators.classification.classifier import ClassifierMixin
from function.attack.attacks.utils import (
    get_labels_np_array,
    check_and_transform_label_format,
)
import torch
import torch.nn.functional as F
if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_TYPE

class ZooAttack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "confidence",
        "targeted",
        "step_size",
        "max_iter",
        "binary_search_steps",
        "initial_const",
        "abort_early",
        "batch_size",
        "adam_beta1",
        "adam_beta2",
        "solver",
        "early_stop_iters",
        "nb_parallel",
        "use_tanh",
    ]
    # targeted, solver, abort_early=True,
    # batch_size=128,max_iter=2000,initial_const=0.01,confidence=0.0,early_stop_iters=100, binary_search_steps=10,
    # step_size=0.01,adam_beta1=0.9,adam_beta2=0.999
    
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        confidence: float = 0,
        targeted: bool = False,
        step_size: float = 1e-2,
        max_iter: int = 3000,
        binary_search_steps: int = 9,
        initial_const: float = 0.01,
        abort_early: bool = True,
        batch_size: int = 128,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        solver: str = "adam",
        early_stop_iters: int = 100,
        nb_parallel: int = 128,
        use_tanh: bool = False,
    ):
        super().__init__(estimator=classifier)

        if len(classifier.input_shape) == 1:
            self.input_is_feature_vector = True
        else:
            self.input_is_feature_vector = False

        self.confidence = confidence
        self.targeted = targeted
        self.step_size = step_size
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.abort_early = abort_early
        self.batch_size = batch_size
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.solver = solver
        self.early_stop_iters = early_stop_iters
        self.nb_parallel = nb_parallel
        self.use_tanh = use_tanh
    
    def coordinate_ADAM(self, losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down, step_size,beta1, beta2, proj):
        for i in range(batch_size):
            grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002 
        # ADAM update
        mt = mt_arr[indice]
        mt = beta1 * mt + (1 - beta1) * grad
        mt_arr[indice] = mt
        vt = vt_arr[indice]
        vt = beta2 * vt + (1 - beta2) * (grad * grad)
        vt_arr[indice] = vt
        epoch = adam_epoch[indice]
        corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
        m = real_modifier.reshape(-1)
        old_val = m[indice] 
        old_val -= step_size * corr * mt / (np.sqrt(vt) + 1e-8)
        # set it back to [-0.5, +0.5] region
        if proj:
            old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
        m[indice] = old_val
        adam_epoch[indice] = epoch + 1
    
    def coordinate_Newton(self, losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down, step_size, beta1, beta2, proj):
        cur_loss = losses[0]
        for i in range(batch_size):
            grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002 
            hess[i] = (losses[i*2+1] - 2 * cur_loss + losses[i*2+2]) / (0.0001 * 0.0001)
        hess[hess < 0] = 1.0
        hess[hess < 0.1] = 0.1
        m = real_modifier.reshape(-1)
        old_val = m[indice] 
        old_val -= step_size * grad / hess
        # set it back to [-0.5, +0.5] region
        if proj:
            old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
        m[indice] = old_val

    def loss_run(self, input,target,model,modifier,use_tanh,use_log,targeted,confidence,const):
        if use_tanh:
            pert_out = torch.tanh(input + modifier + 1)/2
        else:
            pert_out = input + modifier
        output = model(pert_out)
        if use_log:
            output = F.softmax(output,-1)
        if use_tanh:
            loss1 = torch.sum(torch.square(pert_out-torch.tanh(input+1)/2),dim=(1,2,3))
        else:
            loss1 = torch.sum(torch.square(pert_out-input),dim=(1,2,3))
        real = torch.sum(target*output,-1)
        other = torch.max((1-target)*output-(target*10000),-1)[0]
        if use_log:
            real=torch.log(real+1e-30)
            other=torch.log(other+1e-30)
        confidence = torch.tensor(confidence).type(torch.float64).to(self.estimator.device)                         
        if targeted:
            loss2 = torch.max(other-real,confidence)
        else:
            loss2 = torch.max(real-other,confidence)
        loss2 = const*loss2
        l2 = loss1
        loss = loss1 + loss2
        return loss.detach().cpu().numpy(), l2.detach().cpu().numpy(), loss2.detach().cpu().numpy(), output.detach().cpu().numpy(), pert_out.detach().cpu().numpy()

    def l2_attack(self, input, target, model, targeted, use_log, use_tanh, solver, reset_adam_after_found=True,abort_early=True,
        batch_size=256,max_iter=1000,const=0.01,confidence=0.0,early_stop_iters=100, binary_search_steps=9,
        step_size=0.01,adam_beta1=0.9,adam_beta2=0.999):
    
        early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iter // 10

        input = torch.from_numpy(input).to(self.estimator.device)                         
        target = torch.from_numpy(target).to(self.estimator.device)       

        var_len = input.view(-1).size()[0]
        modifier_up = np.zeros(var_len, dtype=np.float32)
        modifier_down = np.zeros(var_len, dtype=np.float32)
        real_modifier = torch.zeros(input.size(),dtype=torch.float32).to(self.estimator.device)                         
        mt = np.zeros(var_len, dtype=np.float32)
        vt = np.zeros(var_len, dtype=np.float32)
        adam_epoch = np.ones(var_len, dtype=np.int32)
        grad=np.zeros(batch_size,dtype=np.float32)
        hess=np.zeros(batch_size,dtype=np.float32)

        upper_bound=1e10
        lower_bound=0.0
        out_best_attack=input[0].clone().detach().cpu().numpy()
        out_best_const=const  
        out_bestl2=1e10
        out_bestscore=-1

        if use_tanh:
            input = 2*input - 1
            input = torch.atanh(input*0.99999)

        if not use_tanh:
            modifier_up = 0.5-input.clone().detach().view(-1).cpu().numpy()
            modifier_down = -0.5-input.clone().detach().view(-1).cpu().numpy()
        
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                if targeted:
                    x[y] -= confidence
                else:
                    x[y] += confidence
                x = np.argmax(x)
            if targeted:
                return x == y
            else:
                return x != y

        for step in range(binary_search_steps):
            bestl2 = 1e10
            prev=1e6
            bestscore=-1
            last_loss2=1.0
            # reset ADAM status
            mt.fill(0)
            vt.fill(0)
            adam_epoch.fill(1)
            stage=0
            
            for iter in range(max_iter):
                if (iter+1)%100 == 0:
                    loss, l2, loss2, _ , __ = self.loss_run(input,target,model,real_modifier,use_tanh,use_log,targeted,confidence,const)
                    # print("[STATS][L2] iter = {}, loss = {:.5f}, loss1 = {:.5f}, loss2 = {:.5f}".format(iter+1, loss[0], l2[0], loss2[0]))
                    # sys.stdout.flush()

                var_list = np.array(range(0, var_len), dtype = np.int32)
                indice = var_list[np.random.choice(var_list.size, batch_size, replace=False)]
                var = np.repeat(real_modifier.detach().cpu().numpy(), batch_size * 2 + 1, axis=0)
                for i in range(batch_size):
                    var[i*2+1].reshape(-1)[indice[i]]+=0.0001
                    var[i*2+2].reshape(-1)[indice[i]]-=0.0001
                var = torch.from_numpy(var)
                var = var.view((-1,)+input.size()[1:]).to(self.estimator.device)              
                losses, l2s, losses2, scores, pert_images = self.loss_run(input,target,model,var,use_tanh,use_log,targeted,confidence,const) 
                real_modifier_numpy = real_modifier.clone().detach().cpu().numpy()
                if solver=="adam":
                    self.coordinate_ADAM(losses,indice,grad,hess,batch_size,mt,vt,real_modifier_numpy,adam_epoch,modifier_up,modifier_down,step_size,adam_beta1,adam_beta2,proj=not use_tanh)
                if solver=="newton":
                    self.coordinate_Newton(losses,indice,grad,hess,batch_size,mt,vt,real_modifier_numpy,adam_epoch,modifier_up,modifier_down,step_size,adam_beta1,adam_beta2,proj=not use_tanh)
                real_modifier=torch.from_numpy(real_modifier_numpy).to(self.estimator.device)                         
                
                if losses2[0]==0.0 and last_loss2!=0.0 and stage==0:
                    if reset_adam_after_found:
                        mt.fill(0)
                        vt.fill(0)
                        adam_epoch.fill(1)
                    stage=1 
                last_loss2=losses2[0]

                if abort_early and (iter+1) % early_stop_iters == 0:
                    if losses[0] > prev*.9999:
                        # print("Early stopping because there is no improvement")
                        break
                    prev = losses[0]
                
                if l2s[0] < bestl2 and compare(scores[0], np.argmax(target.cpu().numpy(),-1)):
                    bestl2 = l2s[0]
                    bestscore = np.argmax(scores[0])

                if l2s[0] < out_bestl2 and compare(scores[0],np.argmax(target.cpu().numpy(),-1)):
                    if out_bestl2 == 1e10:
                    # print("[STATS][L3](First valid attack found!) iter = {}, loss = {:.5f}, loss1 = {:.5f}, loss2 = {:.5f}".format(iter+1, losses[0], l2s[0], losses2[0]))
                    # sys.stdout.flush()
                        pass
                    out_bestl2 = l2s[0]
                    out_bestscore = np.argmax(scores[0])
                    out_best_attack = pert_images[0]
                    out_best_const = const
            
            if compare(bestscore,  np.argmax(target.cpu().numpy(),-1)) and bestscore != -1:
                # print ('old constant: ', const)
                upper_bound = min(upper_bound,const)
                if upper_bound < 1e9:
                    const = (lower_bound + upper_bound)/2
                # print ('new constant: ', const)
            else:
                # print  ('old constant: ', const)
                lower_bound = max(lower_bound,const)
                if upper_bound < 1e9:
                    const = (lower_bound + upper_bound)/2
                else:
                    const *= 10
                # print  ('new constant: ', const)

        return out_best_attack, out_bestscore

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        # if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
        #     raise ValueError(
        #         "This attack has not yet been tested for binary classification with a single output classifier."
        #     )
        # def l2_attack(self, input, target, model, targeted, use_log, use_tanh, solver, reset_adam_after_found=True,abort_early=True,
        # batch_size=128,max_iter=2000,const=0.01,confidence=0.0,early_stop_iters=100, binary_search_steps=10,
        # step_size=0.01,adam_beta1=0.9,adam_beta2=0.999):
        # Compute adversarial examples with implicit batching
        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        r = []
        for batch_id in trange(nb_batches, desc="ZOO"):
            x_batch = x[batch_id*self.batch_size:(batch_id+1)*self.batch_size]
            y_batch = y[batch_id*self.batch_size:(batch_id+1)*self.batch_size]
            # print('go up to',len(inputs))
            # run 1 image at a time, minibatches used for gradient evaluation
            for i in range(len(x_batch)):
                attack, score=self.l2_attack(input=np.expand_dims(x_batch[i],0), target=np.expand_dims(y_batch[i],0), targeted=self.targeted,
                                            model=self.estimator.model, use_log=True, use_tanh=self.use_tanh, solver=self.solver, reset_adam_after_found=True, abort_early=True,
                                            batch_size=self.nb_parallel, max_iter=self.max_iter, const=self.initial_const, confidence=self.confidence,
                                            early_stop_iters=self.early_stop_iters, binary_search_steps=self.binary_search_steps, step_size=self.step_size, adam_beta1=self.adam_beta1,
                                            adam_beta2=self.adam_beta2)
                r.append(attack)
        return np.array(r)