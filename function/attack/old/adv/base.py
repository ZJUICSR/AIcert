import torch


class Base(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_training_mode`.
    """
    def __init__(self, name, model, bounds=[0, 1]):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """
        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]
        self.device = next(model.parameters()).device

        self._training_mode = False
        self._transform_label = self._get_label
        self._targeted = -1
        self._attack_mode = 'default'
        self._return_type = 'float'
        self._kth_min = 1
        self.bounds = bounds
        self.steps = 1
        self.eps = 0

    def set_bounds(self, bounds):
        self.bounds = bounds

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
            
    def get_mode(self):
        r"""
        Get attack mode.

        """
        return self._attack_mode
        
    def set_mode_default(self):
        r"""
        Set attack mode as default mode.

        """
        if self._attack_mode == 'only_default':
            self._attack_mode = "only_default"
        else:
            self._attack_mode = "default"
            
        self._targeted = -1
        self._transform_label = self._get_label
        
    def set_mode_targeted(self, target_map_function=None):
        r"""
        Set attack mode as targeted mode.
  
        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda images, labels:(labels+1)%10.
                None for using input labels as targeted labels. (DEFAULT)

        """
        if self._attack_mode == 'only_default':
            raise ValueError("Changing attack mode is not supported in this attack method.")
            
        self._attack_mode = "targeted"
        self._targeted = 1
        if target_map_function is None:
            self._target_map_function = lambda images, labels:labels
        else:
            self._target_map_function = target_map_function
        self._transform_label = self._get_target_label
        
    def set_mode_least_likely(self, kth_min=1):
        r"""
        Set attack mode as least likely mode.
  
        Arguments:
            kth_min (str): k-th smallest probability used as target labels (DEFAULT: 1)

        """
        if self._attack_mode == 'only_default':
            raise ValueError("Changing attack mode is not supported in this attack method.")
            
        self._attack_mode = "least_likely"
        self._targeted = 1
        self._transform_label = self._get_least_likely_label
        self._kth_min = kth_min
        
    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.

        Arguments:
            type (str): 'float' or 'int'. (DEFAULT: 'float')

        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")

    def set_training_mode(self, flag):
        r"""
        Set training mode during attack process.

        Arguments:
            flag (bool): True for using training mode during attack process.

        """
        self._training_mode = flag

    def save(self, data_loader, save_path=None, verbose=True):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (DEFAULT: True)

        """
        if (self._attack_mode == 'targeted') and (self._target_map_function is None):
            raise ValueError("save is not supported for target_map_function=None")
        
        if save_path is not None:
            image_list = []
            label_list = []

        correct = 0
        total = 0
        l2_distance = []
        
        total_batch = len(data_loader)

        training_mode = self.model.training
        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            batch_size = len(images)
            
            if save_path is not None:
                image_list.append(adv_images.cpu())
                label_list.append(labels.cpu())

            if self._return_type == 'int':
                adv_images = adv_images.float()/255

            if verbose:
                with torch.no_grad():
                    if training_mode:
                        self.model.eval()
                    outputs = self.model(adv_images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (predicted == labels.to(self.device))
                    correct += right_idx.sum()
                    
                    delta = (adv_images - images.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))
                    acc = 100 * float(correct) / total
                    print('- Save Progress: %2.2f %% / Accuracy: %2.2f %% / L2: %1.5f' \
                          % ((step+1)/total_batch*100, acc, torch.cat(l2_distance).mean()), end='\r')

        if save_path is not None:
            x = torch.cat(image_list, 0)
            y = torch.cat(label_list, 0)
            torch.save((x, y), save_path)
            print('\n- Save Complete!')

        if training_mode:
            self.model.train()
        
    def _get_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return labels
    
    def _get_target_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return self._target_map_function(images, labels)
    
    def _get_least_likely_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        if self._kth_min < 0:
            pos = outputs.shape[1] + self._kth_min + 1
        else:
            pos = self._kth_min
        _, labels = torch.kthvalue(outputs.data, pos)
        labels = labels.detach_()
        return labels
    
    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images*255).type(torch.uint8)

    def _switch_model(self):
        r"""
        Function for changing the training mode of the model.
        """
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def __str__(self):
        info = self.__dict__.copy()
        
        del_keys = ['model', 'attack']
        
        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)
                
        for key in del_keys:
            del info[key]
        
        info['attack_mode'] = self._attack_mode
        if info['attack_mode'] == 'only_default':
            info['attack_mode'] = 'default'
            
        info['return_type'] = self._return_type
        
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        training_mode = self.model.training

        if self._training_mode:
            self.model.train()
        else:
            self.model.eval()

        images = self.forward(*input, **kwargs)

        if training_mode:
            self.model.train()
        
        if self._return_type == 'int':
            images = self._to_uint(images)

        return images
