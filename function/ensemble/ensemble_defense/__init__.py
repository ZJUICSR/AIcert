import torch
import copy
from IOtool import IOtool
from function.ex_methods.module.module import Module

# class EnsembleModel(torch.nn.Module):
class EnsembleModel(Module):
    def __init__(self, model_list, device):
        super(EnsembleModel, self).__init__()
        assert len(model_list) > 0
        self.model_list = [copy.deepcopy(model) for model in model_list]
        self.device = device if device else IOtool.get_device()
        # self.model_list = torch.nn.ModuleList([copy.deepcopy(model) for model in model_list])
        self.soft_max = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = None
        for model in self.model_list:
            model.to(self.device)
            if out is None:
                out = model(x)
                continue
            out += model(x)
        out = self.soft_max(out)
        return out


def run(model_list, device = None):
    res_model = EnsembleModel(model_list=model_list, device=device)
    return res_model


if __name__ == '__main__':
    pass


