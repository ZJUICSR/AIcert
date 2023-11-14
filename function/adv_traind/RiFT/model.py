# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .models import *


def create_model(model_name='ResNet18', num_classes=10, device='cuda', patch_size=4, resume=None):
	input_size = 32
	if model_name == "ResNet34":
		model = ResNet34(input_size, num_classes)
	elif model_name == "ResNet18":
		model = ResNet18(input_size, num_classes)
	elif model_name == "ResNet50":
		model = ResNet50(input_size, num_classes)
	elif model_name == "ResNet152":
		model = ResNet152(input_size, num_classes)
	elif model_name == "DenseNet":
		model = DenseNet121(input_size, num_classes)
	elif model_name == "DenseNet50":
		model = DenseNet50(input_size, num_classes)
	elif model_name == "VGG19":
		model = VGG("VGG19", input_size, num_classes)
	elif model_name == "WideResNet34":
		model = WideResNet(image_size=input_size, depth=34, widen_factor=10, num_classes=num_classes)
	elif model_name == "WideResNet28":
		model = WideResNet(image_size=input_size, depth=28, widen_factor=10, num_classes=num_classes)
	elif model_name == "WideResNet22_2":
		model = WideResNet(image_size=input_size, depth=22, widen_factor=2, num_classes=num_classes)
	elif model_name == 'WideResNet34_5':
		model = WideResNet(image_size=input_size, depth=34, widen_factor=5, num_classes=num_classes)

	model = model.to(device)

	if device == 'cuda':
		model = torch.nn.DataParallel(model)

	if resume is not None:
		print(resume)
		checkpoint = torch.load(resume)
		# print(checkpoint.items())
		# exit()
		if "net" in checkpoint.keys():
			model.load_state_dict(checkpoint["net"])
		elif "state_dict" in checkpoint.keys():
			model.load_state_dict(checkpoint["state_dict"])
		elif "model" in checkpoint.keys():
			model.load_state_dict(checkpoint["model"])
		else:
			model.load_state_dict(checkpoint)

	return model

if __name__ == '__main__':
	net = create_model(model_name='DenseNet50', num_classes=10, device='cpu', patch_size=4, resume=None)
	print(net)



