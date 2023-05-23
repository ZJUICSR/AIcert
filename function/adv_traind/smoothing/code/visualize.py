# visualize noisy images
from datasets import get_dataset, DATASETS
import torch
from torchvision.transforms import ToPILImage


def visualize(args_dict):
    toPilImage = ToPILImage()
    dataset = get_dataset(args_dict['dataset'], args_dict['split'])
    image, _ = dataset[args_dict['idx']]
    noise = torch.randn_like(image)
    for noise_sd in args_dict['noise_sds']:
        noisy_image = torch.clamp(image + noise * noise_sd, min=0, max=1)
        pil = toPilImage(noisy_image)
        pil.save('{}/{}_{}.png'.format(args_dict['outdir'], args_dict['idx'], int(noise_sd * 100)))


if __name__ == '__main__':
    args_dict = {
        'dataset': 'cifar10',
        'outdir': '/data/user/WZT/Datasets/smoothing/figures/example_images/cifar10',
        'idx': 10,
        'noise_sds': 10,
        'split': 'test'

    }
    visualize(args_dict)
