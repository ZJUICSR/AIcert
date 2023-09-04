import torchvision.datasets as datasets
import torchvision.transforms as T

trainset = datasets.CIFAR10(root="/mnt/data/AISec/backend/data", train=False,
  download=True)

for i in range(100):
    img,label=trainset[i]
    img.save(f"demo_image/cifar10/{label}_{i}.png")


trainset = datasets.MNIST(root="./data", train=False,
  download=True)

for i in range(100):
    img,label=trainset[i]
    img.save(f"demo_image/mnist/{label}_{i}.png")

