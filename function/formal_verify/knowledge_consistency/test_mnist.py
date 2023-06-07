import torchvision
from skimage import io
import scipy.misc
train_data = torchvision.datasets.MNIST(
        "/mnt/data/AISec/backend/data", train=False, download=True)
image_array,_=train_data[0]
# print(image_array)
# io.imsave("mnist.png",image_array.numpy())
image_array.save("mnist.png")