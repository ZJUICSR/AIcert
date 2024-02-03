## Requirements
* Python 3.6+
* PyTorch 1.0+
* h5py
* tensorboardX

## Data Preparation
First download and unzip the CIFAR-10 and CINIC-10 by running the script `download.sh`

Then manually download the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), put `Anno` into `data/celeba/Anno`, `Eval` into `data/celeba/Eval`, put all align and cropped images to `data/celeba/images`

Run the `preprocess_data.py` to generate data for all experiments (this step involves creating h5py file for CelebA images, so would take some time 1~2 hours)
