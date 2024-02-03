# Download the cifar10 dataset
mkdir -p data/cifar10
wget -P 'data/cifar10' https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
cd data/cifar10
tar -xzf cifar-10-python.tar.gz
mv cifar-10-batches-py/* .
rm -rf cifar-10-batches-py
rm cifar-10-python.tar.gz

# Download the cinic dataset
cd ../..
mkdir -p data/cinic
wget -P 'data/cinic' https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz
cd data/cinic
tar -xzf CINIC-10.tar.gz
rm CINIC-10.tar.gz
