#!/bin/bash
# download_mnist_data.sh

mkdir -p data
cd data

echo "Downloading MNIST dataset..."

# Download MNIST dataset
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

# Extract files
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

echo "MNIST dataset downloaded and extracted to data/"
