#!/bin/bash

# Update and upgrade the system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and pip
sudo apt-get install python3 python3-pip -y

# Install virtual environment
sudo apt-get install python3-venv -y

# Install Poppler
sudo apt-get install poppler-utils -y

# Install CUDA (Ensure compatibility with PyTorch)
# Check if CUDA is already installed
if ! command -v nvcc &> /dev/null
then
    echo "Installing CUDA..."
    # Replace with the correct CUDA version URL
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
    echo "CUDA installation completed."
else
    echo "CUDA is already installed."
fi

# Install cuDNN (Ensure it matches the CUDA version)
# Replace with the appropriate version matching CUDA
CUDNN_VERSION=8.6.0.163
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.6.0/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
tar -xf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
echo "cuDNN installation completed."

# Create a directory for the application
mkdir -p ~/VLM
cd ~/VLM

# Clone the repository (replace with your actual repo URL)
git clone https://github.com/sanjaygbhat/VLM.git .

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Set up environment variables
echo "export SECRET_KEY=iceybella" >> ~/.bashrc
echo "export DATABASE_URL='sqlite:///your_database.db'" >> ~/.bashrc
# Set other necessary environment variables here
source ~/.bashrc

# Create necessary directories
mkdir -p uploads indices

# Initialize the database
flask db init
flask db migrate
flask db upgrade

# Provide instruction to the user
echo "Setup complete. You can now run the application using:"
echo "python run.py"