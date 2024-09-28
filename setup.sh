#!/bin/bash

# Update and upgrade the system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and pip
sudo apt-get install python3 python3-pip -y

# Install virtual environment
sudo apt-get install python3-venv -y

# Install Poppler
sudo apt-get install poppler-utils -y

# Create a directory for the application
mkdir -p ~/VLM
cd ~/VLM

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Clone the repository (assuming it's on GitHub, replace with your actual repo URL)
git clone https://github.com/sanjaygbhat/VLM.git .

# Install required packages
pip install -r requirements.txt

# Set up environment variables
echo "export SECRET_KEY=iceybella" >> ~/.bashrc
echo "export DATABASE_URL='sqlite:///your_database.db'" >> ~/.bashrc
echo "export ANTHROPIC_API_KEY=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" >> ~/.bashrc
source ~/.bashrc

# Create necessary directories
mkdir -p uploads indices

# Initialize the database
flask db init
flask db migrate
flask db upgrade

# Run the application
echo "Setup complete. You can now run the application using:"
echo "python run.py"