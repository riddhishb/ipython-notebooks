#!/bin/bash

echo "Installing necessary packages for Python"
sudo apt-get install python-pip python-dev build-essential
sudo pip install --upgrade pip
pip install numpy
pip install matplotlib
sudo apt-get install python-opencv

echo "Installing Jupyter"
pip install jupyter
echo "Installing ipython shell"
sudo apt-get install ipython

echo "Now Launching jupyter notebook and setting up localhost"
jupyter notebook
