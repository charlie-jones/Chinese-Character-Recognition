#!/bin/bash

python3 -m venv env

source env/bin/activate

pip3 install --upgrade pip
pip3 install Flask
pip3 install pinyin
pip3 install python-mnist
pip3 install numpy



#deactivate 