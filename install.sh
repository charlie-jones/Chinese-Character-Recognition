#!/bin/bash

python3 -m venv env

source env/bin/activate

pip3 install --upgrade pip
pip3 install Flask
pip install pinyin
pip install numpy



#deactivate 