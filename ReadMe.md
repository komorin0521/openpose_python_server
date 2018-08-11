# Overview
This is the python script, which is server of detection using openpose.

OpenPose is as following:
    - https://github.com/CMU-Perceptual-Computing-Lab/openpose


This server has only one API.

1. Getting the image which is embedded the pose results
    URI: '/get_predict_image'

# SoftWare
- Python: 3.x

    I test only 3.5.2

- openpose revision: f49e18421da832ae441f75477035786126357401

# Setup
1. Install openpose
   Install openpose, reading the website of darknet
   Please build with python API

2. Clone of this repository

    `$ git clone https://github.com/komorin0521/openpose_python_server.git`

3. Modify config of `openpose_python_server/config/openpose_server.ini`

    Please modify the model path and so on

4. Install python module using pip
    I recommended virtual python environment, for example pyenv + pyenv-virtualenv

    `$ (sudo) pip3 install -r requirements.txt`

5. Running server
    ./run_server.sh

6. Check the server response from other terminal
    ```bash
    $ curl -XPOST -F file=@(path_to_image_file) http://localhost:3001/get_predict_image > pred_image.jpg`

    eog pred_image.jpg
    ```
