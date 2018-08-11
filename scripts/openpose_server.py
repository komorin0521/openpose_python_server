#!/usr/env python
# -*- coding: utf-8 -*-

import argparse
import configparser
import datetime
import io
import sys
import os

import cv2
from flask import Flask, request, redirect, jsonify
from flask import send_file
from werkzeug import secure_filename
from pykakasi import kakasi

from openpose import *

class OpenPoseServer(Flask):
    def __init__(self, host, name, upload_dir, extensions, model_dir):
        """
        init server class
        """
        super(OpenPoseServer, self).__init__(name)
        self.host = host
        self.config['UPLOAD_FOLDER'] = upload_dir
        self.extensions = extensions
        # self.pose_detector = pose_detector
        self.model_dir = model_dir
        self.converter = None
        self.define_uri()

    def define_uri(self):
        """
        definition of uri
        """
        self.provide_automatic_option = False
        self.add_url_rule('/get_predict_image', None, self.get_predict_image, methods = ['POST'])

    def setup_converter(self):
        """
        """
        mykakasi = kakasi()
        mykakasi.setMode('H', 'a')
        mykakasi.setMode('K', 'a')
        mykakasi.setMode('J', 'a')
        self.converter = mykakasi.getConverter()

    def convert_filename(self, filename):
        """
        converting filename using pykakasi
        """
        return self.converter.do(filename)

    def check_allowfile(self, filename):
        """
        checking extenson
        """
        if len(filename.split(".")) > 1:
            extension = filename.split(".")[-1]
            print("extension is %s" % extension)
            return extension in self.extensions
        else:
            return False

    def get_predict_image(self):
        """
        Getting yolo result
        """
        print("call api of get_predict_image")
        if request.method == 'POST':
            file = request.files['file']
            if file and self.check_allowfile(file.filename):
                print("receive the file, the filename is %s" % file.filename)
                output_filename = "%s_%s" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), self.convert_filename(file.filename))
                print("output filename is %s" % output_filename)
                outputfilepath = os.path.join(self.config['UPLOAD_FOLDER'], output_filename)
                file.save(outputfilepath)
    
                try:
                    img = cv2.imread(outputfilepath)
                except Exception as err:
                    print(err)

                try:
                    pose_detector = create_openpose_instance(self.model_dir)
                    _, pred_img  = pose_detector.forward(img, True)
                except Exception as err:
                    print(err)

                tmpfilename = output_filename.split(os.path.sep)[-1]
                pred_outputfilename = "%s_pred.jpg" % tmpfilename.split('.')[0]
                pred_img_outputfilepath = os.path.join(self.config['UPLOAD_FOLDER'], pred_outputfilename)

                print("pred img outputfilepath: %s" % pred_img_outputfilepath)
                cv2.imwrite(pred_img_outputfilepath, pred_img)

                with open(pred_img_outputfilepath, 'rb') as img:
                    return send_file(io.BytesIO(img.read()),
                            attachment_filename=pred_outputfilename,
                            mimetype='image/%s' % pred_outputfilename.split('.')[-1])

        else:
            res = dict()
            res['status'] = '500'
            res['msg'] = 'The file format is only jpg or png'

def importargs():
    """
    importing args
    """

    parser = argparse.ArgumentParser('This is a server of darknet')
    parser.add_argument("--cfgfilepath", "-cf", help = "config filepath", type=str, required=True)
    args = parser.parse_args()

    return args.cfgfilepath

def readconf(conffilepath):
    """
    read config
    """
    
    config = configparser.ConfigParser()
    config.read(conffilepath)
    try:
        host = config.get("Server", "host")
        port = config.getint("Server", "port")
        upload_dir = config.get("Server", "upload_dir")
        model_dir = config.get("OpenPose", "model_dir")
    except configparser.Error as config_parser_err:
        raise config_parser_err

    return host, port, upload_dir, model_dir


def create_openpose_instance(model_dir):
    """
    creating openpose instance
    """

    try:
        params = dict()
        params["logging_level"] = 3
        params["output_resolution"] = "-1x-1"
        params["net_resolution"] = "-1x368"
        params["model_pose"] = "BODY_25"
        params["alpha_pose"] = 0.6
        params["scale_gap"] = 0.3
        params["scale_number"] = 1
        params["render_threshold"] = 0.05
        # If GPU version is built, and multiple GPUs are available, set the ID here
        params["num_gpu_start"] = 0
        params["disable_blending"] = False
        # Ensure you point to the correct path where models are located
        params["default_model_folder"] = model_dir
        # Construct OpenPose object allocates GPU memory
        openpose = OpenPose(params)
        return openpose
    except Exception as err:
        print(err)
        print(type(err))
        raise err
       

def main():
    """
    main
    """
    
    cfgfilepath = importargs()
    try:
        host, port, upload_dir, model_dir = readconf(cfgfilepath)

    except configparser.Error as config_err:
        print(config_err)
        sys.exit(1)



    inputfilepath = "/home/yusuke/darknet_ws/darknet/data/person.jpg"
    img = cv2.imread(inputfilepath)
    # img, keypoints = pose_detector.forward(img, True)
    

    server = OpenPoseServer(host, 'openpose_server', upload_dir, ['jpg', 'png'], model_dir)
    server.setup_converter()
    print("server run")
    server.run(host=host, port=port)

if __name__ == "__main__":
    main()
