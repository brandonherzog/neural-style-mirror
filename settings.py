# -*- coding: utf-8 -*-
import os
import sys

# run on cpu
GPU = -1
# gpu to use
#GPU = 0

# webcam to use
WEBCAM = 0

# available resolutions
SIZES = [
    '160x120',
    '240x180',
    '320x240',
    '480x360',
    '640x480',
    '960x720',
    #'1280x960',
    ]

# rotate captured image
ROTATE_IMAGE = 0 # degrees

# if kiosk display in fullscreen and hide minimize button
KIOSK = False

# char keys available as shortcuts for selecting model
STYLE_SHORTCUTS = '1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# directory of models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# hardcoded location for chainer-fast-neuralstyle 
# https://github.com/yusuketomoto/chainer-fast-neuralstyle
NEURAL_STYLE_ROOT = os.path.join(os.path.dirname(__file__), 'chainer-fast-neuralstyle')
if NEURAL_STYLE_ROOT:
    sys.path.append(NEURAL_STYLE_ROOT)

# hardcoded path to CUDA
CUDA_PATH = ''
if CUDA_PATH:
    os.environ['PATH'] = CUDA_PATH + os.pathsep + os.environ['PATH']

# hardcoded path to C compiler
if sys.platform == 'win32':
    # at this time Visual Studio 2015 is not supported by CUDA
    CCOMPILER_PATH = r'C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin'
else:
    CCOMPILER_PATH = ''
if CCOMPILER_PATH:
    os.environ['PATH'] = CCOMPILER_PATH + os.pathsep + os.environ['PATH']

# function(window, qimage) that will handle enter event to capture image
CAPTURE_HANDLER = 'savedialog.handle_capture'