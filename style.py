# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import glob
import os
import sys

import numpy as np
from PIL import Image
from chainer import cuda, Variable, serializers
from net import FastStyleNet

import settings

class Style(object):
    def __init__(self, model_path):
        self._model_path = model_path
        self._model = None

    @property
    def model_path(self):
        return self._model_path

    @property
    def name(self):
        return os.path.splitext(os.path.basename(self._model_path))[0]

    def _load(self):
        model = FastStyleNet()
        serializers.load_npz(self._model_path, model)
        if settings.GPU >= 0:
            cuda.get_device(settings.GPU).use()
            model.to_gpu()
        return model

    def preload(self):
        self._model = self._load()

    def unload(self):
        self._model = None

    def stylize(self, image, noise=0):
        model = self._model if self._model else self._load()

        xp = np if settings.GPU < 0 else cuda.cupy

        image = xp.asarray(image.convert('RGB'), dtype=xp.float32).transpose(2, 0, 1)
        image = image.reshape((1,) + image.shape)
        x = Variable(image)

        y = model(x)
        result = cuda.to_cpu(y.data)

        result = result.transpose(0, 2, 3, 1)
        result = result.reshape((result.shape[1:]))
        result = np.uint8(result)
        result = result.copy() # ensure contiguous data in memory
        return result

def load_styles():
    return [Style(file) for file in glob.glob(os.path.join(settings.MODEL_DIR, '*.model'))]
