# neural-style-mirror
A Python-based live webcam demo of real time artistic style transfer using neural networks. The user can select between several different artistic styles and see the style applied to his or her live webcam video stream in near-realtime. Styled images from the live stream image can also be captured and shared.

A kiosk demo has been on display at [Booz Allen Hamilton's Innovation Center](http://www.boozallen.com/consulting/strategic-innovation/dcinnovationcenter).

![Sample Screenshot](/images/screenshot.png)
*Screenshot in windowed mode.*

# Cloning
Perform a recursive clone option to automatically pull the [chainer-fast-neuralstyle](https://github.com/yusuketomoto/chainer-fast-neuralstyle) submodule.
```
git clone --recursive https://github.com/booz-allen-hamilton/neural-style-mirror.git
```

# Running
```
python nsmirror.py
```
Settings can be configured by editing `settings.py`.

# Models
Several style models trained specifically for this demo can be found in the `models` folder.

Any model compatible with [chainer-fast-neuralstyle](https://github.com/yusuketomoto/chainer-fast-neuralstyle) can be added to the `models` folder. A collection of additional style models trained by others can be found at: [chainer-fast-neuralstyle-models](https://github.com/gafr/chainer-fast-neuralstyle-models).

# Requirements
This has been tested successfully with Python 2.7 on Linux and Windows.
* See `requirements.txt` for requirements that can be installed using pip/conda.
* A webcam compatible with the installed version of OpenCV.
* To enable GPU mode, [CUDA](https://developer.nvidia.com/cuda-downloads) and [CUDNN](https://developer.nvidia.com/cudnn]) must be properly installed. Set GPU to the desired GPU in `settings.py`. Without enabling GPU mode, this can be VERY slow.

# Reference
The artistic styling methodology using feed-forward convolutional neural networks was proposed in ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf) by Justin Johnson, Alexandre Alahi, and Li Fei-Fei.

This demo utilizes Yusuke Tomoto's [chainer-fast-neuralstyle](https://github.com/yusuketomoto/chainer-fast-neuralstyle) implementation of the artistic style transfer algorithm.

Initial inspiration drawn from Gene Kogan's [CubistMirror](https://github.com/genekogan/CubistMirror) installation on display at the [alt-AI](http://www.alt-ai.net) conference.