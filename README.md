Detecting Nokia3310 mobile phone using CNTK and Python

## Overview

This example is modified version of predefined CNTK Example of Image detection which can be found at http://github.com/Microsoft/CNTK.

By running `Nokia3310_detection.py` the code will do the following:

* Train and create CNTK based model whih can detect NOkia3310 on an image.
* By providing test images the model can be evaluated and tested.
* The example uses minimal python code, needed to run object detection by using FasterRCNN.
* The model is not trained from zero, it is based on AlexNet pre trained model.


## Running the example

### Setup

To run Nokia3310 object detection example you need a CNTK 2.3, Python 3.5 environment. Beside the basics requiremens you need to install the following additional packages:

```
pip install opencv-python easydict pyyaml future
```

The also code uses prebuild Cython modules for parts of the region proposal network (see `utils/cython_modules`). 

If you want to use the debug output you need to run `pip install pydot_ng` ([website](https://pypi.python.org/pypi/pydot-ng)) and install [graphviz](http://graphviz.org/) to be able to plot the CNTK graphs (the GraphViz executable has to be in the system’s PATH).

### Getting the data and AlexNet model

The example uses the pre-trained AlexNet model which can be downloaded by running the following Python command from the PretrainedModels folder:

`python download_model.py`

### Running the demo

To train and evaluate a detector run

`python Nokia3310_detection.py`

#### Changing the data set

In order to change DataSet you have to provide the images and data. More information about how to prepare image and data can be found at http://bhrnjica.net  


#### Changing the base model

Changing base model is not supported. 
