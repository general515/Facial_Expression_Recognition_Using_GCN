# Facial_Expression_Recognition_GCN

Highlights: 

(1) This model is very light, only 4 convolutional layers and 2 FC layers (including the FC for softmax loss).

(2) Recognition accuracy (pretrained on AffectNet database): 73.36 at FER2013 (test set), 88.91 at FERPlus (test set), 88.92 at RAF-DB.

(3) Very speed (9.0 s per epoch for training on RAF-DB) and lowest resoure requirments.


Note that:

Requires setup the package for Gabor Convolutional networks https://github.com/jxgu1016/Gabor_CNN_PyTorch

'main_train_test.py' provides training and testing. Hyperparams using the default values.

'test_demo.py'  is testing file with trained models (weights) in the directory 'trained_weights'

The directory 'test_demo_without_installing_GCN'  is a tesing demo without instaling GCN package

The RAF-DB can be found at http://www.whdeng.cn/RAF/model1.html#dataset

FER2013 and FERPlus can be found at 
Link：https://pan.baidu.com/s/1265rT59qoUW7AQkaV9DobQ 
password：1111

Please revise path for database
