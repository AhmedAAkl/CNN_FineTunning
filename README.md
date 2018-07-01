# CNN_FineTunning in keras with pretrained imagenet weights

## Training deep neural networks from stratch have some problems ##

1. Take weeks and even months using powerful GPUs in Google or Microsoft Servers.
2. Need huge data set
3. Millions of parameters

## To overcome these problems a method called Transfer learning or Fine tuning ## 
Used to train Deep Neural Nets with small dataset, even in CPUs and only a few thousands of parameters will be trained.


We will learn how to use state-of-the-art Deep Learning models to solve a Supervised Image Classification problem using our own datasets with/without GPU acceleration.

Pre-trained Deep Neural Nets trained on the ImageNet challenge are made public and available in Keras.

###### ImageNet ###### 
is a huge image dataset used to help researchers and educators in computer vision track.
you can check it from here http://www.image-net.org/


** The main idea based on first or ealier layers extract the general features of any objects **
So we can use this feature and using the depth or last layers to extract the specific features of our objects we want to classify or recognize.

The pre-trained models we will consider are VGG16, VGG19, Inception-v3, Xception, ResNet50, InceptionResNetv2 and MobileNet. 


These models for binary classification and currently i will adjust it for multiclassification also.

The code is easy and simple so, you can edit it to run on your owen dataset.

# still preparing other models
