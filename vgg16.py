# -*- coding: utf-8 -*-


from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


from keras import backend as K
K.set_image_dim_ordering("th")
from sklearn.metrics import log_loss


def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    """VGG 16 Model for Keras

    ImageNet Pretrained Weights
    
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of categories for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channel, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights('/imagenet_models_weights/vgg16_weights_th_dim_ordering_th_kernels.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
#    for layer in model.layers[:10]:
#        layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':


    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 1
    batch_size = 32
    nb_epoch = 10

    # specify the cat-dog directories
    
    train_data_dir = ""
    valid_data_dir = ""
    
    # load vgg16 model
    model = vgg16_model(img_rows, img_cols, channel, num_classes)

     # uncomment to print layers index and names
    
#    for i,layer in enumerate(model.layers):
#        print(i,layer)

    # uncomment to print layers  and training status
#    for layer in model.layers:
#        print(layer,layer.trainable)
#        
        
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen.flow_from_directory(directory=train_data_dir,
                                                  target_size=(img_rows,img_cols),                                                  
                                                  class_mode='binary',
                                                  batch_size=batch_size
                                                  )
    
    validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
                                                       target_size=(img_rows,img_cols),                                                       
                                                       class_mode='binary',
                                                       batch_size=batch_size)                                                       
    
    train_images_num = len(train_generator.filenames)
    valid_images_num = len(validation_generator.filenames)
    
    file_path = "output_dir/vgg16.h5"
    early_stopping = EarlyStopping(monitor='val_acc',patience=2,verbose=0,mode='auto')
    checkpoint = ModelCheckpoint(file_path,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callbacks_list = [checkpoint,early_stopping]
    
    
    # Start Fine-tuning
    
    start = time.time()
    model_history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_images_num//batch_size,
                        epochs=nb_epoch,
                        callbacks = callbacks_list,
                        validation_data=validation_generator,
                        validation_steps=valid_images_num//batch_size)

    end = time.time()
    training_time = end - start
    print(training_time)
    
    print("all weights are saved properly")
    
