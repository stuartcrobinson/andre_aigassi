"""
rules of thumb: https://www.reddit.com/r/MachineLearning/comments/3l5qu7/rules_of_thumb_for_cnn_architectures/

https://stats.stackexchange.com/questions/296027/choosing-filter-size-strides-etc-in-a-cnn
https://datascience.stackexchange.com/questions/25097/what-is-the-difference-between-8-filters-twice-and-one-16-filters-in-convolution
https://www.quora.com/Why-does-convolution-neutral-network-increment-the-number-of-filters-in-each-layer

? http://cs231n.github.io/convolutional-networks/

vgg16: https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py

Input(shape=input_shape)

Conv2D(64, (3, 3)
Conv2D(64, (3, 3)
MaxPooling2D((2, 2), strides=(2, 2)

Conv2D(128, (3, 3)
Conv2D(128, (3, 3)
MaxPooling2D((2, 2), strides=(2, 2)

Conv2D(256, (3, 3),
Conv2D(256, (3, 3),
Conv2D(256, (3, 3),
MaxPooling2D((2, 2), strides=(2, 2)

Conv2D(512, (3, 3),
Conv2D(512, (3, 3),
Conv2D(512, (3, 3),
MaxPooling2D((2, 2), strides=(2, 2)

classes=1000

Flatten(name='flatten')(x)
Dense(4096, activation='relu', name='fc1')(x)
Dense(4096, activation='relu', name='fc2')(x)
Dense(classes, activation='softmax', name='predictions')(x)

alexnet:    https://github.com/eweill/keras-deepcv/blob/master/models/classification/alexnet.py
            https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/
            https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637


Conv2D(96, (11, 11), input_shape=img_shape,
MaxPooling2D(pool_size=(2, 2)))

Conv2D(256, (5, 5),
MaxPooling2D(pool_size=(2, 2)))

ZeroPadding2D((1, 1)))
Conv2D(512, (3, 3),
MaxPooling2D(pool_size=(2, 2)))

ZeroPadding2D((1, 1)))
Conv2D(1024, (3, 3),

ZeroPadding2D((1, 1)))
Conv2D(1024, (3, 3),
MaxPooling2D(pool_size=(2, 2)))

Flatten())
Dense(3072))
Dropout(0.5))

Dense(4096))
Dropout(0.5))

Dense(n_classes))
Activation('softmax'))
"""