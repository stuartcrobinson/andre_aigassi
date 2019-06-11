"""
goal: beat https://inoliao.github.io/CoachAI/

define and run CNN on cctf images (cctf = color-coded temporal filter)

how?  read training data as frames.

resize frame

create generator to pull the next frame and feed into model

model (try fewer filters than vgg and alexnet.  see modelOutlines_alexnet_vgg16.py)
        rules of thumb: https://www.reddit.com/r/MachineLearning/comments/3l5qu7/rules_of_thumb_for_cnn_architectures/

Conv2D(16, (3, 3)
Conv2D(16, (3, 3)
MaxPooling2D((4, 4), strides=(2, 2)

Conv2D(16, (3, 3)
Conv2D(16, (3, 3)
MaxPooling2D((4, 4), strides=(2, 2)

Conv2D(16, (3, 3)
Conv2D(16, (3, 3)
MaxPooling2D((4, 4), strides=(2, 2)




"""
