# Face Frontalization Using Multi-Task Deep Neural Networks
Luann Jung, Dylan Zhou

Implementation of 2-task and 3-task neural networks for face frontalization in Tensorflow.

The **.py** files can be run to begin training. For experimentation with hyperparameters and the different networks, we suggest referencing the **.ipynb** version.

A (very small) sample dataset of 10 images is included as **10imgs.zip**.
Loss values are outputted as plaintext as the model trains, but a util for parsing the text to plot loss over epochs is included in **loss_plotting.ipynb**.

## Dependencies and Requirements

tensorflow_io
tensorflow_model_optimization

Checkpoint file for LightCNN: [here](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view)