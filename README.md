This is a feature that uses a Convolutional Neural Network (CNN) algorithm to detect recolored images using deep learning. 
It is trained on a dataset of images and uses image classification techniques to determine whether an image has been recolored or not.

**Project Description**
This project is built based on CNN model using Keras and TensorFlow to determine whether an image is recolored or not. 

**Key Features:**
For image classification, Convolutional Neural Network algorithm. 
Image augmentation during training.
Prediction of image classes whether it is recolored or not after training.

**Requirements**
Python
TensorFlow for handling the computational tasks during model training and prediction.
Keras for building the CNN model
NumPy for numerical operations
pandas for data handling
Matplotlib used to display images
PIL (Pillow) for image manipulation
Scipy - scipy.ndimage for filters

**Directory Structure:**
/feature/training: It is the directory containing training images, organized by class ("recolor" and "original").
/feature/testing: It is the directory containing test images.

**CNN model architecture includes:**
Convolution Layers: Extracting features from input images (32 filters with 3x3 kernel, using ReLU activation).
Max-Pooling Layers: Reducing spatial dimensions (2x2 pool size).
Flattening: Flattening the pooled feature maps into a 1D vector.
Fully Connected Layers: Dense layers for classification, with a final output layer that uses the softmax activation function for multi-class classification (2 classes in this case).
Compilation: The model is compiled using Adam optimizer, categorical cross-entropy loss function, and accuracy metric.

**Data Augmentation and Image Preprocessing:**
The model is trained using the fit_generator method, which takes images from the respective training and testing directories.
It performs training for 10 epochs, with 40 training steps per epoch, and validation using 10 validation steps.
The function returns the training dataset and the trained model.

**Training the CNN:**
The cnn_model function is called with a specified path, which trains the CNN model on the provided image data with separate training and testing directories.
The function returns training_set and the trained CNN model.
Using Keras' ImageDataGenerator, image augmentation is applied to the training dataset while the test data is only rescaled.

**Training the Model:**
The model is trained using the fit_generator method, which takes images from the respective training and testing directories.
It performs training for 10 epochs, with 40 training steps per epoch, and validation using 10 validation steps.
The function returns the training dataset and the trained model.

**Image Prediction:**
After training, a specific test image (in the code, 'recolor.17.jpg') is loaded and preprocessed.
The image is resized to 64x64 pixels.
It is converted into an array using img_to_array.
The shape of the array is adjusted with np.expand_dims to match the expected input shape for the model.

**Prediction:**
The trained model is used to predict the class of the test image using the predict_classes method.
The predicted class is returned and printed as the output.

