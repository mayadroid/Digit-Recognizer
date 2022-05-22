# Digit-Recognizer

I. Summary

We build a CNN model using Tensorflow framework on MNIST dataset for Digit Recognition. We chose the deep learning method for building the model because they are self learning which makes them better at recognizing images more accurately. Out of the many iterations we tried, we got better results (accuracy of 99.482%) at 20 epochs and batch size of 60.

II. Methods used: 

1.	Importing Libraries and Data:
We started by importing necessary python libraries and libraries for the Keras and Pytorch framework on Google Collab. MNIST data was accessed from Kaggle for both Keras and Pytorch. Data consisted of separate files for train and test data. 

2.	Data Preparation:
●	Training data was then divided into training set and validation set
●	Checking for Nulls in train and test data 
●	Reshaped train and test data into a 28x28x1 matrix. We used a single channel because the input is a grayscale image.
●	Encoded the labels on validation data to one hot encoded vectors using to_categorical function

3.	Model Building:
We used Sequential API where we started adding filter layers one after the other. 
After trying a variety of layers in our model, we settled on using the following Keras layers: 
Conv2D : Convolutional layers summarize the presence of features in the input by creating feature maps.
Maxpool2D : This layer helps with down-sampling. It picks maximum value from neighboring pixels. It helps in reducing the effect on change in input image due to rotation or other elemental changes and helps increase the classification accuracy.
BatchNormalization : Works like any normalization method which in turn,helps in speeding up the training process between the layers.
Dropout : Regularization method which helps the model to learn better and thus reduce overfitting
Flatten : Converts feature image into 1D vector
Dense : Artificial neural network classifier used at the end of the model 
We used the activation functions relu and softmax as they worked best for CNN models.

4.	Model Compilation: 
Once we added our layers to the model, we had to set up our model optimizer, loss function and metric function. We used “adam” as our model optimizer as it is a very efficient optimizer in training deep learning models. We chose a loss function that is used for categorical classifications called the “categorical_crossentropy” because we are dealing with one hot encoded labels of the validation dataset instead of integer encoded ones. 
Metric function “accuracy” was chosen to evaluate the performance of our model.

5.	Setting learning rate annelear:
We then set up a learning rate annelear to tune our model. Since the learning rate is an important parameter in neural networks that requires quite a bit of tuning, the annelear comes into the picture here where it helps make the optimizer converge faster and to efficiently reach the global minimum of the loss function. We used the ReduceLROnPlateau function to reduce the learning rate by 30% if the accuracy has not improved after 3 epochs.

6.	Augmentation of Data:
In order to avoid the overfitting problem, we decided to modify our training data slightly by using transformations such as rotate, zoom, shift and flip. This is done to mimic the variations of someone actually writing down the digits. This helps us expand our model by covering all bases thereby creating a robust model.

7.	Setting epochs and batchsize:
After trying a multitude of different values and ranges, we settled on setting 20 epochs with a batch size of 60 to give us the best model performance.

8.	Fitting the model:
We then proceeded to fit the model to our training dataset. At the end of 20 epochs, we observed a validation accuracy of 99.47%.

9.	Plotting the Loss and Accuracy: 
We plot the loss and accuracy of the training and validation data to observe how well the model fits. We are more interested in observing performance and trend of the validation data. 

10.	Predicting results:
Finally, we ran the model on the test dataset and predicted the results. We then assigned the labels with the highest probabilities to the test data and compiled the same into a csv file for submission.

III. Results and findings:

We tested out several variations of our model to identify the best hyperparameters for optimal accuracy. The third version of our model proved to be the best upon the addition of the BatchNormalization() layer. This is where we identified the best epochs to be 20 and best batch size to be 60.
After making the necessary modifications, the training accuracy after 20 epochs was observed to be 99.47%.
Upon deploying the model on the test dataset, we got an improved accuracy of 99.482%.
