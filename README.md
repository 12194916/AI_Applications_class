# AI_Applications_class
```
I updated all week assignments and put them into MidTerm file. I wrote explanations to the each line of the code. 
Then, based on the materials I leaned from the course, I made a model to classify images of me and my two friends. 
It is a project for midterm. Please check it as well.

```

https://github.com/12194916/AI_Applications_class/blob/0a55a08a647078b76d36833648485a8f672fd50e/Muhammadyusuf_12194916_MidTerm/week%234-Intro_to_google_collab_file.ipynb
```
-This is 4 the week work. There, we learned getting a csv file from the link of git hub. And displaying 
the cpu information. And instlling GPU if we have any.Generally we learnt how to get data from different 
sources. Like, kaggle, google drive, git hub so on. We tried to get mnist csv from the kaggle. Then Unzipped it.
```

https://github.com/12194916/AI_Applications_class/blob/0a55a08a647078b76d36833648485a8f672fd50e/Muhammadyusuf_12194916_MidTerm/week%235-BasicOperations.ipynb

```
-This is 5 the week work, where we learnt a lot about basic adding substractin multipliying and other staff 
using tensorflow. If you open the link I gave explanation for each line.
```

https://github.com/12194916/AI_Applications_class/blob/0a55a08a647078b76d36833648485a8f672fd50e/Muhammadyusuf_12194916_MidTerm/week%236-session_1.ipynb
```
-This is 6 the week first session. In that session, we tried to draw the plots of noisy dataset as a line graph. 
We leaned how to add weight to the model and there was layer adding, then all  learning rate, epochs batch size 
are given as we trained our model at the end. I gave an accuracy of 92 for val and train datasets. 
```
https://github.com/12194916/AI_Applications_class/blob/0a55a08a647078b76d36833648485a8f672fd50e/Muhammadyusuf_12194916_MidTerm/week_6_session_2.ipynb
```
-This is 6 the week work 2 nd session. There, we imported fashion mnist dataset from the collab. Then 
added certain layers for hidden and input and output. image size, batch size, epochs are given. Then we 
added weight to all. Then we tried to creat a computational graph. We implemented it using TensorFlow API, 
giving 5 dense layers, 3 for hidden layers. Finally we ploted the noisy and orginal images. 
```
https://github.com/12194916/AI_Applications_class/blob/0a55a08a647078b76d36833648485a8f672fd50e/Muhammadyusuf_12194916_MidTerm/week%237_CIFAR10_withTorch.ipynb
```
-This is a work for week 7 first class. Using the pytorch API, we just imported CIFAR10 dataset, 
that includes many images. We tried to load a data from the dataset, we used matplotlib and data 
loader to draw an image plot.With numpy, the shape was inistalled. We got the data from the torch libery datasets.
```
https://github.com/12194916/AI_Applications_class/blob/0a55a08a647078b76d36833648485a8f672fd50e/Muhammadyusuf_12194916_MidTerm/week%237-1_2,1_3,1_4,1_5.ipynb
```
-This a work for 7 the week second class. We imported the data and imported all the needed liberies for the data. 
We gave epoch size and batch size. We downloaded the data from the torch libary wth the help of dataloader. Here we 
defined what are the test and train data. We got CIFAR10. Then we added Convolutional layers, 3 by 3 size karnel, with 
2 padding. So behind it, there is a multiplication with 3 b3 size matrix. Then activation was ReLu. Then we flattened 
all as a list to male a model.he next process is training. In order to train the model we have to give weights. So 
each weight is added before giving to the train.
```
https://github.com/12194916/AI_Applications_class/blob/0a55a08a647078b76d36833648485a8f672fd50e/Muhammadyusuf_12194916_MidTerm/MidtermProject_Multiclass_FriendsClassification.ipynb
````
-This is a link for my project. Not copied from anywhere, done by myself as the data is images of me and my friends. 
I imported the needed liberies of Keras API. They are some layers, functions to split folders that i can split the 
data for train and testing. To open zip fil, zipfile. I just unzipped the data with the help of ZipFle function. 
And once it is done, it prints out "Done".I gave links for getting the data and exporting the splited data. It takes 
images from input link and split it with 0.2 ratio. Then gives to the output link location.I gave links for the train 
and validation data. I wanted to make the image size as 200x200. So here it is.I used image libery and load_img function 
to load a sample image of one class. 

   Here what I did is called Image Augmantation. Images can be in differect shape and condition, some might be scaled, 
some might be shifted to the left or right. So the model might get confused once it gets new image. To solve the problem, 
we will shift scale or do some edtings to the images and generate new images to train with the batchh size. So it only 
works with cnn, not ANN or any other staff. I scaled the images 255 times by 1/255 to make it easy for the model to work with images. 
And rotation so on ...  "flow_from_directory" this function takes images from the given links and give them certain classes according 
to the files they are in. So target size or classes can be given here. Like categorical if there are more than 2 classes. And the batch 
size is for generating new images. Target_size is just the size of images derived from the link. Then  did this to see image classes and
their names, just wrote that function.
     
  Wow, here it is I made my model here using Sequential API. I used Convolutional layer with the karnel size3 
by 3, activation function is "Relu", The shape model accepts is input shape, then there is MaxPooling with 2 by 2. 
I know what they do deeply, but they are all deep math. Activation for example is a graph of out predictions. convolution 
is multiplication of mask to the image. Pooling is decreasing the image for the model. So I repeated it 3 times. Then I 
flattened the images into a single list. So then I added 3 Dense layers with 16,32,64 neurons. At the end the output layer
got 3 neurons as i have 3 classes only. Here the output comes from the last dense layer with the activation "Softmax" as we 
have more classes.
    Compiling the model, we will enable the loss type according to the number of classes. In order to decrease the loss, 
we need optimizers, so I chose"adam" as it is the best for any model. Metrics is the evaluating type, so just accuracy.we 
are fitting the model with 100 epochs that our model learns the images 100 times. The reult was good, 83 for train and 80 
for testing. So no overfitting or underfitting. It is fitting well. Then It is saving the trained model, so i am saving it 
as h5 file with its weights. 
  A code for the visulazation of train and validation losss.A code for the visulazation of train and validation accuracy. 
As you can see by epoch, it increases as the loss decreases.
    It is like testing process, I got the image and loaded first. Then I corrected its size and dimentions, then gave to the prediction. 
So as the model gives an array, i just took its argmax to know which image has more probability.. Then, it is simple if else.
       #You can download the dataset and train the model to see how t works. 
```
