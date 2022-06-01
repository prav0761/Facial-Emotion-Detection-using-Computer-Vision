# Affectnet
Machines have been able to recognize and differentiate between faces for a few years now, but the human face has a purpose outside of identity. Mouths and eyebrows are vital to human communication and allow us to convey tone and emotion without the use of words or gestures. If machines gained the capability to interpret facial expressions into human emotion, it would open a whole new world of sentiment analysis. While the capability is obviously great, this task is hard for many humans to perform. Humans often misinterpret non-verbal signals either due to the similarity of two emotions or due to incorrect assumptions of tone. For our example we are seeking to discriminate between eight key emotions: Happy, Sad, Contempt, Disgust, Fear, Anger, Surprise, and Neutral. As discussed earlier we can see how some of these emotions represent similar feelings, such as Contempt and Disgust, and how other emotions may evoke similar reactions, such as Fear and Surprise. All these factors make this problem difficult to solve, however, our experimentation leads us to believe that it is possible to attain high accuracy in this classification problem.

![image](https://user-images.githubusercontent.com/93844635/171364349-ec53b292-fdc5-403c-98fd-42b0f7ebdf78.png)


# Blog Link

https://ee460j-affectnet.blogspot.com/( Refer this if you want to see the detailed information on project , model and the datset)

# Project Team
![image](https://user-images.githubusercontent.com/93844635/171367204-d3be20ce-f72e-4c4f-afe0-f9728a99fb77.png)
1. PRAVEEN RADHAKRISHNAN
2. ANTHONY PHAM
3. AVISH THAKRAL
4. NATHAN PAUL
5. NIKHIL GANESH KRISH
6. JACOB DACH

# Background

Our dataset of choice for this problem is AffectNet. This dataset is attractive for image recognition due to its large scale; there are around 1 million images in the dataset, and there is 290k images that have been hand labeled with their depicted emotion. In addition, the AffectNet Dataset contains valence and arousal values which describe emotions in a continuous manner. These values are not commonly found in other facial emotion recognition datasets and allows us to create models that predict an image’s location in this continuous space rather than classify the image within the discrete space described by our eight designated emotions: Neutral, Happy, Sad, Surprise, Anger, Disgust, Fear, and Contempt.

# Datasource

Details about the data source can be found in this paper: https://arxiv.org/pdf/1708.03985.pdf . 

# Our Dataset

The dataset we used has around 290k training images and 5k validation images which was equally distributed.

Baseline Accuracy -62%

Our Accuracy -55%( Weighted Loss Approach)( VGG16) with Adam Optimizer and Weighted loss

# Our Model

We had used transfer learning approach in this dataset and used VGG16 model. This Dataset is highly skewed towards a majority classes of Happy,Neutral etc and thus we focused on developing a model which solves the problem of imbalanced distribution. Various approaches like Transformations, Upsampling, downsampling and weightedloss has been tried and approaches involving transformations and weighted loss had a good progress. Adam Optimized performed well in this dataset. Further details about this project can be found in the blog

# Conclusion

We have  thoughts on how to further improve the model given more time and compute resources.  First, we believe that an ability to further tune the hyperparameters listed (class weights for weighted loss, momentum of optimization, weight decay, and maximum rebalance number) would allow for better performance. This hyperparameter search would also have to be compared on both of our chosen Optimizers because while we feel that we have shown that Adam Optimizer provides superior results, the introduction of the various hyperparameters could change our results as we saw with class weights having a huge effect of SGD Optimization. Next we would like to implement a loss function which dynamically train the weights in weighted loss function.  We also would like to test Node Dropout on our model because we saw quick overfitting of training data in conjunction with degradation of validation accuracy. Finally, we would like to look at various other options for our underlying Convolutional Neural Network as this could greatly affect feature abstraction. We believe that by testing these various decision parameters our model could experience even greater gains and with enough resources we could even perform better than the greatest accuracy we have seen of 63%. 


# Structure of Repo

prepare_dataset.py- Preparing the dataset after downloading the data

train_py,utils.py - Train Function and other utility function

vgg_pretrained.py - Vgg16 model

Model.py- Model used for this dataset

# Branches

image_transform - Branch for transformation experiment, can find results and experiments used for transformation

weighted_loss -  Branch for weighted_loss experiment, can find results and experiments used for weighted_loss

val_aro_predictor -  Novel approach and idea for using valence and arousal data


