# SVM Food-101 Dataset Classification

This project is my part taken from the main project github [food_classification](https://github.com/floxyao/food_classification). I've done two deep learning algorithm, [SSD Inception v2 for Card 9-A Object Detection](https://github.com/Insignite/TensorFlow-Object-Detection-API) and [AlexNet architecture for DogvsCat Classification](https://github.com/Insignite/Alexnet-DogvsCat-Classification), so I would like to dive deeper into Machine learning field by working on an algorithm even earlier than AlexNet. Support Vector Machines (SVM) for multiclasses classification seems fun so I decided to go with it.

## Introduction
<img src="https://github.com/Insignite/SVM-Food101-Classification/blob/master/img/svm_sample.png" height="70%" width="70%">
**Support Vector Machines (SVM)** is a supervised learning model with associated algorithms that analyzes data by plotting data points on N-dimensionals graph (N is the number of features) and performs classification by drawing an optimal hyperplane. Data points that closer to the hyperplane influence the position and the orientation of the hyperplane. With this information, we can optimize the hyperplane by fine tuning **Cost (C)** and **Gradient (g = gamma substitute variable)**. Large **C** decreases the margin of the hyperplane, which allow much less misclassified points and lead to hyperplane attemp to fit as many point as possible, where as small **C** allows more generalization and smoother hyperplane. For **g**, a higher value leads to a lower Ecludien distance between data points and scale down fit area.


## Dataset
[Food-101](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz) is a large dataset consist of 1000 images for 101 type of food. Each images have a range of dimension from 318x318 to 512x512.

<img src="https://github.com/Insignite/SVM-Food101-Classification/blob/master/img/data_sample.png" height="70%" width="70%">

For linux user, extract the download dataset. For windows user, just use compress file extractor like WinRAR.
```
tar xzvf food-101.tar.gz
```

### Dataset structure
```
food-101
  |_ images
      |_ ***CLASSES FOLDER***
          |_ ***IMAGE BELONG TO THE PARENT CLASSES***
  |_ meta
      |_ classes.txt
      |_train.json
      |_ train.txt
      |_ test.json
      |_ test.txt
      |_ labels.txt
  |_ license_agreement.txt
  |_ README.txt
```
### Dataset classes
```
apple_pie	    eggs_benedict	     onion_rings
baby_back_ribs	    escargots		     oysters
baklava		    falafel		     pad_thai
beef_carpaccio	    filet_mignon	     paella
beef_tartare	    fish_and_chips	     pancakes
beet_salad	    foie_gras		     panna_cotta
beignets	    french_fries	     peking_duck
bibimbap	    french_onion_soup	     pho
bread_pudding	    french_toast	     pizza
breakfast_burrito   fried_calamari	     pork_chop
bruschetta	    fried_rice		     poutine
caesar_salad	    frozen_yogurt	     prime_rib
cannoli		    garlic_bread	     pulled_pork_sandwich
caprese_salad	    gnocchi		     ramen
carrot_cake	    greek_salad		     ravioli
ceviche		    grilled_cheese_sandwich  red_velvet_cake
cheesecake	    grilled_salmon	     risotto
cheese_plate	    guacamole		     samosa
chicken_curry	    gyoza		     sashimi
chicken_quesadilla  hamburger		     scallops
chicken_wings	    hot_and_sour_soup	     seaweed_salad
chocolate_cake	    hot_dog		     shrimp_and_grits
chocolate_mousse    huevos_rancheros	     spaghetti_bolognese
churros		    hummus		     spaghetti_carbonara
clam_chowder	    ice_cream		     spring_rolls
club_sandwich	    lasagna		     steak
crab_cakes	    lobster_bisque	     strawberry_shortcake
creme_brulee	    lobster_roll_sandwich    sushi
croque_madame	    macaroni_and_cheese      tacos
cup_cakes	    macarons		     takoyaki
deviled_eggs	    miso_soup		     tiramisu
donuts		    mussels		     tuna_tartare
dumplings	    nachos		     waffles
edamame		    omelette
```

### Dataset Approach
In this project, I will only do classification for noodle as I have limited resource for training and testing. There are 5 noodle classes total:
```
['pad_thai', 'pho', 'ramen', 'spaghetti_bolognese', 'spaghetti_carbonara']
```
With 5 classes, I have 5000 images total. `train.json` and `test.json` splitted into 3750 and 1250 respectively.
Let's load in the data through `train.json`. But first let's look at how the data labeled.
**(Below is a very small scale of train.json content for ONLY 5 classes I am targeting. Original train.json will have all 101 classes)**
```
{
    "pad_thai": ["pad_thai/2735021", "pad_thai/3059603", "pad_thai/3089593", "pad_thai/3175157", "pad_thai/3183627"],
    "ramen": ["ramen/2487409", "ramen/3003899", "ramen/3288667", "ramen/3570678", "ramen/3658881"],
    "spaghetti_bolognese": ["spaghetti_bolognese/2944432", "spaghetti_bolognese/2969047", "spaghetti_bolognese/3087717", "spaghetti_bolognese/3153075", "spaghetti_bolognese/3659120"],
    "spaghetti_carbonara": ["spaghetti_carbonara/2610045", "spaghetti_carbonara/2626986", "spaghetti_carbonara/3149149", "spaghetti_carbonara/3516580", "spaghetti_carbonara/3833174"],
    "pho": ["pho/2599236", "pho/2647478", "pho/2654197", "pho/2696250", "pho/2715359"]
}
```
SVM parameters required a label list and feature list. So I will load data from `train.json` into a dataframe and create a feature list for both HOG and Transfer learning. 
```
Train Dataframe
         filename  label
0     1004763.jpg      0
1     1009595.jpg      0
2     1011059.jpg      0
3     1011238.jpg      0
4     1013966.jpg      0
...           ...    ...
3745   977656.jpg      4
3746   980577.jpg      4
3747   981334.jpg      4
3748   991708.jpg      4
3749   992617.jpg      4

[3750 rows x 2 columns]

HOG Train Feature Shape
(3750, 1942)
Transfer Learning Train Feature Shape
(3750, 6400)
```
## Training
### Training Approach
I built a SVM classification with two approach: 
#### Histogram of Oriented Gradients (HOG)
<img src="https://github.com/Insignite/SVM-Food101-Classification/blob/master/img/hog.PNG" height="70%" width="70%">
By using HOG, it shows that HOG image able the keep the shape of objects very well which allow for an edge detection. The input images will get reshape to 92x92x3 or 128x128x3 (Higher amount of pixel make my laptop much slower for training yet increase the accuracy). I also applied Principal Component Analysis (PCA). It is a method used to reduce number of features (aka reduction in dimensions) in the data by extracting the important data points while retaining as much information as possible. 

#### Transfer learning
<img src="https://github.com/Insignite/SVM-Food101-Classification/blob/master/img/AlexNet.png" height="70%" width="70%">
Transfer learning technique is a method that use pre-trained model to build a new custom model or perform feature extraction. In this project, I will use an pre-trained **AlexNet** model from my teammate for feature extraction. AlexNet input is always 227x227x3 so I will reshape all image to this dimension. I built a new model with all layers of my teammate AlexNet untill *flatten layer* (Displayed in figure), which give output of 5x5x256 = 6400 training features.

### Training parameters
SVM have tree important parameters we should wary about: Kernel type, C and g (C and g explaination in **Introduction** section). Kernel
type is very much depend if the data points is linear seperable. Let's plot 151 images with their first 2 features out of 6400 features into different kernel of SVM. All three plot will have C = 0.5 and g = 2.

<img src="https://github.com/Insignite/SVM-Food101-Classification/blob/master/img/kernel.png" height="70%" width="70%">
It seems like the data points able to classify decently well with all three kernels, but this is only the first 2 features. What if we plot all 6400 features? There will definitely an kernel out perform others. I'd love to able to graph 6400 features but that will be so complicate to do so. There are still C and g that I can adjust to optimize the hyperplane. Let's take a look of various C and g plot.
(Image source: [In depth: Parameter tuning for SVM](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769) )

<figure>
  <a href= "https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769">
    <img src="https://github.com/Insignite/SVM-Food101-Classification/blob/master/img/gamma_sample.PNG" title="Source: In depth: Parameter tuning for SVM">
  </a>
    <a href= "https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769">
    <img src="https://github.com/Insignite/SVM-Food101-Classification/blob/master/img/c_sample.PNG" title="Source: In depth: Parameter tuning for SVM">
  </a>
  </figure>

With so many way C and g can tune the heperplane, how can we find the optimal combination? Let's do something called Grid searching, essentially is running cross validation for all possible combination of Kernel, C, and g on certain range. According to [A Practical Guide to Support Vector Classification](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) paper, exponential growing of C and g give the best result. I will use the paper recommended range C = 2<sup>-5</sup>, 2<sup>-3</sup>, ... ,2<sup>15</sup> and g = 2<sup>-15</sup>, 2<sup>-13</sup>, ... , 2<sup>3</sup>. With all three parameters, I able to create 396 combinations. Below if a sample of some combination runs.

<img src="https://github.com/Insignite/SVM-Food101-Classification/blob/master/img/grid_search.PNG" height="45%" width="35%">

After 396 cross validations run with different parameters, the parameter with highest accuracy is Kernel = Linear, C = 0.5, and g = 2. Now we are ready to train our model.

### Training Model
I initally use Scikit-Learn to train an SVM model, but it takes extremely long for unknown reason. Till this day I still don't know. Stumble upon an suggestion, I switched over to [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) and able to increase training time significantly.

## Result
### Histogram of Oriented Gradients (HOG)
```
- Traing Validation Accuracy: 81.0%
- Test Accuracy: 96.0%
```
### Transfer Learning
```
- Cross Validation Accuracy: 57%
- Test Accuracy: 68.2%
```
<img src="https://github.com/Insignite/SVM-Food101-Classification/blob/master/img/result.PNG" height="70%" width="70%">

### Conclusion
HOG approach have a much higher accuracy compare to Transfer learning approach. This is within my expectation because Transfer learning on AlexNet model required input image to go through a series of filters, which lead to loss of detail and reduction in features. My prediction is that if Transfer learning approach taking earlier layers, rather than taken up to the last Convolutional layer of AlexNet, the accuracy would be better because layers toward beginning of AlexNet architecture given much more features then later layers.
