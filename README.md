# Geoguessr AI

## Geoguessr Neural Net Introduction and Related Works

##### By Jan Charatan, Aidan Wu, Melat Feseha, Victor Hernandez Brito, Colin Kirkpatrick

## 1. Abstract
Geoguessr is a popular game that involves guessing the location of an image based on Streetview. Users use contextual clues such as signs, buildings, and plants to help inform the location they guess. This is a challenging project because there are many environmental subtleties that need to be taken into consideration in order to accurately identify different geographic locations. We created a convolutional neural network that can accurately estimate the location of a given image sourced from Google Maps. Our solution leverages convolutional neural networks because they can learn the numerous geographic complexities, even ones that humans might have trouble identifying. This tool presents a number of ethical questions, such as potential abuse regarding identifying location from personal social media pictures. We believe the potential risks of this tool can be limited by restricting location estimates to country of origin, rather than specific geographic location. 

## 2. Introduction
While our web app is modeled after the popular browser game GeoGuessr, the ability to generalize the process of geographical estimation from simple image input has many powerful applications. Generally, we rely on known landmarks or stored image metadata to identify the location an image was taken. While these are effective methods, they are not always available to us. Our tool enables us to estimate the location of an image, regardless of whether it was taken in an ambiguous location, even in the complete absence of metadata or visual signals such as street signs or building names. This tool could be used by researchers, hobbyists, and travelers to match media to real-world locations. Further applications of this tool could be used to generate original images of terrain that demonstrate the characteristics of a specific location. 

## 3. Related Works

There exists previous academic work that has attempted to use neural networks to identify the location of an image. Suresh et al.’s paper is very similar to ours—they trained a neural network that takes in pictures of the United States and guesses the most probable state using a balanced dataset that equally samples from all 50 states [7]. Another paper that was slightly more broad in scope than Suresh et al. is Müller-Budack et al [6]. This paper sought to guess geolocation without being limited to a particular country—the neural network trained in this paper is notable because it takes into account information from different spatial resolutions. Another paper that tackles a similar problem to the one we are trying to solve is Kim et al [3]. Part of this paper trains a neural network to classify tourist attractions to aid in the goal of identifying which parts of tourist attractions are most appealing to foreign visitors. Furthermore, other academic work has sought to solve related classification problems as well. For example, Li et al. used a neural network to identify the location of a license plate [4]. Miura et al. sought to classify the geolocation of tweets [5]. In industry, Google has received media coverage for their solution to the problem we are tackling [1]. They have trained a model called PlaNet that can correctly identify the correct continent of an image 48.0% of the time [8]. Finally, we referred to resources on convolutional neural networks such as a video by Deeplizard to learn about the approach we should use for our project [2].

## 4. Ethical Sweep
Our primary ethical concerns can be grouped into two broad categories: dataset bias and possible misuse. When collecting our data, we paid particular attention to attempting to ensure that our dataset was equally representative of each nation included in our app. To make sure this was achieved, we used equal numbers of training images for each nation. Our image data was scraped from google maps, which only had data from 55 countries. Therefore, one ethical concern that our group has identified is that our tool only applies to 55 countries, not every country as we had initially hoped. 
 
 Our project inadvertently highlights the ways in which bias against certain regions of the world can show up. Just the fact that we could not grab any images from certain countries is a testament to how difficult it can be, even with the best intentions, to create tools that are unbiased. This highlights how much further we can go in making sure we have the resources to build equitable tools in the future, so that we don’t exclude sections of the world in the tools that we build and the projects that we pursue.  

## 5. Methods
We began by using a Kaggle dataset which contained about 50,000 images from the GeoWorld challenge that were labeled by country. We then created a Resnet50 convolutional neural network using PyTorch Lightning. Other than having a learning rate of 3e-4 and a batch size of 16, we used default values for all hyperparameters. For our optimizer, we used Adam with the same learning rate of 3e-4. In terms of other infrastructure, we used Hydra to pass in hyperparameters to the model and we logged model progress such as training and validation loss with WandB. Additionally, PyTorch Lightning helped us split and load data and do checkpointing.

After being unimpressed by the imbalance of the Kaggle dataset and the biases it was causing in our results, we decided to scrape our own dataset. Using Selenium, we scraped images from a website that generates images from Google Street View, https://randomstreetview.com/. In total, we scraped 2,000 images each for 55 different countries. That gave us a total of 110,000 images of training data. Each image was tagged with its country of origin. We believe this dataset was sufficient in both breadth and width such that our neural net could be accurate at identifying a large number of different countries. Typically, countries like the United States have far more available data, therefore can be overrepresented in datasets. This can make tools like GEoFFREY biased towards overrepresented nations in their level of accuracy. We paid attention to making sure we had an equal distribution of data for each country so that no one country would be over or underrepresented. This does not guarantee that our neural net will have the same level of accuracy for each nation, but it does take a step towards rectifying the bias that was built into our original dataset. Using this dataset, we tried our Resnet50 architecture and we also tried a Resnet18 architecture. Additionally, for the Resnet18 architecture, we tried a lower learning rate.

## 6. Discussion 
We will present the results of our neural network on our validation data to make sure the model is not overfitting the data and with emphasis placed on accuracy with underrepresented countries so that we can mitigate bias in our model, and acknowledge it accordingly when it is there. Our results will also be interactively accessible within a web app. In this web app, users will be able to upload their own images and see which countries our model predicts. Additionally, they will be able to compete against the model. We will upload some images of locations from around the world and our model will guess where the location is and the user will guess where the location is. Depending on who is closer, points will be given out. We aim to get about 75% accuracy in our model, with the correct answer being in the top 1 of the countries in our neural network outputs, although we will be a little careful in the way we interpret our results because our model could just be picking up the fact that our data is biased, and guessing the countries based on that rather than learning actual geospatial features particular to a location.

Previous work in this area has used metrics such as top 1 and top 5 accuracies to understand how well a model is performing. We will provide a direct comparison with these metrics to gauge the performance of our model. It will be important to note that we used a different dataset which has a large impact on how high of an accuracy you can achieve. We want to understand the impact that data has on our model, so our goal is to be able to compare model accuracy with the current dataset we have, that has a pretty distinct bias towards global north countries, with the supplemented dataset we create with the scraped images we get from google maps, along with how the model does with oversampling from underrepresented locations. We will support our claims with the top 1 and top 5 accuracies of our model. Additionally, we will use any relevant figures from our Wandb logging. We will also provide images from our dataset and the model guesses so that readers of our report can easily understand what is going on.

So far, we’ve been able to achieve 50% top 1 accuracy and 80% top 5 accuracy after just 6 epochs. We plan on modifying the model a bit more to increase our top 1 accuracy, without detriment to our top 5 accuracy, which we’d love to increase as well although it is pretty good already. When we examine our model accuracy for specific countries, we see that there are some for which the model is particularly adept at, such as New Zealand, and some for which the model did worse than average.

Once our model is fully trained, we can check for overfitting by giving it photos of countries that are less represented in the data. If the top 1 accuracy for these countries is lower than the average top 1 accuracy, that means that our neural network is overfitting for pictures of countries that are more abundant in the data. We can treat this in multiple ways, but an effective technique could be to modify the photos of the countries which the NN struggles with to generate more data specific to those nations, then train the NN with this new set of data. 

## 7. Reflection
In general, we were surprised about how accurate the model was with very minimal training: when we first created the network, it guessed the country correctly 36% of the time and had the correct country in the top 5 66% of the time, which I think is better than we expected the network to do initially. We currently have it at 50% top 1 accuracy and 80% top 5 accuracy, which we can improve upon by tweaking the type of model we’re using, along with tuning our hyperparameters. Considering then dataset we have as well, this is a pretty good result.

Some things we would do differently is trying to find different types of images to scrape: we have the Kaggle dataset which we supplemented with google maps images, but those provide only a specific kind of look at any particular place, so incorporating images from different angles, focuses, etc. we think would be beneficial in increasing model accuracy. In addition to improving the quality of our dataset, we would spend more time exploring the benefits of different types of neural networks, comparing their architectures and how they contribute to our accuracy. 

Future work would entail a continuation of hyperparameter tuning to increase model accuracy, adding more and different kinds of images to our dataset, as well as training the model to be able to recognize when it cannot infer the location of an image. In addition to that, we also proposed this project with the idea that we would also create a neural network that could generate an image based on text input, particularly, given the name of a location/country, what kind of image would be generated? So in the future, it would be interesting to compare these networks: for example we could have that second neural network generate an image of a location, and pass that image into the neural network we created to see if the neural network can recognize the location as the one originally inputted. 

## Ethical Sweep

**General Questions**

* Should we even be doing this?

Yes, our model could have positive applications such as being a useful tool for identifying location. Although our model will hopefully be somewhat accurate on a broad scale, it is very unlikely that our tool will be accurate on a more granular level, which lowers the possibility of malicious use. 

* What might be the accuracy of a simple non-ML alternative?

A non-ML alternative would be difficult to code. It would be nearly impossible to classify an image’s location without any use of ML.

* What processes will we use to handle appeals/mistakes?

We will attempt to make sure that our model isn’t used in any circumstances where it is critical that the location is correct. We will emphasize that our model isn’t perfect and that the guesses will inevitably have mistakes.

* How diverse is our team?

In certain respects our team is diverse and in others it is not. In terms of geography, we all are from different areas (mostly within the US, however) — Washington, Oregon, New Zealand, Rhode Island, Florida, Germany. This should give us some help in making sure our model reflects geographic diversity. However, we all live in Claremont now and attend the same school, so in that sense we are not very diverse. 

**Data Questions**

* Is our data valid for its intended use?

With careful consideration of the locations from which we are pulling our images, yes. We will need to be deliberate about ensuring that we have equal representation of locations.

* What bias could be in our data? (All data contains bias.)

Our data has the potential to be biased towards unequal representation of different geographies. This includes not only unequal representation of certain nations or continents over others, but also certain types of environments (eg: bias towards cities or rural areas)

* How could we minimize bias in our data and model?

We can minimize the harm from our dataset’s biases by oversampling our underrepresented data. We can do this oversampling in the Kaggle dataset that we’re using initially, and if there’s enough time we can also work to supplement that dataset by adding more scraped images from underrepresented geographic locations. We can also reduce harm by being transparent about the biases that we are aware exist.  

* How should we “audit” our code and data?

Using our data reserved for testing, we can compare how well our code works in different regions. This will give us a sense for the bias present. 

**Impact Questions**

* Do we expect different error rates for different sub-groups in the data?

Yes. It will typically be a lot easier to correctly identify images taken from major cities than those taken in rural areas since there is a lot more text to use for identification. Additionally, we would expect our error rate to be higher in accurately identifying locations at the same latitudes, where climates are similar.

* What are likely misinterpretations of the results and what can be done to prevent those misinterpretations?

People may misinterpret the accuracy of this model. It will be necessary to be clear that this model will not be 100% accurate, and should not be viewed as such. 

* How might we impinge individuals' privacy and/or anonymity?

We might not want to share any of the training data in case it contains personal/sensitive information. Similarly, we might not want to share/store any of the images that people feed into our model.

## References:

[1] Brokaw, A. (2016, February 25). Google's latest Ai doesn't need geotags to figure out a photo's location. The Verge.

[2] Deeplizard, “Convolutional Neural Networks (CNNs) explained.” YouTube, 9 December, 2017.

[3] Kim, Jiyeon, and Youngok Kang. "Automatic classification of photos by tourist attractions using deep learning model and image feature vector clustering." ISPRS International Journal of Geo-Information 11.4 (2022): 245.

[4] Li, Gang, Ruili Zeng, and Ling Lin. "Research on vehicle license plate location based on neural networks." First International Conference on Innovative Computing, Information and Control-Volume I (ICICIC'06). Vol. 3. IEEE, 2006.

[5] Miura, Yasuhide, et al. "Unifying text, metadata, and user network representations with a neural network for geolocation prediction." Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2017.

[6] Muller-Budack, Eric, Kader Pustu-Iren, and Ralph Ewerth. "Geolocation estimation of photos using a hierarchical model and scene classification." Proceedings of the European conference on computer vision (ECCV). 2018.

[7] Suresh, Sudharshan, Nathaniel Chodosh, and Montiel Abello. "DeepGeo: Photo localization with deep neural network." arXiv preprint arXiv:1810.03077 (2018).

[8] Weyand, Tobias, Ilya Kostrikov, and James Philbin. "Planet-photo geolocation with convolutional neural networks." Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14. Springer International Publishing, 2016.
