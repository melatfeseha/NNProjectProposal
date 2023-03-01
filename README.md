# Geoguessr AI

## Geoguessr Neural Net Introduction and Related Works

##### By Jan Charatan, Aidan Wu, Melat Feseha, Victor Hernandez Brito, Colin Kirkpatrick

Abstract
Geoguessr is a popular game that involves guessing the location of an image based on streetview. Users use contextual clues such as signs, buildings and plants to help inform the location they guess. We would like to create a neural network that can accurately identify the location of a given image. This is a challenging project because there are a lot of environmental subtleties that need to be taken into consideration in order to accurately identify different geographic locations. Our solution leverages convolutional neural networks because they can learn the numerous geographic complexities, ones that humans might have trouble identifying, through its training. 


## 1. Introduction

We plan to train a convolutional neural network on our dataset because these are typically used to operate on visual data. Our neural network needs to be able to identify and classify various environmental factors. In order to do this and train our model accurately, we will need to place significant emphasis on the selection of a large, diverse, well tagged dataset that evenly represents our intended regions. 

We intend to use a dataset titled World-Wide Scale Geotagged Image Dataset for Automatic Image Annotation and Reverse Geotagging sourced from Qualinet databases. This dataset includes 14 million geotagged photos crawled from Flickr. This data set exhibits a bias towards countries in Europe and North America, and even though it is quite large, contains virtually no images from certain geographical regions. We must do some work to overcome this, so finding and helping to create more balanced datasets tagged with location data is one of our main technical challenges. Another challenge we will run into is figuring out how to tune our hyperparameters so that our model is as accurate as possible, and training our model with enough iterations on each set of hyperparameters chosen so that we can properly analyze their effects on the accuracy of our model. 


If we have enough time, we also wanted to train a neural network to do the inverse/complementary task of generating an image of a location based on its name. This is a challenging but interesting problem because the way we judge these images is going to have to be pretty subjective, but they might lend some insight into the kinds of things that the neural network is taking into consideration when trying to identify/generate geographic data.

We expect this project to be able to identify countries accurately, and hopefully more specific geographic locations as well. We hope that our model can output the top 3 most likely places that an image was taken from, particularly their city and country, with 90% accuracy of the true location being identified within the top 3.  We also expect to create a neural network that can generate images based on textual input. 


## 2. Related Works
There exists previous academic work that has attempted to use neural networks to identify the location of an image. Suresh et al.’s paper is very similar to ours—they trained a neural network that takes in pictures of the United States and guesses the most probable state using a balanced dataset that equally samples from all 50 states [7]. Another paper that was slightly more broad in scope than Suresh et al. is Müller-Budack et al [6]. This paper sought to guess geolocation without being limited to a particular country—the neural network trained in this paper is notable because it takes into account information from different spatial resolutions. Another paper that tackles a similar problem to the one we are trying to solve is Kim et al [3]. Part of this paper trains a neural network to classify tourist attractions to aid in the goal of identifying which parts of tourist attractions are most appealing to foreign visitors. Furthermore, other academic work has sought to solve related classification problems as well. For example, Li et al. used a neural network to identify the location of a license plate [4]. Miura et al. sought to classify the geolocation of tweets [5]. In industry, Google has received media coverage for their solution to the problem we are tackling [1]. They have trained a model called PlaNet that can correctly identify the correct continent of an image 48.0% of the time [8]. Finally, we referred to resources on convolutional neural networks such as a video by Deeplizard to learn about the approach we should use for our project [2].

## References:

[1] Brokaw, A. (2016, February 25). Google's latest Ai doesn't need geotags to figure out a photo's location. The Verge.

[2] Deeplizard, “Convolutional Neural Networks (CNNs) explained.” YouTube, 9 December, 2017.

[3] Kim, Jiyeon, and Youngok Kang. "Automatic classification of photos by tourist attractions using deep learning model and image feature vector clustering." ISPRS International Journal of Geo-Information 11.4 (2022): 245.

[4] Li, Gang, Ruili Zeng, and Ling Lin. "Research on vehicle license plate location based on neural networks." First International Conference on Innovative Computing, Information and Control-Volume I (ICICIC'06). Vol. 3. IEEE, 2006.

[5] Miura, Yasuhide, et al. "Unifying text, metadata, and user network representations with a neural network for geolocation prediction." Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2017.

[6] Muller-Budack, Eric, Kader Pustu-Iren, and Ralph Ewerth. "Geolocation estimation of photos using a hierarchical model and scene classification." Proceedings of the European conference on computer vision (ECCV). 2018.

[7] Suresh, Sudharshan, Nathaniel Chodosh, and Montiel Abello. "DeepGeo: Photo localization with deep neural network." arXiv preprint arXiv:1810.03077 (2018).

[8] Weyand, Tobias, Ilya Kostrikov, and James Philbin. "Planet-photo geolocation with convolutional neural networks." Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14. Springer International Publishing, 2016.



## Related Works

* https://www.theverge.com/2016/2/25/11112594/google-new-deep-learning-image-location-planet

This article talks about a google AI that recognizes locations based on photographs. It does not use geotags for the location, so it is trained to recognize places just based on the image itself. 

* https://www.youtube.com/watch?v=YRhxdVk_sIs 

This video provides a brief overview into convolutional neural networks, and what the convolutional layers do that make them particularly effective for pattern recognition in images. A convolutional layer consists of matrix operations over subsets of pixels in a given image. A convolutional layer at first, can pick out simple, geometric patterns, but in conjunction with multiple other convolutional layers, can detect more and more complex objects/patterns.

* Suresh, Sudharshan, Nathaniel Chodosh, and Montiel Abello. "DeepGeo: Photo localization with deep neural network." arXiv preprint arXiv:1810.03077 (2018). https://arxiv.org/pdf/1810.03077.pdf

This journal describes a neural net with a very similar goal to our project. The dataset section seemed very relevant to our project. Their neural net attempts to identify which state a 360 degree image was taken. When choosing data, they made sure to gather equal amounts of data from each state. Furthermore, they weighted areas with higher population as higher priority rather than using data from uniformly distributed geography. This will be an important step to consider when we are selecting datasets that will reflect our specific goals. 

* Gang Li, Ruili Zeng and Ling Lin, "Research on Vehicle License Plate Location Based on Neural Networks," First International Conference on Innovative Computing, Information and Control - Volume I (ICICIC'06), Beijing, 2006, pp. 174-177, doi: 10.1109/ICICIC.2006.507.

This journal article describes the process of using a neural network to determine the location of a license plate. This is relevant to our project since contextual clues like license plates are frequently used to determine the location of an image and we could integrate something like this into our process.

* Jiyeon Kim and Youngok Kang. Automatic Classification of Photos by Tourist Attractions Using Deep Learning Model and Image Feature Vector Clustering.
 
This paper explores the use of deep learning models to analyze photos taken by tourists to determine the principal draws to certain cities for tourism. They used a dataset scraped from geotagged photos from TripAdvisor reviews to train a model to identify locations in the photos of tourists. This is a subsection of the problem we are trying to solve.



## Project Scope

1. Find/create a fair(ish) dataset to train our model on that is not weighted towards one area or biome.
2. Create and train a neural network that identifies locations across the globe accurately.
3. Create a neural network that generates an image of a location based on the name given. 
4. Identify relationships that exist between the two models (if any).

## Group Members

* Jan Charatan
* Aidan Wu
* Melat Feseha
* Victor Hernandez Brito
* Colin Kirkpatrick

## Outline

  Geoguessr is a popular game that involves guessing the location of an image based on streetview. Users can use contextual clues to help inform their guesses such as signs and the buildings and plants in the picture. We would like to create a neural net that can accurately identify the location of a given outdoor image (or maybe the top 3 most likely locations that the image was taken). This is a challenging project because it is often unclear where an image was taken. Our solution will need to identify subtle environmental indicators of geographic location. One of the main technical challenges will be getting a large, geographically diverse data set that is tagged with location data. Additionally, the images on Google Maps are typically 360 degree images, so they are bigger and harder to compute into a vector.
  
  Furthermore, we also wanted to train a neural network in parallel to do the inverse/complementary task of generating images based on a given location. This is a challenging but interesting problem because the way we judge these images is going to have to be pretty subjective, but they might lend some insight into the kinds of things that the neural network is taking into consideration when trying to identify/generate geographic data.
  
  Some technical challenges we may run into are:
  1. Finding enough geographically tagged images from across the globe to accurately identify these places
  2. Identifying important parameters to use in tuning our model

  We expect this project to be able to identify countries accurately, and hopefully more specific geographic locations as well. We hope that our model can output the top 3 most likely places that an image was taken from, particularly their city and country, with 90% accuracy of the true location being identified within the top 3.  We also expect to create a neural network that can generate images based on textual input.  

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

Minimizing bias in our data will involve a thoughtful approach to sourcing images. This could involve allotting a certain amount of images to each continent, country or latitude-longitude gridspace. 

* How should we “audit” our code and data?

We will look into how to do this more thoroughly at a later time.

**Impact Questions**

* Do we expect different error rates for different sub-groups in the data?

Yes. It will typically be a lot easier to correctly identify images taken from major cities than those taken in rural areas since there is a lot more text to use for identification. Additionally, we would expect our error rate to be higher in accurately identifying locations at the same latitudes, where climates are similar.

* What are likely misinterpretations of the results and what can be done to prevent those misinterpretations?

People may misinterpret the accuracy of this model. It will be necessary to be clear that this model will not be 100% accurate, and should not be viewed as such. 

* How might we impinge individuals' privacy and/or anonymity?

We might not want to share any of the training data in case it contains personal/sensitive information. Similarly, we might not want to share/store any of the images that people feed into our model.
