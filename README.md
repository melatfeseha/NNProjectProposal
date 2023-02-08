## Geoguessr AI

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

1. Should we even be doing this?

Yes, our model could have positive applications such as being a useful tool for identifying location. Although our model will hopefully be somewhat accurate on a broad scale, it is very unlikely that our tool will be accurate on a more granular level, which lowers the possibility of malicious use. 

2. What might be the accuracy of a simple non-ML alternative?

A non-ML alternative would be difficult to code. It would be nearly impossible to classify an image’s location without any use of ML.

3. What processes will we use to handle appeals/mistakes?

We will attempt to make sure that our model isn’t used in any circumstances where it is critical that the location is correct. We will emphasize that our model isn’t perfect and that the guesses will inevitably have mistakes.

4. How diverse is our team?

In certain respects our team is diverse and in others it is not. In terms of geography, we all are from different areas (mostly within the US, however) — Washington, Oregon, New Zealand, Rhode Island, Florida, Germany. This should give us some help in making sure our model reflects geographic diversity. However, we all live in Claremont now and attend the same school, so in that sense we are not very diverse. 

**Data Questions**

1. Is our data valid for its intended use?

With careful consideration of the locations from which we are pulling our images, yes. We will need to be deliberate about ensuring that we have equal representation of locations.

2. What bias could be in our data? (All data contains bias.)

Our data has the potential to be biased towards unequal representation of different geographies. This includes not only unequal representation of certain nations or continents over others, but also certain types of environments (eg: bias towards cities or rural areas)

3. How could we minimize bias in our data and model?

Minimizing bias in our data will involve a thoughtful approach to sourcing images. This could involve allotting a certain amount of images to each continent, country or latitude-longitude gridspace. 

4. How should we “audit” our code and data?

We will look into how to do this more thoroughly at a later time.

**Impact Questions**

1. Do we expect different error rates for different sub-groups in the data?

Yes. It will typically be a lot easier to correctly identify images taken from major cities than those taken in rural areas since there is a lot more text to use for identification. Additionally, we would expect our error rate to be higher in accurately identifying locations at the same latitudes, where climates are similar.

2. What are likely misinterpretations of the results and what can be done to prevent those misinterpretations?

People may misinterpret the accuracy of this model. It will be necessary to be clear that this model will not be 100% accurate, and should not be viewed as such. 

3. How might we impinge individuals' privacy and/or anonymity?

We might not want to share any of the training data in case it contains personal/sensitive information. Similarly, we might not want to share/store any of the images that people feed into our model.
