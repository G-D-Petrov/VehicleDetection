**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/CarIMG.jpg
[image2]: ./examples/CarHOG.jpg
[image3]: ./examples/NonCarIMG.jpg
[image4]: ./examples/NonCarHOG.jpg
[image5]: ./examples/new_ROI.jpg
[image6]: ./examples/tracking.jpg
[image7]: ./examples/heatmaped.jpg
[image8]: ./examples/heatmap.jpgg
[video1]: ./project_video_output.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the fifth code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image3]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:

![alt text][image2]
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the one in the notebook(the eight node) yielded the best results.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the default setings and hinge loss.
Here are the features that I used:

|:-------------:|:-------------:| 
| color space   | LUV           | 
| orientations  | 8             |
| pix per cell  | 8             |
|cells per block| 2             |
|:-------------:|:-------------:|
 
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for the sliding images is implemented in the ninth code block.
The algorithm implementation is based on the implementation in the Udacity lectures.
The size of each window is 64 by 64, as that is the size of the images used for training the clasifier. 
The overlap is .75 as that produced the most desirable behaviour.
I also desided to limit the ROI to where new cars might appear.
Here is the result:

![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using LUV 1-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here is an image of the heatmap and how that is translated to a bounding box:

![alt text][image7]
![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline fails when there are 2 vehicles on top ot very close to each other. 
This is because they are creating a big heat spot and that is displayed as a big rectangle instead on 2 smaller ones.

The pipeline can also falter in very light enviroments and other extreme conditions. 
This can be mitigated with more augmentation of the images and bigger dataset.

The pipeline is also still not fast enough.
A viable solution would be to use a deep learning approach to the object detection part of the problem.

