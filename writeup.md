## Writeup
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_cali.png "Undistorted"
[image2]: ./output_images/test_undistort.png "Road Transformed"
[image3]: ./output_images/binary.png "Binary Example"
[image4]: ./output_images/lane_warping.png "Warp Example"
[image5]: ./output_images/line_fit.jpg "Fit Visual"
[image6]: ./output_images/back_projected.jpg "Output"
[video1]: ./output_images/final_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the lines 30 through 36 of the file called `distortion_correction.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
On the left of this picture you see the distorted image on the right you can see the undistorted one using the camera parameters.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in the functions of `image_filter.py`).
In that file, ech function is used to filter the image with a different technique:
* `_abs_sobel_thresh`: Calculates the gradient using the sobel gradient for the specified direction and filters the outcome
     with a threshold.
* `_mag_thresh`: Filters img by applying sobel operator in both directions and the filtering the magnitude of the gradient
* `_dir_threshold`: Filters image for gradients with a certain direction.
* `_color_thresh`: Filters the image by color in the S channel of HLS color scale.
* `get_binary_image`: Applies all filters to the image to create a binary image with only the unfiltered pixels remaining.
    The filters are tuned for lane detection
    
Here's an example of my output for this step.

![alt text][image3]

The code for my perspective transform uses the cv2.getPerspectiveTransform and cv2.warpPerspective function to warp the image.  
You can find the code for warping in line 43 through 45 in my `main.py`. The `getPerspectiveTransform` function takes
as input an the `src` and `dst` point to warp the image.
The `warpPerspective` function takes as inputs an image (`binary_image`),
as well as the transformation matrix and the size of the image.  I chose the hardcode the source and destination points in the following manner:

```python
src_pt = np.float32([[img_size[1]*7/16, img_size[0]*2/3],
                     [img_size[1]*79/128, img_size[0]*2/3],
                     [img_size[1]*15/16, img_size[0]],
                     [img_size[1]*247/1280, img_size[0]]])

dst_pt = np.float32([[img_size[1]*7/32, img_size[0]*5/9],
                     [img_size[1]*15/16, img_size[0]*5/9],
                     [img_size[1]*105/128, img_size[0]],
                     [img_size[1]*7/32, img_size[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 560, 480      | 280, 400        | 
| 791, 480      | 1200, 400      |
| 1200, 720     | 1050, 720      |
| 247, 720      | 280, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a 
test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

Then I use a sanity check to see if the lanes were found in the last time step. The ckeck is done in line
48 of my `main.py`. The check it self is located in the `sanity_check()` function in the `line.py`. If the previous 
line is reasonable, then the next line is search according to the last lines position using a margin around the
last line. This is done in line 50 of the `main.py` file. The code for the search you cen find in the 
`existing_line_search` function of the `existing_line_seach.py`. If the previous line was not found (for example
at the very beginning) or the previous line doesn't make sense, the lines are searched by a sliding window approach.
This is done in line 56 of the `main.py`. The code for searching in a sliding window manner can be found in the 
`slising_window` function of the `sliding_window.py`.
The found lines are fittet with a polynomial (line 48, 49 in the `existing_line_search.py` and line 97, 98 in the 
`sliding_window.py`). The functions then
return all necessary informations to plot the line and use them for updates of the found line (line 59, 60 of the 
`main.py`)
So the fittet lines look like this:

![alt text][image5]

#### Here I describe how and where I calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the line curvature in the file `calc_curvature` in lines 5 to 14. Note that I used the 
suggested road parameters from the project preparation to transfer the pixel values to meters.
This procedure I did for the left and right lanes and then I calculated the mean of both curvatures.
This way I get the approximate curvature the car is actually driving.
I did this during the update of the lines in line 61 in my code in `line.py`.
the other function `calc_dist` in `calc_curvature.py` calculated the distance of the car from the middle of the road,
given that the camera is exactly in the middle of the car and directly pointing ahead in a 90° angle from the lanes.
Tha calculation is started in line 66 of the `main.py`. 


#### Here I provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 18 through 32 in my code in `img_projector.py` in the function `project_back()`.  Here is an example of my result on a test image:

![alt text][image6]

---
### Pipeline (video)

Here's a [link to my video result](./output_images/final_output.mp4)
---

### Discussion

#### Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

My pipeline is robust on sections of the street where the lane lines are clearly visible and continuous. 
The crutual part is the edge detection and the filtering because a very good filtered image with clear and visible
lines is very easy to fit with a polynomial. However if there is noise in the image, the detection is much more
difficult. And this is where my approach can be improved. On parts of the video, where the lanes are covered by
shadowes or are indistinguishable from the rest of the road, the estimation can fail. I introduced smoothing over
the last 10 frames to overcome short passages, where this might happen. However on longer sections this approach
still might fail. To overcome this, one could use the knowledge about the road and it's width to further narrow
down the search area for the other lane if one lane was found. One could consider introducing a confidence score
for each lane, that measures, how certain the lane was tracked (for example using the standard deviation of the last
lane parameters or similar). This score can be used to predict a lane that is uncertain. That way, is one lane was
found very uncertain, the other lane might be used to better predict the untertain one since they are always considered
to be parallel.
