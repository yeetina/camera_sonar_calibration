# camera_sonar_calibration
## Getting Started:
### Charuco Target
This repository uses a charuco pattern as a calibration target. In charuco_utils.py, there is a function called generate_charuco_board_image that will allow you to save an image of the pattern. This code is designed for an 8x11 grid. If you want to use a different shape, see notes. You will need to print out the design on a waterproof material and place screws or bolts in the centers of specific squares (show diagram). The target can be scaled up or down as needed, but it is recommended to use a size of A4 or larger. In the field for square size, be sure to input the side length of the squares in meters.

### Camera Calibration
You will need to have the intrinsic calibration matrices for your camera before calibrating it with the sonar. This repository provides a couple calibration scripts in camera_tools. There is one for a charuco target and one for a chessboard target. You can use any method to obtain your calibration matrices. You will need to input the matrices by saving them into a json file or pasting them as numpy arrays in initialize_camera in calibration_gui.py. 
Detect_charuco_pos can help you check if you camera calibration is correct or at least reasonable.

### Data collection
A tool for saving images from a camera is provided. When you run save_images.py, press "s" to save a frame and it will automatically be named with a correctly-formatted timestamp.

### Data Input
To use the gui, you will need to have folders of sonar and camera images set up in the right way. The file structure will look like this:
```
your_folder/
    |-camera
    |-sonar
```
The code is designed to accept camera images with file names in timestamp format YYYYMMDD_HHMMSS. The sonar images used a similar naming convention of Oculus_YYYYMMDD_HHMMSS, where the timestamp matches one of the images. It is not necessary to name the sonar images in a specific way; the only requirement is that when the camera file names and sonar file names are passed into python’s sorted function, they both must return in the same order.

### Sonar Data
This program accepts sonar input in the format of a rectangular image showing the sonar display. To isolate the sonar data from your input images, first run the sonar cropping tool. It will allow you to indicate the shape and size of the sonar arc so that this program can extract the relevant data. Before running, make sure to imput the filaname for one of your sonar images at the bottom of this file. It is assumed that the sonar arc is in the same position in all of your sonar images. The resulting values from this tool will be automatically saved to sonar_cropping_params.json. You can also change the file name by... but if you do, make sure you also change it in inage_sonar_utils.

### Sonar Parameters
In image_sonar_utils.py, you will find the SonarInfo class. This class was written for an Oculus M3000d sonar and includes values specific to that device. Make sure to input the aperture, range resolution, and angular resolution of the sonar device used in data collection. It is also designed to accept the range that was used for data collection in meters and a boolean wide to indicate whether the sonar was used in wide angle mode or not. The Oculus sonar has different specifications if used in wide angle mode. 

## Using the GUI
### Generating inputparams.json 
The first time you run the gui on a new folder of data, you will be prompted to input some parameters. Input range and width as prompted. The external translation and rotation vectors will be used as initial values for the calibration and will be used to check the accuracy of _. See matlab tools section* for a way to calculate these, or put all zeros if _. The input should be a list of three floats separated by a comma and space (ex: 0.1, 1.5, 0.0). 

## Gui Layout
When you run the gui the display will look something like this if your camera and sonar data are good. (image)
In the center is the display where you can select sonar points. On top, there are some tools to help you view the image better. Hover your mouse over these buttons to find out what they do. 
The left has a charuco target diagram, four camera views each showing different information, and the calculated coordinate transformations.
To calculate the calibration, start by selecting a point on the central window corresponding to one of the bolts. A (labeling?) window should pop up. Using the top left image as a guide, anter the label for that point. (Uppercase or lowercase doesn't matter.

At the bottom of the window there is a row of buttons. 
Breif decriptions of each button:
- Next: This just moves to the next image in the folder
- Skip: move onto the next image and mark the current one to be skipped every time the gui runs
- Mark Good: saves the image and its timestamp as "good". This label is kept for future iterations?
- UN Mark Good: Removes the image and its timestamp from good folder
- Prev Good and Next Good: move to the previous/next image that is marked "good"
- Remove Label: Allows you to remove a specific label or remove all labels. Another way to move a label is simply to click on the new location and retype the label.
- Recalibrate: Every time a new point is added, the calibration calculation for that image is updated, but press this button if you want to update the calibration for all images. The overall calibration value is also updated every time you switch to a different image.
  
## Troubleshooting common problems

## How to change things without breaking the code

## What's happening behind the scenes
Order of events when running calibration_gui
1. setup_layout
2. sonar params json
3. create SonarInfo object
4. create SensorData object
5. calculate polar transform
6. load_state
7. call next button
8. call update_plots
- update pair label
- detect_charuco_board
- save camera pose or skip
- plot_raw_camera_data
- plot_charuco_detections
- plot_camera_targets_from_camera
- sonar polar transform
- calibrate_sonar
