# camera_sonar_calibration

## Getting Started:

### Charuco Target
This repository uses a charuco pattern as a calibration target. The default target configuration is an 8x11 square grid using the aruco dictionary 'DICT_4X4_250'. [^1] In charuco_utils.py, there is a function called generate_charuco_board_image that will allow you to save an image of this pattern. You will need to print out the design on a waterproof material and place screws or bolts in the centers of the labeled squares.
<img width="360" height="258" alt="image" src="https://github.com/user-attachments/assets/763bee1c-ee50-4a41-b36e-ed5d11f92d2c" />  
The target can be scaled up or down as needed, but it is recommended to use a size of A4 or larger. At the top of charuco_utils.py, be sure to update the length variables. SQUARE_LENGTH is the side length in meters of the whole chessboard square and MARKER_LENGTH is the size of the aruco marker (typically 75% of square length).

### Camera Calibration
You will need to have the intrinsic calibration matrices for your camera before perfoming external calibration with the sonar. You can use any method to obtain your calibration matrices. This repository provides a couple calibration scripts in camera_tools - one for a charuco target and one for a chessboard target. In order for the gui to access the calibration matricies, you can save them as a json file and pass the filename as a parameter in initialize_camera (line 134) or paste them as numpy arrays in initialize_camera (lines 193-197). 
Detect_charuco_pos can confirm if the charuco detection is working. It returns the translation and rotation vectors of the charuco target origin, which can be used to check if your camera calibration is correct.

### Data collection
A tool for saving images from a camera is provided. When you run save_images.py, press "s" to save a frame and it will automatically be named with a correctly-formatted timestamp.   
It is difficult to collect data from the Oculus sonar software and the camera at the same time. Collecting camera data on one computer and sonar data on another was found to be a successful way to save both types of images nearly simultaneously.

### Data Input
To use the gui, you must input the filepath to your data folder as rootdir at the end of calibration_gui. This root folder must contain folders named "sonar" and "camera" containing your images.  
The code is designed to accept camera images with file names in timestamp format YYYYMMDD_HHMMSS. The sonar images use a similar naming convention of Oculus_YYYYMMDD_HHMMSS. There should be equal numbers of sonar and camera images and  the image pairs should have identical timestamps. It is not necessary to name the sonar images in a specific way; the only requirement is that when the camera file names and sonar file names are passed into python’s sorted function, they both must return in the same order.

### Sonar Data
This program accepts sonar input in the form of a rectangular image depicting the sonar arc. To isolate the sonar data from your input images, first run the sonar cropping tool. It will allow you to indicate the shape and size of the sonar arc so that this program can extract the relevant data. Before running, make sure to input the same rootdir filepath that is used in calibration_gui.py. It is assumed that the sonar arc is in the same position for all sonar images in this file. The resulting values from this tool will be automatically saved as sonar_cropping_params.json in the rootdir folder. 

### Sonar Parameters
In image_sonar_utils.py, you will find the SonarInfo class. This class was written for an Oculus M3000d sonar and includes values specific to that device. Make sure the aperture, range resolution, and angular resolution match the specifications of the sonar device used in data collection. Every instance of this class accepts range and width parameters. You will need to know the range that was used for data collection in meters and whether the sonar was used in wide angle mode or not (which is a boolean). The Oculus sonar has different specifications if used in wide angle mode, and the SonarInfo class is designed to reflect that. 

## Using the GUI
### Generating inputparams.json 
The first time you run the gui on a new folder of data, you will be prompted to input some parameters. Input range and width as prompted. The external translation and rotation vectors will be used as initial values for the calibration and can be used to check the accuracy of the calculated calibration vectors. These vectors indicate the translation and rotation of the camera in the sonar's frame of reference. Find these values from the physical configuration of the sonar-camera setup [^2] or put all zeros if they are unknown. The input should be a list of three floats separated by a comma and space (ex: 0.1, 1.5, 0.0). 

## Gui Layout
When you run the gui the display will look something like this if your camera and sonar data are good and everything is set up correctly.  
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/4accc6de-a68b-453c-ba76-56653bbca8a9" />  
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

## How it Works
### Initialization
When calibration_gui.py is run, the following actions will happen:
1. setup_layout() is called to create the layout of the GUI and initialize the various plots
2. setup_data() is called, which performs several functions.
- Accessing the sonar parameters json file or prompting the user to create it.
- Creating a SonarInfo object
- Calculating the sonar image transformation, which converts the sonar arc in cartesian coordinates into a rectangular representation of the polar coordinates
- Creating a SensorData object to organize and step through the image pairs
- Calling LoadState() to retrieve data from previous uses of the gui
3. handle_next_button is called, which then calls update_plots() to populate all of the plots on the gui
### Updating Plots
Most of the functionality of this software happens within update_plots(). 
1. Image label is updated with timestamp and "good" label
2. detect_charuco_board() from charuco_utils is called to identify corners and find the translation and rotation vectors of the target's position relative to the camera. The image pair is automatically skipped if no board is detected.
3. plot_raw_camera_data displays the camera image in grayscale in the figure titled "Raw Camera Image"
4. plot_charuco_detections uses detected charuco corners to annotate the raw image in the figure titled "Detected Aruco / Charuco"
5. plot_camera_targets_from_camera uses the target's position vectors to plot the calculated locations of the bolts to be detected by sonar. It also draws the axes of the target origin to check if the detected rotation vector is correct. This is displayed in the figure titled "Charuco-Derived Locations"
6. The sonar image is mapped from the original format to a polar matrix
7. calibrate_sonar is called. The goal of extrinsic calibration is to find the correct translation and rotation vectors to project points from the camera's frame into the sonar's frame. There are several steps to this process:
- For all of the user-labeled points, the locations on the sonar image are stored as sonar_points and the positions on the target are stored as target_points (get_sonar_target_correspondences)
- Using the known transformation from the camera to the target, the target_points are converted to 3D coordinates in the camera's frame, which are then stored as camera_points
- The calibrate_sonar function from image_sonar_utils is called next. It uses an error function which takes in a translation and rotation vector and uses them to project camera_points into the sonar's frame. The result of the error function is the sum of difference between the projected locations and the actual user-labeled locations. Calibrate_sonar uses the Nelder-Mead minimization algorithm to find vectors that minimize this error function. To help avoid falling into a local minimum, it first minimizes the translation vector while keeping rotation constant. Then, it uses the resulting translation vector to perform a full optimization.
- calibrate_sonar also uses the camera to target and sonar to camera transformations to calculate the sonar to target vectors.

### User interaction


[^1]: If you want to use a different configuration, changes will need to be made to the global variables of charuco_utils.py and the function init_charuco_sonar in image_sonar_utils.py
[^2]: See relative_pose_calculator.m for a way to calculate the external position vectors
