# camera_sonar_calibration
## Getting Started:
### Charuco Target
This repository uses a charuco pattern as a calibration target. In charuco_utils.py, there is a function called generate_charuco_board_image that will allow you to save an image of the pattern. This code is designed for an 8x11 grid. If you want to use a different shape, see notes. You will need to print out the design on a waterproof material and place screws or bolts in the centers of specific squares (show diagram). The target can be scaled up or down as needed, but it is recommended to use a size of A4 or larger. In the field for square size, be sure to input the side length of the squares in meters.

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



Things to include:
- Charuco board generation and parameters (make sure to talk about square size param)
- Camera calibration and input of mtx and dst
- 

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
