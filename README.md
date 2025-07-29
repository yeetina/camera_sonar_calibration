# camera_sonar_calibration
## Getting started:
### Data input
To use the gui, you will need to have folders of sonar and camera images set up in a specific way. The file structure will look like this:
your_folder/
 -camera
-sonar
The code is designed to accept camera images with file names in timestamp format YYYYMMDD_HHMMSS. The sonar images used a similar naming convention of Oculus_YYYYMMDD_HHMMSS, where the timestamp matches one of the images. It is not necessary to name the sonar images in a specific way; the only requirement is that when the camera file names and sonar file names are passed into python’s sorted function, they both must return in the same order.

This program accepts sonar input in the format of a rectangular image showing the sonar display. To isolate the sonar data from your input images, first run the sonar cropping tool. It will allow you to indicate the shape and size of the sonar arc so that this program can extract the relevant data. Before running, make sure to imput the filaname for one of your sonar images at the bottom of this file. It is assumed that the sonar arc is in the same position in all of your sonar images. The resulting values from this tool will be automatically saved to sonar_cropping_params.json. You can also change the file name by... but if you do, make sure you also change it in inage_sonar_utils.

Things to include:
- Charuco board generation and parameters (make sure to talk about square size param)
- 