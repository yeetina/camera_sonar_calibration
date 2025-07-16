import cv2
import numpy as np
import time


colorim = cv2.imread("C:/Users/corri/OneDrive/Documents/SonarExperimentData/testpairs/sonar/Oculus_20250619_163741.jpg")
sonar_im = cv2.cvtColor(colorim.copy(), cv2.COLOR_RGB2GRAY)
# print(sonar_im.shape)
#cropped = sonar_im[50:942, 620:1232]
cropped = sonar_im[50:942, 117:1733]
#cv2.line(cropped,(104,620),(942,926),(255,255,255),2)
#cv2.line(cropped,(104,1232),(942,926),(255,255,255),2)
# cv2.line(cropped,(0,54),(305,892),(255,255,255),2)
# cv2.line(cropped,(610,54),(305,892),(255,255,255),2)
mask = np.zeros(cropped.shape[:2], dtype="uint8")
cv2.ellipse(mask, (808,892), (892, 892), 0.0, 205, 335, (255), -1)
#cropped = cv2.resize(cropped, None, None, fx = .8, fy = .8)
#masked = cv2.bitwise_and(cropped, cropped, mask=mask)
#cnts, _ = cv2.findContours(cropped, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#print(len(cnts))
#cv2.drawContours(colorim, cnts, -1, (255,0,0), 5)

def crop_sonar_arc(image, wide=False):
    """
    Isolate the sonar display from images saved by oculus software
    """
    sonar_im = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
    if wide:
        angle_start, angle_end = 205, 335
        centerpoint = (808,892)
        cropped = sonar_im[50:942, 117:1733]
    else:
        angle_start, angle_end = 249.5, 289.4
        centerpoint = (305,892)
        cropped = sonar_im[50:942, 620:1232]

    mask = np.zeros(cropped.shape[:2], dtype="uint8")
    cv2.ellipse(mask, centerpoint, (892, 892), 0.0, angle_start, angle_end, (255), -1)
    return cv2.bitwise_and(cropped, cropped, mask=mask)

masked = crop_sonar_arc(colorim, True)

def get_polar_coords(pixelx, pixely, originx, originy):
    rad = np.sqrt((pixely-originy)**2 + (pixelx-originx)**2)
    try:
        theta = np.atan((pixelx-originx)/(pixely-originy)) #in radians I think
    except:
        theta = 0
    return rad, theta

def create_transform_map(range_m, wide = False):
    if range_m >= 1.5:
        r_res = .0025
    else:
        r_res = .002

    if wide:
        aper = 130
        th_res = 0.6
        theta_bins = 216 #(aper/th_res)
        range_bins = int(range_m/r_res) 
        x_center = 808
    else:
        aper = 40
        th_res = 0.4
        theta_bins = int(aper/th_res) #columns
        range_bins = int(range_m/r_res)
        x_center = 305
    
    polar_map = [[(0, 0)]*theta_bins] * range_bins
    x_map = np.zeros((range_bins, theta_bins), dtype=np.float32)
    y_map = np.zeros((range_bins, theta_bins), dtype=np.float32)
    
    #np.zeros((range_bins, theta_bins), dtype=cv2.CV_32FC2)
        #x is thetabin, y is rangebin
    for y in range(range_bins):
        for x in range(theta_bins):
            theta_rad = (x*th_res - aper/2) * np.pi/180
            r_pix = (range_bins-y-1) * r_res * 892/range_m
            dx = r_pix*np.sin(theta_rad)
            dy = r_pix*np.cos(theta_rad)
            #print(x, y, dx, dy)
            xfinal = x_center + dx
            yfinal = 892-dy
            x_map[y][x] = xfinal
            y_map[y][x] = yfinal

    outmap = np.array(polar_map).astype(np.float32)
    return x_map, y_map

xmap, ymap = create_transform_map(2, True)
outim = cv2.remap(masked, xmap, ymap, cv2.INTER_LINEAR)

# options: unordered list of range, theta points or preallocate array based on resolutions
# .4 degree resolution, 40 degree total, maybe 2.5mm range resolution, range of 1.5m
# theta_bins = 100 #columns
# range_bins = 600 #rows
# rectangular = np.zeros((600,100), dtype="uint8")
# height, width = masked.shape
# print(width, height)
# thmax = 0
# thmin = 100
# for y in range(height):
#     for x in range(width):
#         if masked[y][x] != 0:
#             r, th = get_polar_coords(x, y, 305, 892)
#             rscale = 600/892
#             tscale = 100/40
#             th = (th * 180/np.pi) + 20
#             r *= rscale
#             th *= tscale

#             if th > thmax:
#                 thmax = th
#                 print(th)
#             if th < thmin:
#                 thmin = th
#                 print("min ", th)
#             overflow = []
#             if r >= 600 or r < 0 or th >= 100 or th < 0:
#                 overflow.append((r,th),)
#             row = min(np.floor(r),599)
#             col = min(np.floor(th-1),99)
#             row = max(row, 0)
#             col = max(col,0)
            
#             #print(x, y, row, col)
#             rectangular[int(row)][int(col)] = masked[y][x]
#             # the idea is to multiply r and theta by something and take the floor of that
#             # number to sort them into spots in the rectangular image
#             # r is 0 to 892 and we want 0 to 600
#             # theta is -20 to 20 and we want 0 to 100 (also convert atan result to deg)
# print(overflow)
# print(thmax)
# rectangular = cv2.flip(rectangular,-1)

def raw_sonar_to_rectangular(masked, range_m, wide=False):
    #masked = crop_sonar_arc(image, wide)
    #masked = image[:, 57:968]

    #if range is greater than 1.5m, use lower range resolution
    if range_m >= 1.5:
        r_res = .0025
    else:
        r_res = .002

    if wide:
        aper = 130
        th_res = 0.6
        theta_bins = 216 #(aper/th_res)
        range_bins = int(range_m/r_res) #or 2.5
        x_center = 808
        #x_center = 455
    else:
        aper = 40
        th_res = 0.4
        theta_bins = int(aper/th_res) #columns
        range_bins = int(range_m/r_res)
        x_center = 305
    
    #print(range_bins, theta_bins)
    rectangular = np.zeros((range_bins, theta_bins), dtype="uint8")
    height, width = masked.shape
    
    for y in range(height):
        for x in range(width):
            if masked[y][x] != 0: 
                r, th = get_polar_coords(x, y, x_center, 892) #507***
                rscale = range_bins/892 #507***
                tscale = theta_bins/aper
                th = (th * 180/np.pi) + (aper/2)
                r *= rscale
                th *= tscale

                row = min(np.floor(r), range_bins-1)
                col = min(np.floor(th), theta_bins-1)
                row = max(row, 0)
                col = max(col,0)
                
                rectangular[int(row)][int(col)] = masked[y][x]
    return rectangular #cv2.flip(rectangular,-1)***


rectangular = raw_sonar_to_rectangular(colorim, 2.0, wide=True)

# ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250
# BOARD_ROWS = 8
# BOARD_COLS = 11
# SQUARE_LENGTH = .04       
# MARKER_LENGTH = .03           
# MARGIN_PX = 0  

# dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
# charuco_board = cv2.aruco.CharucoBoard((BOARD_COLS, BOARD_ROWS), SQUARE_LENGTH, MARKER_LENGTH, dictionary) 

# square_pixels = 200
# ncols, nrows = charuco_board.getChessboardSize()
# xpixels = ncols * square_pixels
# ypixels = nrows * square_pixels
# board_img = charuco_board.generateImage((xpixels, ypixels))

# cv2.imwrite("charuco11x8.png", board_img)

while True:
    # show the image
    sidebyside = cv2.hconcat([outim, rectangular])
    cv2.imshow("Output", sidebyside)
    cv2.waitKey(3)
                
    #press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("test_images/mapcomparison.png", sidebyside)
        break

cv2.destroyAllWindows()