sonar_im = imread("s7c314.jpg");
[rows, cols, channels] = size(sonar_im);
%divider = round(cols/2); % not actually split in the middle, but ignore for now
sonar = sonar_im(1:end,57:968,:);
camera = sonar_im(1:end,divider+1:end,:);
centerpoint = [512 507]

figure(1);
imshow(sonar)
%968, 285 - 57,283
%I think it's also 130 deg

%%
oculus = imread("C:/Users/corri/OneDrive/Documents/SonarExperimentData/oculus_sonar_jun_17/Oculus_20250618_103326.jpg");
cropped = oculus(50:942,620:1232,:);
figure(3);
imshow(cropped)
%point (406+520,906+36)
%left (100+520,67+36) right (712+520,69+36)

%pixel_to_polar(40,100,507,872)
figure(4);
imshow("test_images\rectangular_wide.png")

function [r, theta] = pixel_to_polar(row, col, height, width)
    origin = [height round(width/2)]
    r = norm([row col]-origin)
    theta = atan((height-row)/(col-origin(2)))
end

%%
sonar = imread("C:/Users/corri/OneDrive/Documents/SonarExperimentData/07-21-2025/sonar/Oculus_20250721_153416.jpg");
crop_sonar = sonar(50:942,568:1180,:);
imshow(crop_sonar)