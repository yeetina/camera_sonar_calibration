patternDims = [10 7];
family = "DICT_4X4_100";
imageSize = [2000 1400];
checkerSize = 200;
markerSize = 150;


I = generateCharucoBoard(imageSize,patternDims,family,checkerSize,markerSize);
imshow(I)