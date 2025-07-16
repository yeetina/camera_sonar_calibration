[tvec, rmat] = calculateCSPose(117.4, 123, 50, 0, 6, 2, 3)
rvec = rotmat2vec3d(rmat)

% Add sonar angle, sonar offsets
function [t, rot] = calculateSCPose(x, y, z, pitch1, pitch2, roll, yaw)
% Calculate translation and roation that maps sonar points into camera
% frame
%
% :return t: 3x1 translation vector
% :return rot: 3x3 rotation matrix
    pitch = pitch2;
    
    phi = 5*pitch*pi/180;
    gamma = 5*yaw*pi/180;
    theta = 5*roll*pi/180;
    rotX = [1 0 0; 
            0 cos(phi) -sin(phi);
            0 sin(phi) cos(phi)];
    rotY = [cos(gamma) 0 sin(gamma);
            0 1 0;
            -sin(gamma) 0 cos(gamma)];
    rotZ = [cos(theta) -sin(theta) 0;
            sin(theta) cos(theta) 0;
            0 0 1];
    rotXY = rotX*rotY;
    rot = rotXY*rotZ;

    xtotal = -(45+x+55);
    ytotal = 20+y+35;
    ztotal = -(20+z)+63+8;
    offsetz = -42.8;  %offset from x-axis to O1
    offsetx = 42.8+52.8;  %offset from y-axis to O2
    camx = 52.8; camz = 63; %camy = ; 

    t = [xtotal; 0; 0] + rotX*[0; ytotal; offsetz] + rotXY*[offsetx; 0; ztotal];  %rot*[camx; 0; 0];

end

function [t, rot] = calculateCSPose(x, y, z, pitch1, pitch2, roll, yaw)
% Calculate translation and roation that maps camera points into sonar
% frame
%
% Coordinate system: x right, y down, z forward
% :return t: 3x1 translation vector
% :return rot: 3x3 rotation matrix
    pitch = pitch1+pitch2; %pitch up is positive
    phi = 5*pitch*pi/180;
    gamma = 5*yaw*pi/180;
    theta = 5*roll*pi/180;
    rotX = [1 0 0; 
            0 cos(phi) -sin(phi);
            0 sin(phi) cos(phi)];
    rotY = [cos(gamma) 0 sin(gamma);
            0 1 0;
            -sin(gamma) 0 cos(gamma)];
    rotZ = [cos(theta) -sin(theta) 0;
            sin(theta) cos(theta) 0;
            0 0 1];
    rotZY = rotZ*rotY;
    rot = rotZY*rotX;

    xtotal = -(45+x+55);
    ytotal = 20+y+35;
    ztotal = 20+z-63-8; %TODO: add camera focal point offset
    offsetz = 42.8;  %offset from O1 to O2 in z direction
    offsetx = 42.8+52.8;  %offset from O to O1 in x direction
    %camx = 52.8; camz = 63; camy = ; 
    sony = -29-18; sonz = -91.39+60+58.8; %find real vals

    t = rotZ*[offsetx; 0; ztotal] + rotZY*[0; ytotal; offsetz] + rot*[xtotal; sony; sonz];  %rot*[camx; 0; 0]; 
    clf;
    figure(1);
    plot3([0, 0],[0,0],[0,-80], "b-", "LineWidth", 5)
    hold on;
    segment0 = rotZ*[0 52.8 52.8 offsetx;  0 0 0 0;  -71 -71 ztotal ztotal];
    plot3(segment0(1,:),segment0(2,:),segment0(3,:))
    segment1 = rotZY
    xlabel("x"); ylabel("y"); zlabel("z");
end