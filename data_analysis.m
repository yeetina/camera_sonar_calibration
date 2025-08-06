poses = readtable("posesdata.csv");
rx = table2array(poses(:,"Var1"));
ry = table2array(poses(:,"Var2"));
rz = table2array(poses(:,"Var3"));
tx = table2array(poses(:,"Var4"));
ty = table2array(poses(:,"Var5"));
tz = table2array(poses(:,"Var6"));
tz = 2-tz;

z_vecs = calculate_skew(rx, ry, rz)
x_starts = tx.';
x_ends = x_starts + z_vecs(1,:)
y_starts = ty.';
y_ends = y_starts + z_vecs(2,:)

chart = [x_starts; y_starts; x_ends; y_ends]


hold on;
xlim([-.5 .33])
ylim([-0.2, 0.3])
bubblechart(tx, ty, tz, "blue", 'MarkerFaceAlpha',0.4)
bubblesize([18 28])
% we want x0 at 60% from the left and y0 at 40% from the top
set(gca(),'YDir','reverse')
plot([x_starts; x_ends], [y_starts; y_ends], "k", "LineWidth", 2)
plot(x_starts, y_starts, "k.", "MarkerSize", 10)
xlabel("camera X (m)")
ylabel("camera Y (m)")
%%
imshow("C:/Users/corri/OneDrive/Documents/SonarExperimentData/07-23-2025/camera/20250723_163603.png")
% 382, 252
% center is 769, 290
% dx = -387, dy = -38
% -.2724, -.03
% x = -.54 to ,359
% y = -.229 to .339

function Zvec = calculate_skew(rx, ry, rz)
    Zvec = [];
    for i = 1:1:length(rx)
        radx = rx(i);
        rady = ry(i);
        radz = rz(i);
        rotX = [1 0 0; 
                0 cos(radx) -sin(radx);
                0 sin(radx) cos(radx)];
        rotY = [cos(rady) 0 sin(rady);
                0 1 0;
                -sin(rady) 0 cos(rady)];
        rotZ = [cos(radz) -sin(radz) 0;
                sin(radz) cos(radz) 0;
                0 0 1];
        rot = rotX*rotY*rotZ;
        z_unit = [0; 0; 0.1];
        Zvec = [Zvec rot*z_unit];
    end
end