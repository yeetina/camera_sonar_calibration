poses = readtable("posesdata.csv");
poses2 = readtable("posesdata2.csv");

rx = [table2array(poses(:,"Var1")); table2array(poses2(:,"Var1"))];
ry = [table2array(poses(:,"Var2")); table2array(poses2(:,"Var2"))];
rz = [table2array(poses(:,"Var3")); table2array(poses2(:,"Var3"))];
tx = table2array(poses(:,"Var4"));
ty = table2array(poses(:,"Var5"));
tz = table2array(poses(:,"Var6"));
tz = 2-tz;

tx2 = table2array(poses2(:,"Var4"));
ty2 = table2array(poses2(:,"Var5"));
tz2 = table2array(poses2(:,"Var6"));
tz2 = 2-tz2;

z_vecs = calculate_skew(rx, ry, rz)
x_starts = [tx; tx2].';
x_ends = x_starts + z_vecs(1,:)
y_starts = [ty; ty2].';
y_ends = y_starts + z_vecs(2,:)

chart = [x_starts; y_starts; x_ends; y_ends]


hold on;
xlim([-.5 .33])
ylim([-0.4, 0.6])
bubblechart(tx, ty, tz, "blue", 'MarkerFaceAlpha', 0.4)
bubblechart(tx2, ty2, tz2, "green", 'MarkerFaceAlpha', 0.4)
bubblesize([18 28])
% we want x0 at 60% from the left and y0 at 40% from the top
set(gca(),'YDir','reverse')
plot([x_starts; x_ends], [y_starts; y_ends], "k", "LineWidth", 2)
plot(x_starts, y_starts, "k.", "MarkerSize", 10)
xlabel("camera X (m)")
ylabel("camera Y (m)")

%% 
calibration_data = readtable("data5.csv");
calibration_data = renamevars(calibration_data,["Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "Var8"], ...
    ["Pairs", "Error", "Rx", "Ry", "Rz", "Tx", "Ty", "Tz"]);
num_pairs = table2array(calibration_data(:,"Pairs"));
rvecs = table2array(calibration_data(:, ["Rx", "Ry", "Rz"]));
tvecs = table2array(calibration_data(:, ["Tx", "Ty", "Tz"]));
r_ext = [0  -0.2618  0];
t_ext = [0.2092  -0.1608  0.1032];
r_errs = rvecs - r_ext;
t_errs = 100*(tvecs - t_ext);
r_sum = sqrt(r_errs(:,1).^2 + r_errs(:,2).^2 + r_errs(:,3).^2);
t_sum = sqrt(t_errs(:,1).^2 + t_errs(:,2).^2 + t_errs(:,3).^2); 
t_avgs = splitapply(@mean, t_sum, num_pairs);
r_avgs = splitapply(@mean, r_sum, num_pairs);

figure(1);
clf;
plot(num_pairs, t_sum, "k.")
hold on;
plot(t_avgs, "r.", MarkerSize=12)
plot(t_avgs, "r-")
title("Total Translation Error (cm)")

figure(2);
clf;
plot(num_pairs, r_sum, "k.")
hold on;
plot(r_avgs, "r.", MarkerSize=12)
plot(r_avgs, "r-")
title("Total Rotation Error (rad)")

%%
figure(3);
t = tiledlayout(3, 1, "TileSpacing", "compact");
title(t,'Translation Errors')
xlabel(t, "Number of images used in calibration")
ylabel(t, "Error (cm)")

nexttile
plot(num_pairs, t_errs(:,1), "r.", MarkerSize=8)
hold on;
tx_avgs = splitapply(@mean, t_errs(:,1), num_pairs);
plot(tx_avgs, "k.", MarkerSize=12, MarkerEdgeColor="k")
set(gca,'xtick',[])
grid on;
title("x-axis")

nexttile
plot(num_pairs, t_errs(:,2), "g.", MarkerSize=8)
hold on;
ty_avgs = splitapply(@mean, t_errs(:,2), num_pairs);
plot(ty_avgs, "k.", MarkerSize=12, MarkerEdgeColor="k")
set(gca,'xtick',[])
grid on;
title("y-axis")

nexttile
plot(num_pairs, t_errs(:,3), "b.", MarkerSize=8)
hold on;
tz_avgs = splitapply(@mean, t_errs(:,3), num_pairs);
plot(tz_avgs, "k.", MarkerSize=12)
grid on;
title("z-axis")

%%
figure(4);
t = tiledlayout(3, 1, "TileSpacing", "compact");
title(t,'Rotation Errors')
xlabel(t, "Number of images used in calibration")
ylabel(t, "Error (rad)")

nexttile
plot(num_pairs, r_errs(:,1), "r.", MarkerSize=8)
hold on;
rx_avgs = splitapply(@mean, r_errs(:,1), num_pairs);
plot(rx_avgs, "k.", MarkerSize=12, MarkerEdgeColor="k")
set(gca,'xtick',[])
grid on;
title("about x-axis")

nexttile
plot(num_pairs, r_errs(:,2), "g.", MarkerSize=8)
hold on;
ry_avgs = splitapply(@mean, r_errs(:,2), num_pairs);
plot(ry_avgs, "k.", MarkerSize=12, MarkerEdgeColor="k")
set(gca,'xtick',[])
grid on;
title("about y-axis")

nexttile
plot(num_pairs, r_errs(:,3), "b.", MarkerSize=8)
hold on;
rz_avgs = splitapply(@mean, r_errs(:,3), num_pairs);
plot(rz_avgs, "k.", MarkerSize=12)
grid on;
title("about z-axis")

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