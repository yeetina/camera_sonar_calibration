%% Initialize data
labels = ["A1"    "A2"    "A3"    "A4"    "A5"];
%    "B1"    "B2"    "B3"    "B4"    "B5"
%    "C1"    "C2"    "C3"    "C4"    "C5"
%    "D1"    "D2"    "D3"    "D4"    "D5"];

sonar = []

target = [[.02 .02 1]; [.1 .02 1]; [.06 .06 1]; [.14 .06 1]; [.02 .1 1]; [.1 .1 1]].'

st_tvec = [-0.22122347; 0.55729985; -0.16833972]
st_rot = [[ 0.9552258   0.24662684 -0.16345907];
 [ 0.29442119 -0.84705277  0.44251301];
 [-0.02932287 -0.47082566 -0.88173883]]

sonar_frame = st_tvec + st_rot*target

plot3(sonar_frame(1,:), sonar_frame(2,:), sonar_frame(3,:), "ro")

%% 
points = [-0.1972   -0.1208   -0.1491   -0.0727   -0.1775   -0.1010;
    0.5462    0.5698    0.5241   0.5477    0.4785    0.5020;
   -0.1783   -0.1807   -0.1983   -0.2007   -0.2160   -0.2184];

xx = points(1, :);
yy = points(2, :);
zz = points(3, :);
ranges = sqrt(xx .* xx + yy .* yy + zz .* zz);
angles = atan(xx./yy);  %rad

%% 
xs = ranges.*sin(angles);
ys = ranges.*cos(angles);

Pa = target.';

ys3 = [ys.' ys.' ys.'];
xs3 = [xs.' xs.' xs.'];

A = [ys3.*Pa  -xs3.*Pa];

Ata = A.'*A

e = eig(Ata)

%%
rr = [0.6075    0.6099    0.5799    0.5542    0.5567];
angles = [-19.8516  -11.9697  -15.8804  -20.3524  -11.3758];
azi_rad = deg2rad(angles)
sonar_pts = [];

for i = 1:1:length(rr)
    arc = []
    for elev = -5:.2:5
        elev_rad = deg2rad(elev);
        zz = rr(i) * sin(elev_rad);
        xx = rr(i) * cos(elev_rad) .* cos(azi_rad(i));
        yy = rr(i) * cos(elev_rad) .* sin(azi_rad(i));
        arc = [arc; [xx yy zz]]
    end
    sonar_pts = [sonar_pts; arc]
end


%% 

%plot3(sonar_pts(:,1), sonar_pts(:,2), sonar_pts(:,3), "r.")
%fitNormal(sonar_pts, true)
ptCloud = pointCloud(sonar_pts)
[model,inlierIndices,outlierIndices] = pcfitplane(ptCloud, .02)
plane1 = select(ptCloud,inlierIndices);
pcshow(plane1)

function n = fitNormal(data, show_graph)
%FITNORMAL - Fit a plane to the set of coordinates
%
%For a passed list of points in (x,y,z) cartesian coordinates,
%find the plane that best fits the data, the unit vector
%normal to that plane with an initial point at the average
%of the x, y, and z values.
%
% :param data: Matrix composed of of N sets of (x,y,z) coordinates
%              with dimensions Nx3
% :type data: Nx3 matrix
%
% :param show_graph: Option to display plot the result (default false)
% :type show_graph: logical
%
% :return n: Unit vector that is normal to the fit plane
% :type n: 3x1 vector
	
	if nargin == 1
		show_graph = false;
	end
	
	for i = 1:3
		X = data;
		X(:,i) = 1;
		
		X_m = X' * X;
		if det(X_m) == 0
			can_solve(i) = 0;
			continue
		end
		can_solve(i) = 1;
		
		% Construct and normalize the normal vector
		coeff = (X_m)^-1 * X' * data(:,i);
		c_neg = -coeff;
		c_neg(i) = 1;
		coeff(i) = 1;
		n(:,i) = c_neg / norm(coeff);
		
	end
	
	if sum(can_solve) == 0
		error('Planar fit to the data caused a singular matrix.')
		return
	end
	
	% Calculating residuals for each fit
	center = mean(data);
	off_center = [data(:,1)-center(1) data(:,2)-center(2) data(:,3)-center(3)];
	for i = 1:3
		if can_solve(i) == 0
			residual_sum(i) = NaN;
			continue
		end
		
		residuals = off_center * n(:,i);
		residual_sum(i) = sum(residuals .* residuals);
		
	end
	
	% Find the lowest residual index
	best_fit = find(residual_sum == min(residual_sum));
	
	% Possible that equal mins so just use the first index found
	n = n(:,best_fit(1));
	
	if ~show_graph
		return
	end
	
	range = max(max(data) - min(data)) / 2;
	mid_pt = (max(data) - min(data)) / 2 + min(data);
	xlim = [-1 1]*range + mid_pt(1);
	ylim = [-1 1]*range + mid_pt(2);
	zlim = [-1 1]*range + mid_pt(3);
	L=plot3(data(:,1),data(:,2),data(:,3),'ro','Markerfacecolor','r'); % Plot the original data points
	hold on;
	set(get(L, 'Parent'),'DataAspectRatio',[1 1 1],'XLim',xlim,'YLim',ylim,'ZLim',zlim);
	
	norm_data = [mean(data); mean(data) + (n' * range)];

    offset = sqrt(mid_pt(1)^2 + mid_pt(2)^2 + mid_pt(3)^2)
    constantplane(norm_data, offset)
	
	% Plot the original data points
	L=plot3(norm_data(:,1),norm_data(:,2),norm_data(:,3),'b-','LineWidth',3);
	set(get(get(L,'parent'),'XLabel'),'String','x','FontSize',14,'FontWeight','bold')
	set(get(get(L,'parent'),'YLabel'),'String','y','FontSize',14,'FontWeight','bold')
	set(get(get(L,'parent'),'ZLabel'),'String','z','FontSize',14,'FontWeight','bold')
	title(sprintf('Normal Vector: <%0.3f, %0.3f, %0.3f>',n),'FontWeight','bold','FontSize',14)
	grid on;
	axis square;
	hold off;
end
