import numpy as np
import pickle
import calibration_gui as gui

def generate_calibration_groups(good):
    """
    Will output all possible groupings of timestamps from good
    If good has N timestamps, it will return all groups of size 1 to N
    """
    if len(good) == 1:
        return [(good[0],)]
    else:
        first = (good[0],)
        others = generate_calibration_groups(good[1:])
        combos = [first+other for other in others]
        return [first]+others+combos
        
def save_csv_data(timestamp_groups):
    #combos = self.generate_calibration_groups(list(self.calibration_results.keys()))
    my_data = []
    for group in timestamp_groups:
        err, r, t = gui.multi_calibration(group)
        rlin = [r[0][0], r[1][0], r[2][0]]
        tlin = [t[0][0], t[1][0], t[2][0]]
        my_data.append([len(group), err, *rlin, *tlin])
    #print(my_data)
    my_data = np.array(my_data)
    np.savetxt("data4.csv", my_data, 
            delimiter = ",")    
        
poses_file = "C:/Users/corri/OneDrive/Documents/SonarExperimentData/07-23-2025/output/camera_poses.pkl"
with open(poses_file, "rb") as fp:
    poses = pickle.load(fp)
    flattened = []
    for vec in poses.values():
        rvec, tvec = vec[0], vec[1]
        rvec = np.reshape(rvec, (1, 3))
        tvec = np.reshape(tvec, (1, 3))
        line = np.concat([rvec, tvec], axis=1)
        flattened.append(line)
    print(flattened)
    poses = np.array(flattened)
    poses = np.vstack(poses)
    print("poses", poses.shape, poses)

np.savetxt("posesdata.csv", poses, delimiter = ",")