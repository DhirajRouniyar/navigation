import cv2
import numpy as np
from scipy.io import loadmat

class Observation_Model:
    def __init__(self):
        self.id_lst = []
        self.image_lst = []
        self.corner_p1_lst = []
        self.corner_p2_lst = []
        self.corner_p3_lst = []
        self.corner_p4_lst = []
        self.timestamp_list = []
        self.rpy_list = []
        self.omg_list = []
        self.acc_list = []
        self.vicon_data = []
        self.time = []
        self.camera_matrix = np.array([[314.1779, 0, 199.4848], [0, 314.2218, 113.7838], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911], dtype=np.float32)
        #Above from the given file
        self.img_coords = {}
        self.pos = []
        self.rot = []
        self.world_coords()

    def read_data(self, file):
        """
        Read the given .mat file and process 
        Input:
            file: .mat file containing the data_
        Return:
            data_list
            vicon_data
            time
        """
        data = loadmat(file, simplify_cells=True)
        data_list = data['data']
        self.time = data['time']

        for index, data_ in enumerate(data_list):
            self.image_lst.append(data_['img'])
            self.id_lst.append(data_['id'])
            self.corner_p1_lst.append(data_['p1'])
            self.corner_p2_lst.append(data_['p2'])
            self.corner_p3_lst.append(data_['p3'])
            self.corner_p4_lst.append(data_['p4'])

            self.rpy_list.append(data_['rpy'])
            self.acc_list.append(data_['acc'])

            self.img_coords[index] = {}
            if isinstance(data_['id'], int):

                self.img_coords[index][data_['id']] = np.array([[data_['p1'][0], data_['p1'][1]],
                                                                [data_['p2'][0], data_['p2'][1]],
                                                                [data_['p3'][0], data_['p3'][1]],
                                                                [data_['p4'][0], data_['p4'][1]]])
                self.timestamp_list.append(data_['t'])
                continue

            for idx, tag in enumerate(data_['id']):
                self.img_coords[index][tag] = np.array([[data_['p1'][0][idx], data_['p1'][1][idx]],
                                                        [data_['p2'][0][idx], data_['p2'][1][idx]],
                                                        [data_['p3'][0][idx], data_['p3'][1][idx]],
                                                        [data_['p4'][0][idx], data_['p4'][1][idx]]])
            if data_['id'].shape[0] > 1:
                self.timestamp_list.append(data_['t'])


        self.vicon_data = data['vicon']

        return data_list, self.vicon_data, self.time

    def world_coords(self):
        """
        AprilTags world coordinates
        Return:
            None
        """
        pos = [
            [0, 12, 24, 36, 48, 60, 72, 84, 96],
            [1, 13, 25, 37, 49, 61, 73, 85, 97],
            [2, 14, 26, 38, 50, 62, 74, 86, 98],
            [3, 15, 27, 39, 51, 63, 75, 87, 99],
            [4, 16, 28, 40, 52, 64, 76, 88, 100],
            [5, 17, 29, 41, 53, 65, 77, 89, 101],
            [6, 18, 30, 42, 54, 66, 78, 90, 102],
            [7, 19, 31, 43, 55, 67, 79, 91, 103],
            [8, 20, 32, 44, 56, 68, 80, 92, 104],
            [9, 21, 33, 45, 57, 69, 81, 93, 105],
            [10, 22, 34, 46, 58, 70, 82, 94, 106],
            [11, 23, 35, 47, 59, 71, 83, 95, 107]
        ]
        tag_pos = np.array(pos)
        # Store 4 corners of the AprilTag
        self.tag_corners = {}
        tag_size = 0.152  # in meters

        # Spacing between tags
        spacing_x = 0.152  # in meters
        spacing_y = 0.152  # in meters
        special_spacing_y = 0.178  # in meters

        # Each tag iterate
        for i in range(tag_pos.shape[0]):
            for j in range(tag_pos.shape[1]):
                x_val = i * spacing_x * 2
                y_val = j * spacing_y * 2

                # special spacing, adjust y_val coordinate for 
                if j + 1 >= 3:
                    y_val += (special_spacing_y - spacing_y)
                if j + 1 >= 6:
                    y_val += (special_spacing_y - spacing_y)
                self.tag_corners[tag_pos[i, j]] = np.array([
                                                        [x_val + tag_size, y_val, 0],  # bottom left corner
                                                        [x_val + tag_size, y_val + tag_size, 0],  # bottom right corner
                                                        [x_val, y_val + tag_size, 0],  # top right corner
                                                        [x_val, y_val, 0] # top left corner
                                                    ])

    def estimate_pose(self, tags, idx):
    
        image_points = []
        object_points = []

        for i, tag in enumerate(tags):
            image_points.append(self.img_coords[idx][tag][0])
            image_points.append(self.img_coords[idx][tag][1])
            image_points.append(self.img_coords[idx][tag][2])
            image_points.append(self.img_coords[idx][tag][3])

            object_points.append(self.tag_corners[tag][0])
            object_points.append(self.tag_corners[tag][1])
            object_points.append(self.tag_corners[tag][2])
            object_points.append(self.tag_corners[tag][3])

        # Compute the pose of the drone using the solvePnP function
        _, rvec, tvec = cv2.solvePnP(np.array(object_points), 
                                       np.array(image_points),
                                       self.camera_matrix, 
                                       self.dist_coeffs)
        
        camera_pos = np.array([-0.04, 0.0, -0.03])  # coordinates of the camera
        yaw = np.pi / 4                             # Rotation mat, yaw angle, Yaw angle of the camera
        rot_z = self.rotation_matrix_z(yaw)
        rot_x = self.rotation_matrix_x(np.pi)       # Rotation mat, pitch angle
        camera_R = rot_x @ rot_z                    # Rotation mat of the camera
        camera_drone_mat = np.eye(4)                # Transformation mat, camera to drone
        camera_drone_mat[:3, :3] = camera_R
        camera_drone_mat[:3, 3] = camera_pos
        R, _ = cv2.Rodrigues(rvec)
        camera_object_mat = np.eye(4)               # Transformation mat, object to camera
        camera_object_mat[:3, :3] = R
        camera_object_mat[:3, 3] = tvec.flatten()
        
        object_drone_mat = np.linalg.inv(camera_object_mat) @ camera_drone_mat # Transformation mat, object to drone
        object_drone_rot = self.rotation_to_euler(object_drone_mat[:3, :3])
        object_drone_trans = object_drone_mat[:3, 3]

        self.pos.append([object_drone_trans[0], object_drone_trans[1], object_drone_trans[2]])
        self.rot.append([object_drone_rot[0], object_drone_rot[1], object_drone_rot[2]])
        return [object_drone_trans[0], object_drone_trans[1], object_drone_trans[2]], [object_drone_rot[0], object_drone_rot[1], object_drone_rot[2]]

    def rotation_to_euler(self, Rot):
        yaw = np.arctan(-Rot[0, 1] / Rot[1, 1])
        roll = np.arctan(Rot[2,1] * np.cos(yaw) / Rot[1,1])
        pitch = np.arctan(-Rot[2, 0] / Rot[2, 2])
        return np.array([yaw, pitch, roll])

    def rotation_matrix_x(self, angle):
        return np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])

    def rotation_matrix_y(self, angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])

    def rotation_matrix_z(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])