import numpy as np
import cv2
from typing import Tuple, Dict
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=4):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    if qs is not None:
        qs = qs.astype(np.int32)
        for k in range(0,4):
           i,j=k,(k+1)%4
           image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness) # use LINE_AA for opencv3

           i,j=k+4,(k+1)%4 + 4
           image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

           i,j=k,k+4
           image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
    return image

class Box4Track(Box):
    def to_camera_frame(self,
                        pose_record: Dict,
                        calibrated_sensor_record: Dict) -> None:
        """
        Transform box from global frame to camera frame
        """
        # move box to ego_vehicle coord system
        self.translate(-np.array(pose_record['translation']))
        self.rotate(Quaternion(pose_record['rotation']).inverse)

        # move box to camera coord system
        self.translate(-np.array(calibrated_sensor_record['translation']))
        self.rotate(Quaternion(calibrated_sensor_record['rotation']).inverse)

    def render_track(self,
                     im: np.ndarray,
                     view: np.ndarray,
                     normalize: bool = False,
                     color: Tuple = (255, 0, 0),
                     linewidth: int = 2) -> None:
        """
        Render untight bounding box on imaae, with track name on it
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]
        # find untight bounding box
        min_size = np.amin(corners, axis=1)
        max_size = np.amax(corners, axis=1)
        im=draw_projected_box3d(image=im,qs=corners.T,color=color)
        # cv2.rectangle(im,
        #               (int(min_size[0]), int(min_size[1])),
        #               (int(max_size[0]), int(max_size[1])),
        #               color, linewidth)
        # draw label
        label = self.name + ' ' + str(self.label)

        text_thickness = 1
        font_scale = 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)

        text_x = min_size[0]
        text_y = max(min_size[1] - text_thickness, 0)

        textbox_x = min(min_size[0] + text_size[0][0], im.shape[1])
        textbox_y = max(min_size[1] - 2 * text_thickness - text_size[0][1], 0)

        cv2.rectangle(im,
                      (int(min_size[0]), int(min_size[1])),
                      (int(textbox_x), int(textbox_y)),
                      color, -1)
        cv2.putText(im,
                    label,
                    (int(text_x), int(text_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), text_thickness)
