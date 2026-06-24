import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple
from collections import defaultdict

from tracking.tracker import TrackedObject

@dataclass
class PitchTrackedObject:
    track_id: int
    bbox: List
    class_name: str
    pitch_cord: Tuple[float, float]

class Transform:
    def __init__(
        self, 
        yolo,
        team_assigner,
        base_pitch_map,
        pitch_cord,
        pitch_img_path="/home/roozbeh/programming/DL/DL_uni/football/top-view-of-green-football-pitch-or-soccer-field-vector.jpg"):
        
        self.yolo = yolo
        self.team_assigner = team_assigner
        
        self.base_pitch_map = base_pitch_map
        self.pitch_h, self.pitch_w = pitch_cord
        
        self.pitch_points = np.array([
            [94, 65], [94, 230], [94, 384], [94, 596], [94, 733], [94, 916],
            [164, 384], [164, 596],
            [236, 485],
            [298, 240], [306, 395], [306, 581], [306, 735],
            [733, 65], [734, 384], [732, 594], [731, 916],
            [1162, 242], [1155, 315], [1162, 579], [1166, 741],
            [1235, 485],
            [1316, 385], [1315, 599],
            [1376, 64], [1372, 239], [1374, 383], [1376, 596], [1372, 735], [1376, 919],
            [626, 484], [843, 479]
        ], dtype=np.float32)
        
        self.r_goal_points = np.array([[93, 438], [93, 536]], dtype=np.float32)
        self.l_goal_points = np.array([[1378, 436], [1376, 538]], dtype=np.float32)
        self.prev_ball_pos = None

        
        self.pitch_tracks = defaultdict(list)
        
        self.last_H = None
        self.current_pitch_map = None
        

        self.h_alpha = 0.85
        self.max_reproj_error = 20
    
    def keypoint_extraction(self, img):

        result = self.yolo.call_yolo_pose(img)
    
        if (
            result.keypoints is None
            or result.keypoints.xy is None
            or len(result.keypoints.xy) == 0
        ):
            return None, None

        keypoints = result.keypoints.xy.cpu().numpy()[0]
        conf = result.keypoints.conf.cpu().numpy()[0]

        
        valid_mask = conf > 0.5

        frame_points = keypoints[valid_mask]
        pitch_points = self.pitch_points[valid_mask]

        non_zero_mask = ~np.all(frame_points == 0, axis=1)

        frame_points = frame_points[non_zero_mask]
        pitch_points = pitch_points[non_zero_mask]

        return frame_points.astype(np.float32), pitch_points.astype(np.float32)
        
        
    # def _find_homography(self, frame_points, pitch_points):

    #     # if len(frame_points) < 4:
    #     #     return None, None

    #     H, mask = cv2.findHomography(frame_points, pitch_points, cv2.RANSAC, 10.0)

    #     return H, mask
    
    def _find_homography(
        self,
        frame_points,
        pitch_points
    ):
        
        if len(frame_points) < 4:
            return self.last_H, None

        H_new, mask = cv2.findHomography(frame_points, pitch_points, cv2.RANSAC, 5.0)

        if H_new is None:
            return self.last_H, None


        reproj_error = self.compute_reprojection_error(H_new, frame_points, pitch_points)

        if reproj_error > self.max_reproj_error:
            return self.last_H, None

        if self.last_H is None:

            H_smoothed = H_new

        else:

            H_smoothed = self.h_alpha * self.last_H + (1.0 - self.h_alpha) * H_new


            H_smoothed = H_smoothed / H_smoothed[2, 2]

        self.last_H = H_smoothed

        return H_smoothed, mask  
    
    def transform_frame_point_to_pitch(self, all_objects, H):
        pts = []

        for obj in all_objects:

            x1, y1, x2, y2 = map(int, obj.bbox)

            x_center = (x1 + x2) / 2
            bottom_y = y2

            pts.append([x_center, bottom_y])


        pts = np.array(pts, dtype=np.float32)
        pts = pts.reshape(-1, 1, 2)


        transformed_pts = cv2.perspectiveTransform(
            pts,
            H
        ).reshape(-1, 2)
        
        return transformed_pts
   
    def goal_checker(self, ball_pitch_cord):

        if self.prev_ball_pos is None:
            self.prev_ball_pos = ball_pitch_cord
            return False

        prev_x, prev_y = self.prev_ball_pos
        curr_x, curr_y = ball_pitch_cord

        goal = False

        if (
            self.r_goal_points[0][1] < curr_y < self.r_goal_points[1][1]
            and prev_x <= self.r_goal_points[0][0]
            and curr_x > self.r_goal_points[0][0]
        ):
            goal = True

        if (
            self.l_goal_points[0][1] < curr_y < self.l_goal_points[1][1]
            and prev_x >= self.l_goal_points[0][0]
            and curr_x < self.l_goal_points[0][0]
        ):
            goal = True

        self.prev_ball_pos = ball_pitch_cord

        return goal

    def compute_reprojection_error(
        self, H, frame_points, pitch_points):
        projected = cv2.perspectiveTransform(
            frame_points.reshape(-1, 1, 2), H).reshape(-1, 2)

        errors = np.linalg.norm(projected - pitch_points, axis=1)

        return np.mean(errors)
