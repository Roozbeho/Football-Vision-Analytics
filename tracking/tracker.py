import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pickle
from dataclasses import dataclass
from typing import List
from collections import defaultdict

@dataclass
class TrackedObject:
    track_id: int
    bbox: List
    confidence: float
    class_name: str
    # feat_map: np.array


class Tracker:
    def __init__(self, yolo_model, device: str):
        self.model = yolo_model
        self.tracker = sv.ByteTrack(track_activation_threshold=0.3, minimum_matching_threshold=0.5)
        
        self.class_map = {
            0: 'ball',
            1: 'goalkeeper',
            2: 'player',
            3: 'referee'
        }
        self.tracks = defaultdict(list)
        self.ball_pos = ()
        
    def detect(self, frame):
        results, feat_maps = self.model.call_yolo(frame)
        detection = sv.Detections.from_ultralytics(results[0])
        return detection, results[0], results[0].orig_shape
                            
    def track(self, frame, frame_id):
        detections, result, (H, W) = self.detect(frame)
        tracked_objs = self.tracker.update_with_detections(detections)
        
        frame_tracks = []
        
        frame_tracks = {
            "players": [],
            "referees": [],
            "goalkeepers": [],
            "ball": []
        }
        
        for bbox, track_id, class_id, conf in zip(
            tracked_objs.xyxy, tracked_objs.tracker_id, tracked_objs.class_id, 
            tracked_objs.confidence
        ):
            
            class_name = self.class_map.get(int(class_id), "unknown")
            
            obj = TrackedObject(
                track_id=int(track_id) if track_id is not None else -1,
                bbox=bbox.tolist() ,
                confidence=float(conf),
                class_name=class_name,
            )
            
            if class_name == "player":
                frame_tracks["players"].append(obj)

            elif class_name == "referee":
                frame_tracks["referees"].append(obj)

            elif class_name == "goalkeeper":
                frame_tracks["goalkeepers"].append(obj)

            elif class_name == "ball":
                frame_tracks["ball"].append(obj)
        
        if not frame_tracks["ball"]:
            self._update_ball_position(detections)

            if self.ball_pos:

                obj = TrackedObject(
                    track_id=-1,
                    bbox=list(self.ball_pos),
                    confidence=1.0,
                    class_name="ball",
                )

                frame_tracks["ball"].append(obj)
        else:
            self.ball_pos = frame_tracks["ball"][0].bbox
                        
                        
        self.tracks[frame_id] = frame_tracks
            
        return frame_tracks

    def _update_ball_position(self, detections):

        ball_indices = np.where(detections.class_id == 0)[0]

        if len(ball_indices) == 0:
            return False

        if not self.ball_pos:

            idx = ball_indices[0]

            self.ball_pos = detections.xyxy[idx].tolist()

            return True

        prev_x1, prev_y1, prev_x2, prev_y2 = self.ball_pos

        prev_cx = (prev_x1 + prev_x2) / 2
        prev_cy = (prev_y1 + prev_y2) / 2

        best_dist = float("inf")
        best_bbox = None

        for idx in ball_indices:
            conf = detections.confidence[idx]

            x1, y1, x2, y2 = detections.xyxy[idx]

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            dist = np.sqrt(
                (cx - prev_cx) ** 2 +
                (cy - prev_cy) ** 2
            )

            if dist < best_dist:
                best_dist = dist
                best_bbox = [x1, y1, x2, y2]

        self.ball_pos = best_bbox

