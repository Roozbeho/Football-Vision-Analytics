import cv2
import numpy as np
from argparse import ArgumentParser

import config
from yolo import CustomYOLO
from tracking.tracker import Tracker
from team_assignment.team_assignment import TeamAssigner
from transformation.transformer import Transform
from ball_assigner.ball_assigner import BallAssigner


from visualization.frame_drawer import FrameDrawer
from visualization.pitch_drawer import PitchDrawer
from visualization.overly_renderer import visualize_pitch_points
from visualization.ball_possession import visualize_ball_possession

class FootballAnalyzer:
    def __init__(self, yolo):
        base_pitch_map = cv2.imread(config.PITCH_IMG_PATH)
        self.pitch_H, self.pitch_w = base_pitch_map.shape[:2]

        self.tracker = Tracker(yolo, config.DEVICE)
        self.team_assigner = TeamAssigner(yolo)
        self.transform = Transform(yolo, self.team_assigner, base_pitch_map, base_pitch_map.shape[:2])
        self.frame_drawer = FrameDrawer(self.team_assigner)
        self.pitch_drawer = PitchDrawer(self.team_assigner)
        self.ball_assigner = BallAssigner(self.team_assigner)

        self.cap = cv2.VideoCapture(config.VIDEO_PATH)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            config.OUTPUT_PATH,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
    def process_frame(self, frame, frame_id):
        self.tracker.track(frame, frame_id)
        
        self.team_assigner.player_color_extraction(frame, self.tracker.tracks[frame_id]['players'])
            
        frame_points, pitch_points = self.transform.keypoint_extraction(frame)
        
        closest_player = self.ball_assigner.assign_ball(
            self.tracker.tracks[frame_id]['players'], self.tracker.tracks[frame_id]['ball'][0])
        
        (possesion_1, posession2) = self.ball_assigner.get_team_possession(frame, closest_player)        
        
        if frame_points is not None:
            H, mask = self.transform._find_homography(frame_points, pitch_points)
            if H is not None:
                self.transform.last_H = H
        else:
            H = self.transform.last_H
            
        current_pitch_map = self.transform.base_pitch_map.copy()
        
        return H, current_pitch_map, possesion_1, posession2
    
    def run(self):
        frame_id: int = 0
        
        cv2.namedWindow("football analytics", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("football analytics", 1280, 720)
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            
            if not success:
                break
            
            H, current_pitch_map, possesion_1, posession2 = self.process_frame(frame, frame_id)
            
            players = self.tracker.tracks[frame_id].get('players', [])
            referees = self.tracker.tracks[frame_id].get('referees', [])
            goalkeepers = self.tracker.tracks[frame_id].get('goalkeepers', [])
            ball = self.tracker.tracks[frame_id].get('ball', None)
            
            all_objects = []

            all_objects.extend(players)
            all_objects.extend(referees)
            all_objects.extend(goalkeepers)

            if ball:
                all_objects.extend(ball)
            
            if H is not None:
                transformed_pts = self.transform.transform_frame_point_to_pitch(all_objects, H)
            else:
                transformed_pts = np.zeros_like(all_objects).tolist()
            
            for obj, pitch_pt in zip(all_objects, transformed_pts):


                frame = self.frame_drawer.draw(frame, obj)

                if H is not None:
                    current_pitch_map = self.pitch_drawer.draw(
                        frame, current_pitch_map, obj, pitch_pt)
                    
                    if obj.class_name == 'ball':
                        if self.transform.goal_checker(pitch_pt):
                            print("*"*500)

            frame = visualize_ball_possession(frame, (possesion_1, posession2))
            frame = visualize_pitch_points(frame, current_pitch_map, self.pitch_H, self.pitch_w)
                
            self.writer.write(frame)
            cv2.imshow("football analytics", frame)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_id += 1
            
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--object_model_path", type=str, default=config.OBJECT_MODEL_PATH, help="path to the player detection model")
    parser.add_argument("--keypoint_model_path", type=str, default=config.KEYPOINT_MODEL_PATH, help="path to the keypoint detection model")
    parser.add_argument("--video_path", type=str, default=config.VIDEO_PATH, help="path to the input video")
    parser.add_argument("--output_path", type=str, default=config.OUTPUT_PATH, help="path to the output video")
    parser.add_argument("--pitch_img_path", type=str, default=config.PITCH_IMG_PATH, help="path to the pitch image")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="device to run the models on")
        
    args = parser.parse_args()
        
    yolo = CustomYOLO(args.object_model_path, args.keypoint_model_path, args.device)
    
    footballanalyzer = FootballAnalyzer(yolo)
    
    footballanalyzer.run()