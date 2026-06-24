import cv2
import numpy as np

from visualization.base_renderer import BaseRenderer

class FrameDrawer(BaseRenderer):
    def __init__(self, team_assigner):
        self.team_assigner = team_assigner
        
    def draw(self, frame, obj):
        if obj.class_name == "player": 
            frame = self._draw_player(frame, obj)
            
        if obj.class_name == "goalkeeper": 
            frame = self._draw_goalkeeper(frame, obj)
            
        if obj.class_name == "referee": 
            frame = self._draw_refree(frame, obj)
            
        if obj.class_name == "ball": 
            frame = self._draw_ball(frame, obj)
            
                
        
        return frame
    
    def _draw_id(self, frame, obj, color):
        x1, y1, x2, y2 = map(int, obj.bbox)
        
        x_center = (x1 + x2) // 2
        if obj.track_id is not None:
            x1_rec = x_center - 40 // 2
            x2_rec = x_center + 40 // 2
            y1_rec = (y2 - 20 // 2) + 15
            y2_rec = (y2 + 20 // 2) + 15
            
            cv2.rectangle(
                frame, (int(x1_rec), int(y1_rec)), (int(x2_rec), int(y2_rec)),
                color, cv2.FILLED
            )
            
            cv2.putText(
                frame, f"{obj.track_id}", (int(x1_rec+12), int(y1_rec+15)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                0.6, (0, 0, 0), 2
            )
            
        return frame
    
    def _draw_player(self, frame, obj):
        x1, y1, x2, y2 = map(int, obj.bbox)
        
        color, _ = self.team_assigner.get_players_team(frame, obj)
        # color = self.PLAYER_COLOR[team]
        
        x_center = (x1 + x2) // 2
        W = int(y2 - y1)

        cv2.ellipse(
            frame, center=(x_center, y2), axes=(int(0.8 * W), int(0.2*W)),
            angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4
        )
        
        frame = self._draw_id(frame, obj, color)
        
        return frame
    
    def _draw_goalkeeper(self, frame, obj):
        x1, y1, x2, y2 = map(int, obj.bbox)
        
        x_center = (x1 + x2) // 2
        W = int(y2 - y1)

        cv2.ellipse(
            frame, center=(x_center, y2), axes=(int(0.8 * W), int(0.2*W)),
            angle=0.0, startAngle=-45, endAngle=235, color=self.GOALKEEPER_COLOR, thickness=2, lineType=cv2.LINE_4
        )
        
        frame = self._draw_id(frame, obj, self.GOALKEEPER_COLOR)
        
        return frame
    
    def _draw_refree(self, frame, obj):
        x1, y1, x2, y2 = map(int, obj.bbox)

        x_center = (x1 + x2) // 2
        W = int(y2 - y1)

        cv2.ellipse(
            frame, center=(x_center, y2), axes=(int(0.8 * W), int(0.2*W)),
            angle=0.0, startAngle=-45, endAngle=235, color=self.REFEREE_COLOR, thickness=2, lineType=cv2.LINE_4
        )
        
        return frame
    
    def _draw_ball(self, frame, obj):
        x1, y1, x2, y2 = map(int, obj.bbox)
        x_center = (x1 + x2) // 2
    
        tri_points = np.array([[x_center, y1], [x_center-10, y1-20], [x_center+10, y1-20]])
        
        cv2.drawContours(frame, [tri_points], 0, self.BALL_COLOR, cv2.FILLED)
        
        return frame

