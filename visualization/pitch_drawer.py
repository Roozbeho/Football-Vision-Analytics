import cv2
import numpy as np

from visualization.base_renderer import BaseRenderer

class PitchDrawer(BaseRenderer):   
    PLAYER_PITCH_COLOR = {1: [210, 80, 160], 2: [70, 200, 220]}
    
    def __init__(self, team_assigner):
        self.team_assigner = team_assigner
             
    def draw(self, frame, current_pitch, obj, pitch_cord):
        x, y = map(int, pitch_cord)
        
        if obj.class_name == "player": 
            color, cluster_idx = self.team_assigner.get_players_team(frame, obj)
            
            current_pitch = cv2.circle(current_pitch, (x, y), 25, color, -1)
            
        if obj.class_name == "goalkeeper": 
            current_pitch = cv2.circle(current_pitch, (x, y), 25, self.GOALKEEPER_COLOR, -1)
            
        if obj.class_name == "referee": 
            current_pitch = cv2.circle(current_pitch, (x, y), 25, self.REFEREE_COLOR, -1)
            
        if obj.class_name == "ball": 
            current_pitch = cv2.circle(current_pitch, (x, y), 15, self.BALL_COLOR, -1)

        return current_pitch

    
