import numpy as np

from team_assignment.team_assignment import TeamAssigner
import config

class BallAssigner:
    def __init__(self, team_assigner: TeamAssigner, threshold: int = 35):
        self.team_assigner = team_assigner
        
        self.threshold = config.MAX_BALL_DISTANCE
        
        self._team_1_possession = 0
        self._team_2_possession = 0
        
        self.team1_proto = np.array([0, 0, 0], dtype=np.float64)
        self.team2_proto = np.array([255, 255, 255], dtype=np.float64)
        
    def assign_ball(self, players, ball):
        b_x1, b_y1, b_x2, b_y2 = map(int, ball.bbox)
        
        ball_cx = (b_x1 + b_x2) // 2
        ball_cy = (b_y1 + b_y2) // 2
        
        min_dis = float('inf')
        closest_player = None
        for player in players:
            x1, y1, x2, y2 = map(int, player.bbox)
            
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            dis = np.sqrt((ball_cx - cx) ** 2 + (ball_cy - cy) ** 2)
            
            if dis < min_dis and dis < self.threshold:
                closest_player = player
                min_dis = dis
                
        
        return closest_player       
    
    def get_team_possession(self, frame, closest_frame_player):
        if closest_frame_player is not None:
            color, cluster_idx = self.team_assigner.get_players_team(frame, closest_frame_player)
            color = color.reshape(1, -1)
            
            
            d1 = np.linalg.norm(
                color.reshape(-1).astype(np.int64) -
                self.team1_proto.astype(np.int64)
            )

            d2 = np.linalg.norm(
                color.reshape(-1).astype(np.int64) -
                self.team2_proto.astype(np.int64)
            )

            team_id = np.argmin([d1, d2]) 
            
            # team_id = self.team_assigner.kmeans.predict(color)[0] + 1
            
            if team_id == 0 :
                self._team_1_possession += 1
            else:
                self._team_2_possession += 1
                
        tot = self._team_1_possession + self._team_2_possession
        
        if tot == 0:
            return (0, 0)
            
        return (
            (self._team_1_possession / tot) * 100, 
            (self._team_2_possession / tot) * 100
        )