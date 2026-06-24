from abc import ABC, abstractmethod
from config import PLAYER_COLORS, GOALKEEPER_COLOR, REFEREE_COLOR, BALL_COLOR

class BaseRenderer(ABC):
    PLAYER_COLOR = PLAYER_COLORS
    GOALKEEPER_COLOR = GOALKEEPER_COLOR
    REFEREE_COLOR = REFEREE_COLOR
    BALL_COLOR = BALL_COLOR

    @abstractmethod
    def draw(self, *args, **kwargs):
        pass