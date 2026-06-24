import numpy as np
import pytest

class DummyTeamAssigner:
    def get_players_team(self, frame, obj):
        return (255, 0, 0), 1

class DummyPlayer:
    class_name = "player"
    bbox = [10, 10, 50, 80]
    track_id = 2

class DummyGoalkeeper:
    class_name = "goalkeeper"
    bbox = [20, 20, 60, 90]
    track_id = 1

class DummyReferee:
    class_name = "referee"
    bbox = [15, 15, 55, 85]
    track_id = 3

class DummyBall:
    class_name = "ball"
    bbox = [30, 30, 40, 40]

@pytest.fixture
def frame():
    return np.zeros((200, 200, 3), dtype=np.uint8)

@pytest.fixture
def pitch():
    return np.zeros((300, 300, 3), dtype=np.uint8)

@pytest.fixture
def assigner():
    return DummyTeamAssigner()
