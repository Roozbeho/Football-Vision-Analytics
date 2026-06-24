import pytest
import numpy as np

from tests.conftest import DummyPlayer, DummyGoalkeeper, DummyReferee, DummyBall
from visualization.frame_drawer import FrameDrawer
from visualization.pitch_drawer import PitchDrawer


class TestFrameDrawer:

    @pytest.fixture(autouse=True)
    def setup(self, frame, assigner):
        self.frame = frame
        self.drawer = FrameDrawer(assigner)

    @pytest.mark.parametrize("obj", [
        DummyPlayer(),
        DummyGoalkeeper(),
        DummyReferee(),
        DummyBall()
    ])
    def test_draw_all_types(self, obj):
        out = self.drawer.draw(self.frame, obj)
        assert out.shape == self.frame.shape


    def test_zero_bbox(self, assigner):
        class Z:
            class_name = "player"
            bbox = [0, 0, 0, 0]
            track_id = 1

        out = FrameDrawer(assigner).draw(self.frame, Z())
        assert out.shape == self.frame.shape


    def test_none_track_id(self, assigner):
        class N:
            class_name = "player"
            bbox = [10, 10, 50, 50]
            track_id = None

        out = FrameDrawer(assigner).draw(self.frame, N())
        assert out.shape == self.frame.shape

class TestPitchDrawer:

    def test_player(self, frame, pitch, assigner):
        out = PitchDrawer(assigner).draw(
            frame, pitch, DummyPlayer(), (100, 120)
        )
        assert out.shape == pitch.shape


    def test_ball(self, frame, pitch, assigner):
        out = PitchDrawer(assigner).draw(
            frame, pitch, DummyBall(), (50, 60)
        )
        assert out.shape == pitch.shape