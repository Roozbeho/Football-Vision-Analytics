class StubTransform:
    prev_ball_pos = None
    def goal_checker(self, pos):
        self.prev_ball_pos = pos
        return False

class TestGoalChecker:
    def setup_method(self):
        self.t = StubTransform()

    def test_no_goal(self):       
        assert self.t.goal_checker((510, 300)) is False
    def test_state_update(self):  
        self.t.goal_checker((110, 100))
        assert self.t.prev_ball_pos == (110, 100)
    def test_vertical(self):      
        assert self.t.goal_checker((10, 200)) is False
