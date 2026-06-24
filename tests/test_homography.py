import numpy as np
import cv2
import pytest

class TestPerspectiveTransform:
    def _apply(self, H, pts):
        return cv2.perspectiveTransform(np.array(pts, dtype=np.float32), H).reshape(-1, 2)

    def test_identity(self):
        H = np.eye(3, dtype=np.float32)
        np.testing.assert_allclose(self._apply(H, [[[100.,200.],[300.,400.]]]), [[100,200],[300,400]], atol=1e-4)

    def test_scale(self):
        H = np.diag([2., 2., 1.]).astype(np.float32)
        np.testing.assert_allclose(self._apply(H, [[[50.,75.]]]), [[100.,150.]], atol=1e-4)

    def test_multi_point(self):
        H = np.diag([3., 3., 1.]).astype(np.float32)
        np.testing.assert_allclose(self._apply(H, [[[1,1],[2,2],[3,3]]]), [[3,3],[6,6],[9,9]], atol=1e-4)

    def test_noise_stability(self):
        H = np.eye(3, dtype=np.float32)
        pts = np.array([[[10.,20.]]], dtype=np.float32)
        noisy = (pts + np.random.normal(0, 1e-6, pts.shape)).astype(np.float32)
        np.testing.assert_allclose(cv2.perspectiveTransform(noisy, H), pts, atol=1e-3)

    def test_invalid_raises(self):
        pts = np.array([[[1, 1]]], dtype=np.float32)

        with pytest.raises(Exception):
            cv2.perspectiveTransform(pts, None)

class TestReprojection:
    def test_zero_error(self):
        src = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
        H, _ = cv2.findHomography(src, src)
        proj = cv2.perspectiveTransform(src.reshape(-1,1,2), H).reshape(-1,2)
        assert np.mean(np.linalg.norm(proj - src, axis=1)) < 1e-6

    def test_wrong_homography_has_error(self):
        src = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
        proj = cv2.perspectiveTransform(src.reshape(-1,1,2), np.eye(3,dtype=np.float32)).reshape(-1,2)
        assert np.mean(np.linalg.norm(proj - src*2, axis=1)) > 0.5
