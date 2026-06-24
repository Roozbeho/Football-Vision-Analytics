
OBJECT_MODEL_PATH = "runs/obj_detector/weights/best.pt"
KEYPOINT_MODEL_PATH = "runs/keypoint_detection/weights/best.pt"
VIDEO_PATH = "data/input_video.mp4"
OUTPUT_PATH = "output.mp4"
PITCH_IMG_PATH = "data/top-view-of-green-football-pitch-or-soccer-field-vector.jpg"


DEVICE = "cuda"  

CONFIDENCE_THRESHOLD = 0.5
KEYPOINT_CONF_THRESHOLD = 0.5

KMEANS_N_CLUSTERS = 2

MAX_BALL_DISTANCE = 70 

PLAYER_COLORS = [(255, 0, 0), (0, 0, 255)]  
GOALKEEPER_COLOR = (255, 126, 255)
REFEREE_COLOR = (0, 0, 0)
BALL_COLOR = (0, 255, 255)


FRAME_SKIP = 1

VIDEO_CODEC = "mp4v"
OUTPUT_FPS = None  
