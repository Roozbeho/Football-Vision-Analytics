from ultralytics import YOLO

class CustomYOLO:
    def __init__(self, model_weight_path: str, pose_model_weight_path: str, device: str="cuda"):
        self.yolo = YOLO(model_weight_path).to(device)
        self.yolo_pose = YOLO(pose_model_weight_path).to(device)


    def call_yolo(self, img):

        results = self.yolo(img)
        
        self.image_size = results[0].orig_shape

        return results, None

    def call_yolo_pose(self, img):
        results = self.yolo_pose(img)
        
        return results[0]
    