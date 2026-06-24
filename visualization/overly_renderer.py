import cv2 

def visualize_pitch_points(frame, current_pitch_map, pitch_h, pitch_w):

    overlay_w = 650
    overlay_h = int((pitch_h / pitch_w) * overlay_w)

    mini_pitch = cv2.resize(current_pitch_map, (overlay_w, overlay_h))

    frame_h, frame_w = frame.shape[:2]

    x_offset = frame_w - overlay_w - 20
    y_offset = frame_h - overlay_h - 20
    
    roi = frame[y_offset:y_offset + overlay_h,x_offset:x_offset + overlay_w]
    
    alpha = 0.7  
    beta = 1.0 - alpha
    blended = cv2.addWeighted(mini_pitch, alpha, roi, beta, 0)

    frame[y_offset:y_offset + overlay_h, x_offset:x_offset + overlay_w] = blended

    return frame

