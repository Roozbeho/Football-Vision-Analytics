import cv2

def visualize_ball_possession(frame, ball_possession):

    frame_h, frame_w = frame.shape[:2]

    overlay_w, overlay_h = 260, 120
    margin = 20

    x1 = margin
    y1 = frame_h - overlay_h - margin
    x2 = x1 + overlay_w
    y2 = y1 + overlay_h

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (30, 30, 30),
        -1
    )

    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    team1_pos, team2_pos = ball_possession

    cv2.putText(frame, f"Team 1: {team1_pos:.1f}%", (x1 + 15, y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Team 2: {team2_pos:.1f}%", (x1 + 15, y1 + 80), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    return frame