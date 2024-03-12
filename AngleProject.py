import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()
count = 0
pcount = 0
cv2.namedWindow('Image', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
old_coords = []
standing_start_time = None
sitting_start_time = None
total_standing_time = 0
total_sitting_time = 0

def find_difference(old, new):
    x1, y1 = old
    x2, y2 = new
    diff1 = abs(x1 - x2)
    diff2 = abs(y1 - y2)
    return (diff1 + diff2) / 2

def check_pose(old_coords, top, bottom, left, right, height, thresh=13):
    global standing_start_time, sitting_start_time, total_standing_time, total_sitting_time
    
    if top[1] > height:
        if sitting_start_time:
            sitting_end_time = time.time()
            total_sitting_time += sitting_end_time - sitting_start_time
            sitting_start_time = None
        if standing_start_time is None:
            standing_start_time = time.time()
        return 'Sitting', 0
    
    if len(old_coords) == 0:
        old_coords.append([top, bottom, left, right])
        return None, 0
    else:
        top_old, bottom_old, left_old, right_old = old_coords[0]
        diff1 = find_difference(top_old, top)
        diff2 = find_difference(bottom_old, bottom)
        diff3 = find_difference(left_old, left)
        diff4 = find_difference(right_old, right)
        avg = (diff1 + diff2 + diff3 + diff4) / 4
        if avg > thresh:
            if standing_start_time:
                standing_end_time = time.time()
                total_standing_time += standing_end_time - standing_start_time
                standing_start_time = None
            if sitting_start_time is None:
                sitting_start_time = time.time()
            return 'Moving', avg
        else:
            if sitting_start_time is None:
                sitting_start_time = time.time()
            return 'Standing', avg

while True:
    success, img = cap.read()
    count += 1
    img = detector.find_pose(img, False)
    lm_list = detector.find_position(img, False)
    if len(lm_list) != 0:
        _ = detector.find_angle(img, 12, 14, 16)
        _ = detector.find_angle(img, 11, 13, 15)
        _ = detector.find_angle(img, 14, 12, 24)
        _ = detector.find_angle(img, 13, 11, 23)
        _ = detector.find_angle(img, 12, 24, 26)
        _ = detector.find_angle(img, 11, 23, 25)
        _ = detector.find_angle(img, 24, 26, 28)
        _ = detector.find_angle(img, 23, 25, 27)
        top = detector.find_angle(img, 11, 12, 14, draw=False)
        bottom = detector.find_angle(img, 28, 32, 30, draw=False)
        left = detector.find_angle(img, 16, 18, 20, draw=False)
        right = detector.find_angle(img, 15, 17, 19, draw=False)
        h, w = img.shape[:2]
        
        h_line = int(0.5 * h)
        pose, avg = check_pose(old_coords, top[1], bottom[1], left[1], right[1], h_line)
        if count == 20:
            old_coords = []
            old_coords.append([top[1], bottom[1], left[1], right[1]])
            count = 0

    cv2.line(img, (0, h_line), (w, h_line), (0, 255, 0), 4)
    if pose:
        cv2.rectangle(img, (0, 10), (300, 80), (0, 0, 0), -1)
        cv2.putText(img, 'Action: {}'.format(pose), (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 4)
    
    standing_time_str = f"Standing Time: {int(total_standing_time)}s"
    sitting_time_str = f"Sitting Time: {int(total_sitting_time)}s"
    cv2.putText(img, standing_time_str, (0, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 4)
    cv2.putText(img, sitting_time_str, (0, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 4)

    im = cv2.resize(img, (500, 400))
    cv2.imshow("Image", im)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
