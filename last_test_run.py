import cv2
import numpy as np
import pygame

cap = cv2.VideoCapture(1)

first_frame = None

pygame.mixer.init()

sound_playing = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
    if first_frame is None:
        first_frame = gray
        continue
 
    frame_delta = cv2.absdiff(first_frame, gray)
    # Adjust the threshold value here (higher value means less sensitivity)
    thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        
        if cv2.contourArea(contour) < 1000:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Motion Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        motion_detected = True

    
    if motion_detected and not sound_playing:
        pygame.mixer.music.load('alert.wav')
        pygame.mixer.music.play(-1)
        sound_playing = True

    if not motion_detected and sound_playing:
        pygame.mixer.music.stop()
        sound_playing = False

    cv2.imshow("Motion Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
