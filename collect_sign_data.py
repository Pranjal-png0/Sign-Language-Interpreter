import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

csv_file = 'sign_language_data.csv'

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f'x{i}', f'y{i}', f'z{i}']
        header += ['label']
        csv_writer.writerow(header)

cap = cv2.VideoCapture(0)

print("Press a key for the sign (e.g. 'a', 'b', 'c'). Press 'q' to quit.")

while True:
    success, image = cap.read()
    if not success:
        print("No frame captured, skipping...")
        continue

    
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    result = hands.process(rgb_image)

    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    
    cv2.imshow('Collect Sign Data', image)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if result.multi_hand_landmarks and key != 255 and key != ord('q'):
        label = chr(key) 

        
        landmarks = []
        hand_landmarks = result.multi_hand_landmarks[0]
        for lm in hand_landmarks.landmark:
            landmarks += [lm.x, lm.y, lm.z]

        
        with open(csv_file, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(landmarks + [label])

        print(f"Saved data for sign '{label}'")


cap.release()
cv2.destroyAllWindows()