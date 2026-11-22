import cv2
import os

label = "Z"  
folder = f"dataset/{label}"

os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.putText(frame, "Press SPACE to capture", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capture Images", frame)

    key = cv2.waitKey(1)

    
   if key == 32:
        img_path = f"{folder}/{count}.jpg"
        cv2.imwrite(img_path, frame)
        print("Saved:", img_path)
        count += 1


    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()