import cv2
import tensorflow as tf
import numpy as np
from collections import Counter, deque


model = tf.keras.models.load_model("sign_model.h5")


labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


cap = cv2.VideoCapture(0)


history = deque(maxlen=20)
display_label = ""

while True:
    ret, frame = cap.read()

    if not ret:
        break


    frame = cv2.flip(frame, 1)

    
    x1, y1, x2, y2 = 200, 80, 440, 320
    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    img = cv2.resize(roi, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    
    prediction = model.predict(img, verbose=0)[0]

    class_id = np.argmax(prediction)
    confidence = float(prediction[class_id])

    history.append(class_id)

    if len(history) == history.maxlen:
        most_common = Counter(history).most_common(1)[0]

        if most_common[1] >= 16 and confidence >= 0.95:
            display_label = labels[most_common[0]]

    
    cv2.putText(
        frame,
        f"{display_label} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sign Language", frame)

    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()