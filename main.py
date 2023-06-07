import cv2
import mediapipe as mp

mp_obj = mp.solutions.objectron
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

objectron = mp_obj.Objectron(
    static_image_mode=False,
    max_num_objects=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    model_name='Cup'
)

while cap.isOpened():
    success, image = cap.read()

    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = objectron.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_draw.draw_landmarks(
                image,
                detected_object.landmarks_2d,
                mp_obj.BOX_CONNECTIONS
            )

            mp_draw.draw_axis(
                image,
                detected_object.rotation,
                detected_object.translation
            )

    cv2.imshow('MediaPipe Objectron', cv2.flip(image, 1))
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
