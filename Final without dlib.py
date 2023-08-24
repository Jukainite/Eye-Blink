import cv2
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model
import utils, math
import mediapipe as mp

IMG_SIZE = (64, 56)
B_SIZE = (34, 26)
margin = 95
class_labels = ['center', 'left', 'right']
# Left eyes indices
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
# right eyes indices
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]


font_letter = cv2.FONT_HERSHEY_PLAIN
model = load_model('models/gazev3.1.h5')
# model_b = load_model('models/blinkdetection.h5')
model_b = load_model('models/model.h5')
map_face_mesh = mp.solutions.face_mesh

# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord



def detect_gaze(eye_img):
    pred_l = model.predict(eye_img)
    accuracy = int(np.array(pred_l).max() * 100)
    gaze = class_labels[np.argmax(pred_l)]
    return gaze


def detect_blink(eye_img):
    pred_B = model_b.predict(eye_img)
    status = pred_B[0][0]
    status = status * 100
    status = round(status, 3)
    return status


def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


# main
cap = cv2.VideoCapture(0)
# pattern = []
# frames = 10
# pattern_length = 0
blinked = 0
left_winked =0
right_winked=0
blinking_detected = True
frames_to_blink = 6
blinking_frames = 0
with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        output = np.zeros((900, 820, 3), dtype="uint8")
        ret, img = cap.read()
        img = cv2.flip(img, flipCode=1)
        h, w = (112, 128)
        if not ret:
          break
        frame_height, frame_width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # faces = detector(gray)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(img, results, False)

            cv2.polylines(img, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                         cv2.LINE_AA)
            cv2.polylines(img, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                         cv2.LINE_AA)


            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            right_coords =np.array(right_coords)
            left_coords = np.array(left_coords)

            # ~~~~~~~~~~~~~~~~~56,64 EYE IMAGE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            eye_img_l, eye_rect_l = crop_eye(gray, eye_points=left_coords)
            eye_img_r, eye_rect_r = crop_eye(gray, eye_points=right_coords)
            # ~~~~~~~~~~~~~~~~FOR THE EYE FINAL_WINDOW~~~~~~~~~~~~~~~~~~~~~~#
            eye_img_l_view = cv2.resize(eye_img_l, dsize=(128, 112))
            eye_img_l_view = cv2.cvtColor(eye_img_l_view, cv2.COLOR_BGR2RGB)
            eye_img_r_view = cv2.resize(eye_img_r, dsize=(128, 112))
            eye_img_r_view = cv2.cvtColor(eye_img_r_view, cv2.COLOR_BGR2RGB)
            # ~~~~~~~~~~~~~~~~~FOR THE BLINK DETECTION~~~~~~~~~~~~~~~~~~~~~~~
            eye_blink_left = cv2.resize(eye_img_l.copy(), B_SIZE)
            eye_blink_right = cv2.resize(eye_img_r.copy(), B_SIZE)
            eye_blink_left_i = eye_blink_left.reshape((1, B_SIZE[1], B_SIZE[0], 1)).astype(np.float32) / 255.
            eye_blink_right_i = eye_blink_right.reshape((1, B_SIZE[1], B_SIZE[0], 1)).astype(np.float32) / 255.
            # ~~~~~~~~~~~~~~~~FOR THE GAZE DETECTIOM~~~~~~~~~~~~~~~~~~~~~~~~#
            eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
            eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
            eye_input_g_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
            eye_input_g_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

            # ~~~~~~~~~~~~~~~~~~PREDICTION PROCESS~~~~~~~~~~~~~~~~~~~~~~~~~~#

            status_l = detect_blink(eye_blink_left_i)
            status_r = detect_blink(eye_blink_right_i)
            gaze_l = detect_gaze(eye_input_g_l)
            gaze_r = detect_gaze(eye_input_g_r)
            if gaze_r or gaze_l == class_labels[2]:
                blinking_frames += 1

            elif gaze_r or gaze_l == class_labels[1]:
                blinking_frames += 1


            elif status_l < 0.1 and status_r < 0.1:
                blinking_frames += 1

            else:
                blinking_frames = 0
            # ~~~~~~~~~~~~~~~~~~~~~~~FINAL_WINDOWS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

            output = cv2.line(output, (400, 200), (400, 0), (0, 255, 0), thickness=2)
            cv2.putText(output, "LEFT EYE GAZE", (10, 180), font_letter, 1, (255, 255, 51), 1)
            cv2.putText(output, "LEFT EYE OPENING %", (200, 180), font_letter, 1, (255, 255, 51), 1)
            cv2.putText(output, "RIGHT EYE GAZE", (440, 180), font_letter, 1, (255, 255, 51), 1)
            cv2.putText(output, "RIGHT EYE OPENING %", (621, 180), font_letter, 1, (255, 255, 51), 1)

            if (status_l < 10 or status_r <10)and not blinking_detected:
                blinked += 1
                blinking_detected = True

            if blinking_detected:
                cv2.putText(output, "---BLINKING----", (250, 300), font_letter, 2, (153, 153, 255),
                            2)

            if blinking_detected and status_l >= 10 and status_r >10:
                blinking_detected = False

            cv2.putText(output, f"Blinked = {blinked}", (250, 250), font_letter, 2, (153, 153, 255), 1)
            output[0:112, 0:128] = eye_img_l_view
            cv2.putText(output, gaze_l, (30, 150), font_letter, 2, (0, 255, 0), 2)
            output[0:112, margin + w:(margin + w) + w] = eye_img_l_view
            cv2.putText(output, (str(status_l) + "%"), ((margin + w), 150), font_letter, 2, (0, 0, 255), 2)
            output[0:112, 2 * margin + 2 * w:(2 * margin + 2 * w) + w] = eye_img_r_view
            cv2.putText(output, gaze_r, ((2 * margin + 2 * w) + 30, 150), font_letter, 2, (0, 0, 255), 2)
            output[0:112, 3 * margin + 3 * w:(3 * margin + 3 * w) + w] = eye_img_r_view
            cv2.putText(output, (str(status_r) + "%"), ((3 * margin + 3 * w), 150), font_letter, 2, (0, 0, 255), 2)
            output[235 + 100:715 + 100, 80:720] = img

            cv2.imshow('result', output)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # print(pattern)
