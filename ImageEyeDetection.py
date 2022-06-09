
from openvino_detectors import OpenvinoFaceDetector, OpenvinoFaceLandmarksFinder
from openvino.inference_engine import IENetwork, IECore


class ImageEyeDetection:
    def __init__(self, multi_face_mode=False):
        self.multi_face_mode = multi_face_mode
        self.face_landmark_finder = OpenvinoFaceLandmarksFinder() # Any hyperparameters?
        self.face_detector = OpenvinoFaceDetector(detection_threshold=0.4)

    def find_big_face(self, faces):
        max_area = 0
        max_index = 0
        for i, (left, top, right, bottom, confidence) in enumerate(faces):
            area = abs(right - left) * abs(bottom - top)
            if area > max_area:
                max_area = area
                max_index = i

        return [faces[max_index]]


    def DetectEyeonImage(self, frame, faces=None):
        face_eye_detect = []

        # 1. Detect the face on image
        if faces is None:
            faces = self.face_detector.detect(frame)

        faces_count = len(faces)

        t = 0

        if faces_count > 1 and not self.multi_face_mode:
            faces = self.find_big_face(faces)
        face = None
        for (left, top, right, bottom, confidence) in faces:
            face_pos = (left, top, right, bottom)
            # print(face_pos)
            # face=frame[top:bottom,left:right]


            # 2. Detect eyes on the detected face
            left_eye_pos, right_eye_pos = self.face_landmark_finder.get_eyes_pos(frame, left, top, right, bottom)
            l_left, l_top, l_right, l_bottom = left_eye_pos
            left_eye = frame[l_top:l_bottom, l_left:l_right]
            r_left, r_top, r_right, r_bottom = right_eye_pos
            right_eye = frame[r_top:r_bottom, r_left:r_right]


        try:
            face_eye_detect.append((left_eye_pos, right_eye_pos,  face_pos))
        except Exception as e:
            t += 1
            print("Can't find a face")
            pass
        return face_eye_detect
