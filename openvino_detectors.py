from __future__ import print_function

import os
from time import time
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore




def get_head_by_face(frame, left, top, right, bottom):
    frame_height, frame_width = frame.shape[:-1]
    height = bottom - top
    width = right - left
    if height > width:
        diff = height - width
        left_diff = int(diff / 2)
        right_diff = diff - left_diff
        if left >= left_diff and right + right_diff < frame_width:
            head = frame[top:bottom, (left - left_diff):(right + right_diff)]
        elif left < left_diff:
            head = frame[top:bottom, 0:height]
        else:
            head = frame[top:bottom, (frame_width - height):frame_width]
    elif width > height:
        diff = width - height
        top_diff = int(diff / 2)
        bottom_diff = diff - top_diff
        if top >= top_diff and bottom + bottom_diff < frame_height:
            head = frame[(top - top_diff):(bottom + bottom_diff), left:right]
        elif top < top_diff:
            head = frame[0:width, left:right]
        else:
            head = frame[(frame_height - width):frame_height, left:right]
    else:
        head = frame[top:bottom, left:right]
    return head


FACE_DETECTOR = "__FACE_DETECTOR__"

FACE_LANDMARK = "__FACE_LANDMARK__"


ie = IECore()

OPENVINO_DETECTORS = {
    FACE_DETECTOR: ie.read_network(
        model=("./models/face-detection-0202.xml"),
    ),
    FACE_LANDMARK: ie.read_network(
        model=("./models/facial-landmarks-35-adas-0002.xml"),
    )

}


FACE_DETECTOR_PLUGIN = None
FACE_LANDMARK_PLUGIN = None




def get_face_detector_plugin():
    global FACE_DETECTOR_PLUGIN
    if FACE_DETECTOR_PLUGIN is None:
        FACE_DETECTOR_PLUGIN = ie.load_network(network=OPENVINO_DETECTORS[FACE_DETECTOR], device_name='CPU')
    return FACE_DETECTOR_PLUGIN

def get_face_landmark_plugin():
    global FACE_LANDMARK_PLUGIN
    if FACE_LANDMARK_PLUGIN is None:
        FACE_LANDMARK_PLUGIN = ie.load_network(network=OPENVINO_DETECTORS[FACE_LANDMARK], device_name='CPU')
    return FACE_LANDMARK_PLUGIN




OPENVINO_DETECTORS_PLUGIN_LOADED = {
    FACE_DETECTOR: get_face_detector_plugin,
    FACE_LANDMARK: get_face_landmark_plugin
}


class OpenvinoDetector:
    def __init__(self, detector_xml, detection_threshold=0):

        # Plugin initialization for specified device and load extensions library if specified
        # plugin = IEPlugin(device="CPU")
        # plugin.add_cpu_extension(cpu_lib)

        # Read detector IR
        # detector_bin = os.path.splitext(detector_xml)[0] + ".bin"
        detector_net = OPENVINO_DETECTORS[detector_xml]

        #self.d_in = next(iter(detector_net.inputs))
        self.d_in = next(iter(detector_net.input_info))
        self.d_out = next(iter(detector_net.outputs))
        detector_net.batch_size = 1

        # Read and pre-process input images
        self.d_n, self.d_c, self.d_h, self.d_w = detector_net.input_info[self.d_in].input_data.shape
        self.d_images = np.ndarray(shape=(self.d_n, self.d_c, self.d_h, self.d_w))

        # Loading models to the plugin
        self.d_exec_net = OPENVINO_DETECTORS_PLUGIN_LOADED[detector_xml]()

        self.detection_threshold = detection_threshold

    def get_detections(self, frame):

        height, width = frame.shape[:-1]
        if height * self.d_w > self.d_h * width:
            new_width = self.d_w * height / self.d_h
            new_height = height
            border_size = int((new_width - width) / 2)
            frame = cv2.copyMakeBorder(frame, top=0, bottom=0, left=border_size, right=border_size,
                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif height * self.d_w < self.d_h * width:
            new_width = width
            new_height = self.d_h * width / self.d_w
            border_size = int((new_height - height) / 2)
            frame = cv2.copyMakeBorder(frame, top=border_size, bottom=border_size, left=0, right=0,
                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            new_width = width
            new_height = height

        if (new_width, new_height) != (self.d_w, self.d_h):
            d_frame = cv2.resize(frame, (self.d_w, self.d_h))
        else:
            d_frame = frame

        # Change data layout from HWC to CHW
        self.d_images[0] = d_frame.transpose((2, 0, 1))

        d_res = self.d_exec_net.infer(inputs={self.d_in: self.d_images})[self.d_out]
        # d_res = None
        return d_res, new_height, new_width

    def convert_detections(self, det, height, width, new_height, new_width):
        left, top, right, bottom = det
        left = max(0, int(left * new_width - (new_width - width) / 2))
        right = min(int(right * new_width - (new_width - width) / 2), width - 1)

        top = max(0, int(top * new_height - (new_height - height) / 2))
        bottom = min(int(bottom * new_height - (new_height - height) / 2), height - 1)

        return left, top, right, bottom


class OpenvinoFaceDetector(OpenvinoDetector):
    def __init__(self, detection_threshold):
        super().__init__(
            detector_xml=FACE_DETECTOR,
            detection_threshold=detection_threshold)
        self.frames = 0
        self.total_time = 0.0

    def detect(self, frame):

        start = time()
        height, width = frame.shape[:-1]
        detections_raw, new_height, new_width = self.get_detections(frame)
        detections = detections_raw[0][0]
        result = []
        for _, _, confidence, left, top, right, bottom in detections:
            if confidence > self.detection_threshold:
                left, top, right, bottom = self.convert_detections((left, top, right, bottom), height, width,
                                                                   new_height, new_width)
                result.append((left, top, right, bottom, float(confidence)))
        self.total_time += time() - start
        self.frames += 1
        return result


class OpenvinoFaceLandmarksFinder(OpenvinoDetector):
    def __init__(self):
        super().__init__(
            detector_xml=FACE_LANDMARK,
            detection_threshold=0
        )
        self.frames = 0
        self.total_time = 0.0

    def get_landmarks(self, frame, left, top, right, bottom):
        # this is coordinates of faces
        face = cv2.resize(frame[top:bottom, left:right], (self.d_w, self.d_h))
        self.d_images[0] = face.transpose((2, 0, 1))
        l_res = np.squeeze(self.d_exec_net.infer(inputs={self.d_in: self.d_images})[self.d_out])
        for i in range(70):
            if i % 2 == 0:
                l_res[i] = left + (right - left) * l_res[i]
            else:
                l_res[i] = top + (bottom - top) * l_res[i]
        return l_res

    def get_eyes_pos(self, frame, left, top, right, bottom):
        start = time()
        landmarks = self.get_landmarks(frame, left, top, right, bottom)


        d_x = landmarks[2] - landmarks[0]
        d_y = landmarks[3] - landmarks[1]
        eye_size = np.math.sqrt(d_x ** 2 + d_y ** 2) * 2.0 #1.75 #0.75
        left_eye_x = (landmarks[2]-10 + landmarks[0]) / 2
        left_eye_y = (landmarks[3] + landmarks[1]) / 2
        right_eye_x = (landmarks[6]+10 + landmarks[4]) / 2
        right_eye_y = (landmarks[7] + landmarks[5]) / 2

        left_eye_pos = int(left_eye_x - eye_size / 2), int(left_eye_y - eye_size / 2), \
                       int(left_eye_x + eye_size / 4), int(left_eye_y + eye_size / 4)
        right_eye_pos = int(right_eye_x - eye_size / 4), int(right_eye_y - eye_size / 2), \
                        int(right_eye_x + eye_size / 2), int(right_eye_y + eye_size / 4)



        # compute the eye size
        #left_eye = frame[landmarks[2]:landmarks[0], landmarks[3]:landmarks[1]]
        center1_1 = (int(left_eye_x), int(left_eye_y)) # left eye
        center2_1 = (int(right_eye_x), int(right_eye_y)) # right eye

        center1 = (int(landmarks[2]), int(landmarks[3]))  # left eye
        center2 = (int(landmarks[0]), int(landmarks[1]))  # left eye
        center3 = (int(landmarks[4]), int(landmarks[5]))  # right eye
        center4 = (int(landmarks[6]), int(landmarks[7]))  # right eye

        center1_1_bottom = (int(left_eye_x), int(left_eye_pos[1]))
        center1_1_up = (int(left_eye_x), int(left_eye_pos[3]))
        center1_1_left= (int(left_eye_pos[0]), int(left_eye_y))
        center1_1_right = (int(left_eye_pos[2]), int(left_eye_y))

        center2_1_bottom = (int(right_eye_x), int(right_eye_pos[1]))
        center2_1_up = (int(right_eye_x), int(right_eye_pos[3]))
        center2_1_left= (int(right_eye_pos[0]), int(right_eye_y))
        center2_1_right = (int(right_eye_pos[2]), int(right_eye_y))


        self.total_time += time() - start
        self.frames += 1
        return left_eye_pos, right_eye_pos


