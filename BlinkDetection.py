import cv2
import numpy as np

from ImageEyeDetection import ImageEyeDetection
from openvino.inference_engine import IECore

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)


class EyeStateClassifier():
    def __init__(self):
        self.eye_detector = ImageEyeDetection()  # OpenvinoFaceLandmarksFinder
        self.ie = IECore()

    def Eyes_open_close_model_init(self):
        EYES_OPEN_CLOSE_MODEL = "./models/open-closed-eye.xml"
        net_EOC = self.ie.read_network(model=EYES_OPEN_CLOSE_MODEL,
                                  weights="./models/open-closed-eye.bin")
        exec_net_EOC = self.ie.load_network(net_EOC,
                                       device_name='CPU')
        return net_EOC, exec_net_EOC

    def Eyes_open_close_detection(self, frame, net_EOC, exec_net_EOC):


        # Change data layout from HWC to CHW
        #image = resized.transpose(2, 0, 1)

        dim = (32, 32)
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        alpha = 75 * (32 * 32) / np.sum(image)
        beta = 0

        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        #cv2.imshow('eye', image)
        #cv2.waitKey(FPS_MS)


        input_layer = next(iter(net_EOC.input_info))
        n, c, h, w = net_EOC.input_info[input_layer].input_data.shape

        outputs = exec_net_EOC.infer({input_layer: image})
        outs = next(iter(outputs.values()))

        return outs


    def prediction(self, name, duration):

        net_EOC, exec_net_EOC = self.Eyes_open_close_model_init()

        # write a marked video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frameSize = (int(640), int(480))
        fps1 = 10 #FPS_MS
        new_name = "Annotated_blink_" + name + ".avi"
        new_video = cv2.VideoWriter(new_name, fourcc=fourcc, fps=fps1, apiPreference=0,
                                    frameSize=frameSize)
        print(name+".webm")
        cap = cv2.VideoCapture('./demo/'+ name+".webm")

        TOTAL = 0  # total number of blinks
        COUNTER = 0
        false_prediction = 0 # when both are not same
        num_opened = 0; count_frame = 0
        labels = []; ind_closed = []; blink_ind = []


        #cap = cv2.VideoCapture(0) # web camera

        # Run to the test video
        while True:
            ret, frame = cap.read()

            if not ret:
                #print(ret)
                break

            try:
                # 1. Detect the eye
                out = self.eye_detector.DetectEyeonImage(frame)  # (left_eye_pos, right_eye_pos,  face_pos)
                l_left, l_top, l_right, l_bottom = out[0][0]
                left_eye = frame[l_top:l_bottom, l_left:l_right]

                r_left, r_top, r_right, r_bottom = out[0][1] 
                right_eye = frame[r_top:r_bottom, r_left:r_right]


                # 2. Prediction
                outs = self.Eyes_open_close_detection(left_eye, net_EOC, exec_net_EOC)
                outs1 = self.Eyes_open_close_detection(right_eye, net_EOC, exec_net_EOC)

                labels.append(outs[0][0][0])

                if (outs[0][0][0] > outs[0][1][0]) and (outs1[0][0][0] > outs1[0][1][0]):
                    pred_class = 'closed'
                    ind_closed.append(count_frame)

                    cv2.putText(frame, "closed ", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "Frame : {}".format(count_frame), (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.rectangle(frame, (int(l_left), int(l_top)), (int(l_right), int(l_bottom)), COLOR_RED, 1)
                    cv2.rectangle(frame, (int(r_left), int(r_top)), (int(r_right), int(r_bottom)), COLOR_RED, 1)

                elif (outs[0][0][0] > outs[0][1][0]) and (outs1[0][0][0] < outs1[0][1][0]) or (outs[0][0][0] < outs[0][1][0]) and (outs1[0][0][0] > outs1[0][1][0]):
                    pred_class = 'left or right'
                    false_prediction +=1
                    cv2.rectangle(frame, (int(l_left), int(l_top)), (int(l_right), int(l_bottom)), COLOR_GREEN, 1)
                    cv2.rectangle(frame, (int(r_left), int(r_top)), (int(r_right), int(r_bottom)), COLOR_GREEN, 1)
                else:
                    pred_class = 'opened'
                    num_opened +=1
                    cv2.rectangle(frame, (int(l_left), int(l_top)), (int(l_right), int(l_bottom)), COLOR_BLUE, 1)
                    cv2.rectangle(frame, (int(r_left), int(r_top)), (int(r_right), int(r_bottom)), COLOR_BLUE, 1)


                # 3. Define the blink
                if pred_class == 'closed':
                    COUNTER += 1
                else:
                    # if the eyes were closed for a sufficient number of frames
                    # then increment the total number of blinks
                    if COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        blink_ind.append(count_frame)
                        # reset the eye frame counter
                    COUNTER = 0
                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Frame : {}".format(count_frame), (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            except Exception as e:
                print("Error is")

                pass

            #cv2.imshow('frame', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

            new_video.write(frame)
            count_frame += 1

        cap.release()
        cv2.destroyAllWindows()

        #print('count_frame ', count_frame)

        print("Count frames: %4.2f, Closed: %4.2f, Opened: %4.2f, Closed just one eye: %4.2f" % (count_frame, len(ind_closed), num_opened, false_prediction))
        print(" ")

        # total_frames = fps * duration (fps - frame per second)
        blink_timing = [round(x * duration / count_frame, 2) for x in blink_ind]

        return blink_timing
