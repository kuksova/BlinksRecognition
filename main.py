import sys
from BlinkDetection import EyeStateClassifier
from load_data_csv import load_data_csv
from metrics import binary_classification_metrics


def video(name, duration):

    blink_detector = EyeStateClassifier()

    '''Setup'''
    blink_detector.EYE_AR_CONSEC_FRAMES = 1 # hyperparameter
    eps = 0.9 # hyperparameter
    # flexible size of ROI eye

    '''Load source data'''
    path = './demo/'+ name +'.csv'
    truth_timing = load_data_csv(path)

    if duration == [] or truth_timing == []:
        print("Missed info about the video duration or blinks")
        return [0.0, 0.0, 0.0, 0.0]

    '''Predict blinks'''
    blink_timing = blink_detector.prediction(name, duration)
    print('True timing ', truth_timing)
    print('Predict timing ', blink_timing)
    print("")
    print("Counts Found Blinks: ", len(blink_timing), "True Blinks: ", len(truth_timing))

    '''Calculate accuracy'''
    precision, recall, f1 = binary_classification_metrics(blink_timing, truth_timing, eps)
    print("")
    print("Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (precision, recall, f1))



if __name__ == '__main__':
    # input_video from the command line
    # input WEBcam
    # input saved video

    name = sys.argv[1]
    duration = int(sys.argv[2])
    #name = "video1"
    #duration = 18
    video(name, duration)




