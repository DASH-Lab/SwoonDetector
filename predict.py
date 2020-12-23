import glob
import os
# os.system('python3 detection_module.py')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
# from detection_module import *
from yolov5_test.detection_module import *
from classifier_ns_2.module_2class import *
# from classifier_ensemble.module_ensem_2class import *
# from mmdetection_module.detection_module import *
from itertools import product
import torch
import cv2
import json
import pprint
import numpy as np

import imageio
GET_EVERY = 3
FRAME_RATE = 15
IOU_THRESHOLD = 0.2
IOU_NMS_THRESHOLD = 0.4
CONF_THRESHOLD = 0.22
classifier = None
detector = None



class SwoonTracker:
    def __init__(self, bbox_coord, filename, initial_frame, swoon_conf):
        self.bboxes = [bbox_coord]
        self.class_list = [1]
        self.filenames = [filename]
        self.initial_frame = initial_frame
        self.swoon_confidence = [swoon_conf]
        self.index_list = [initial_frame]
        self.last_bbox = bbox_coord
        self.is_swoon = False
        self.end_frame = initial_frame
        self.swoon_by_end = False
        self.sum_frame = np.array(bbox_coord)
        self.average_frame = np.array(bbox_coord)
        self.swoon_count = 1

    def append(self, bbox_coord, class_result, index, filename, conf):
        if len(bbox_coord) != 0:
            self.last_bbox = bbox_coord
            self.swoon_count += 1
            self.sum_frame += np.array(bbox_coord)
            #print('sumframe', self.sum_frame)
            self.average_frame = self.sum_frame.astype(np.int32) / self.swoon_count
            #print('average frame', self.average_frame)
        self.bboxes.append(bbox_coord)
        self.class_list.append(class_result)
        self.index_list.append(index)
        self.filenames.append(filename)
        self.swoon_confidence.append(conf)

    def __str__(self):
        return "[Tracker Print] \n index_list: {} \n bboxes: {}".format(self.index_list, self.bboxes)
def analysis_tracker(tracker_list, image_list):
    global detector, classifier
    image_list = np.array(image_list)
    output_result = [[] for i in range(len(image_list))]
    output_confidence = [[] for i in range(len(image_list))]
    ELIMINATE_FRAMES = 6
    for idx, tracker in enumerate(tracker_list):

        start_index = final_index = tracker.initial_frame
        swoon_count = 0
        end_index = 0

        for id, (class_, index) in enumerate(zip(tracker.class_list, tracker.index_list)):
            if class_ == 1:
                final_index = index
                swoon_count += 1
                end_index = id
        if final_index - start_index <= FRAME_RATE * 9: # 9초 이하로 쓰러진 상태면 기각
            continue

        swoon_index = tracker.index_list[:end_index+1] #[255, 265, 275, ...]
        swoon_class = tracker.class_list[:end_index+1] # [1, 1, 1, ...]
        swoon_confidence = tracker.swoon_confidence[:end_index+1]

        if sum(swoon_class) < 0.35 * len(swoon_class): # "쓰러짐 구간 중" 쓰러진 사람이 35% 이하일 경우 기각
            continue

        # width or height 가 너무 작은 경우 예외 처리
        init_width = tracker.bboxes[0][2] - tracker.bboxes[0][0]
        init_height = tracker.bboxes[0][3] - tracker.bboxes[0][1]
        if min(init_width, init_height) < 25:
            print("0 box cut")
            continue
        prev_coordinate = tracker.bboxes[0] # 가장 처음 디텍트된 bounding box
        count = 0
      #  print(tracker.bboxes[:len(swoon_class)])
        prev_confidence = tracker.swoon_confidence[0]
        for idx_, (class_, coordinate, conf) in enumerate(zip(swoon_class, tracker.bboxes[:len(swoon_class)], swoon_confidence)): # 빈 bbox가 append 되었으면, 앞의 bbox coordinate으로 값을 채운다.
            if idx_ == 0: continue
            if class_ == 0:
                count += 1
            else:
                if count > 0:
                    diff_coordinate = (np.array(coordinate, dtype=np.float) - np.array(prev_coordinate, dtype=np.float))
                    diff_confidence = conf - prev_confidence
                    for ii, box in enumerate(tracker.bboxes[idx_-count:idx_]):
                        tracker.bboxes[idx_-count+ii] = (np.array(prev_coordinate, dtype=np.float) + ((ii+1) / count) * diff_coordinate).astype(np.int).tolist()
                        tracker.swoon_confidence[idx_-count+ii] = prev_confidence + ((ii+1) / count) * diff_confidence
                    count = 0
                prev_coordinate = coordinate
                prev_confidence = conf
       # print("revise -->",tracker.bboxes[:len(swoon_class)])

        swoon_class_ = [1] * len(swoon_class)
        tracker_list[idx].class_list[:end_index+1] = swoon_class_ # 11100111 --> 11111111

        #Find first swoon
        prev_frame = swoon_index[0] - GET_EVERY + 1


        index_list = list(range(prev_frame, swoon_index[0]))


        def get_first_swoon(index_list): #이진탐색으로 최초 쓰러진 위치를 탐색
            pivot = (len(index_list) - 1) // 2

            out = detector.predict(cv2.imread(image_list[index_list[pivot]]), IOU_NMS_THRESHOLD, CONF_THRESHOLD)
            patch_images = out['img']
            coordinates = out['label']

            output_class = classifier.predict(patch_images)
            swoon_coords = []
            for coord, class_ in zip(coordinates, output_class):  # 한 프레임 안에 쓰러진 사람 좌표 append
                if class_ == 1:  # swoon case
                    swoon_coords.append(coord)
            max_iou = -1
            for swoon_coord in swoon_coords:
                iou = cal_iou(swoon_coord, tracker.bboxes[0])
                if max_iou < iou:
                    max_iou = iou
            if len(index_list) == 1:
                if max_iou < IOU_THRESHOLD:
                    return index_list[0] + 1
                else:
                    return index_list[0]
            else:
                if max_iou < IOU_THRESHOLD:
                    return get_first_swoon(index_list[pivot+1:])
                else:
                    return get_first_swoon(index_list[:pivot+1])

        def get_last_swoon(index_list): # 이진탐색으로 최초 쓰러진 위치를 탐색

            pivot = (len(index_list)-1) // 2
            out = detector.predict(cv2.imread(image_list[index_list[pivot]]), IOU_NMS_THRESHOLD, CONF_THRESHOLD)
            patch_images = out['img']
            coordinates = out['label']
            output_class = classifier.predict(patch_images)
            swoon_coords = []
            for coord, class_ in zip(coordinates, output_class):  # 한 프레임 안에 쓰러진 사람 좌표 append
                if class_ == 1:  # swoon case
                    swoon_coords.append(coord)
            max_iou = -1
            match_coordinate = None
            for swoon_coord in swoon_coords:
                iou = cal_iou(swoon_coord, tracker.bboxes[0])
                if max_iou < iou:
                    max_iou = iou
            if len(index_list):
                if max_iou < IOU_THRESHOLD:
                    return index_list[0] - 1
                else:
                    return index_list[0]
            else:
                if max_iou < IOU_THRESHOLD:
                    get_last_swoon(index_list[:pivot+1])
                else:
                    get_last_swoon(index_list[pivot+1:])

        first_swoon_index = get_first_swoon(index_list) # 첫번째 쓰러진 위치를 가져옴.
        if first_swoon_index == 1:
            first_swoon_index = 0



        last_frame = swoon_index[-1]
        if last_frame > tracker.index_list[-3]: # 쓰러짐이 동영상 끝까지 지속될 경우
            tracker_list[idx].swoon_by_end = True
            last_swoon_index = len(image_list)-1
        else:
            last_next_frame = last_frame + GET_EVERY
            last_index_list = list(range(last_frame+1, last_next_frame))
            last_swoon_index = get_last_swoon(last_index_list)


        swoon_section = list(range(first_swoon_index, last_swoon_index+1))
        first_swoon_coord = tracker.bboxes[0]
        first_swoon_conf = tracker.swoon_confidence[0]
        if first_swoon_index != tracker.index_list[0]:
            out = detector.predict(cv2.imread(image_list[first_swoon_index]), IOU_NMS_THRESHOLD, CONF_THRESHOLD)
            coordinates = out['label']
            confs = out['score']
            max_iou = -1
            real_first_swoon_coord = tracker.bboxes[0]
            real_first_swoon_conf = 0
            for coordinate in coordinates:
                iou = cal_iou(coordinate, first_swoon_coord)
                if max_iou < iou:
                    max_iou = iou
                    real_first_swoon_conf = first_swoon_conf
                    real_first_swoon_coord = coordinate
        else:
            real_first_swoon_coord = first_swoon_coord
            real_first_swoon_conf = first_swoon_conf

        swoon_index_list = tracker.index_list[:end_index+1]
        swoon_boxes = tracker.bboxes[:end_index+1]
        swoon_confs = tracker.swoon_confidence[:end_index+1]
        i = 0
        ccount = 1
        if swoon_section[0] != swoon_index_list[0]:
            ccount = swoon_index_list[0] - swoon_section[0] + 1
        for idx_1, swoon_sec in enumerate(swoon_section):
            #print(swoon_sec, swoon_index_list[i], swoon_index_list[0], swoon_index_list[-1])
            if swoon_index_list[0] > swoon_sec:
                remain_count = swoon_index_list[0] - swoon_sec # 5 4 3 2 1
                diff_ = (np.array(first_swoon_coord, dtype=np.float) - np.array(real_first_swoon_coord, dtype=np.float))
                output_result[swoon_sec].append(np.round(np.array(real_first_swoon_coord) + (((ccount - remain_count) / ccount) * diff_)).astype(np.int).tolist())
                diff_swoon = first_swoon_conf - real_first_swoon_conf
                output_confidence[swoon_sec].append(real_first_swoon_conf + ((ccount - remain_count) / ccount) * diff_swoon)
            elif swoon_index_list[0] <= swoon_sec < swoon_index_list[-1]:
                first_box = swoon_boxes[i]
                next_box = swoon_boxes[i+1]
                first_conf = swoon_confs[i]
                next_conf = swoon_confs[i+1]
               # print(swoon_boxes, next_box)
                diff = (np.array(next_box, dtype=np.float) - np.array(first_box, dtype=np.float))
                output_box = (np.round(np.array(first_box) + (((swoon_sec - swoon_index_list[i])/GET_EVERY) * diff))).astype(np.int).tolist()
                output_result[swoon_sec].append(output_box)

                diff_conf = next_conf - first_conf
                output_confidence[swoon_sec].append(first_conf + ((swoon_sec - swoon_index_list[i]) / GET_EVERY) * diff_conf)
                #print('hell ,,',swoon_sec, swoon_index_list[i], swoon_index_list[0], swoon_index_list[-1])
                if swoon_sec == swoon_index_list[i+1]:
                    i += 1
            else:
                output_result[swoon_sec].append(swoon_boxes[-1])
                output_confidence[swoon_sec].append(swoon_confs[-1])

        if swoon_section[0] > 10: # 쓰러진 시작점이 영상의 초반 부분이 아니면 아래 작업 수행
            for idx_1, swoon_sec in enumerate(swoon_section[:ELIMINATE_FRAMES]): # 처음 몇 프레임은 버리는 프레임
                output_result[swoon_sec].pop()
                output_confidence[swoon_sec].pop()


    return output_result, output_confidence

def cal_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

class classifier_h:
    def __init__(self, ratio_threshold):
        self.ratio_threshold = ratio_threshold

    def predict(self, imgs):
        output_list = []
        for img in imgs:
            h, w, _ = img.shape
            if w / h > self.ratio_threshold:
                output_list.append(1)
            else:
                output_list.append(0)
        return output_list

def main():
    global classifier, detector
    #global CONF_THRESHOLD, IOU_NMS_THRESHOLD
    # print("torch version:", torch.__version__)
    # print("usable cuda:", torch.cuda.is_available())
    # # OUTPUT_FILENAME = './test_iitp_detector_nms_test/t1_res_nms055_high_iou_minus_conf.json'
    test_folder = sys.argv[1]
    #test_folder = '/media/data2/AGC/IITP_Track01_Sample'
    #test_folder = '/media/data2/AGC_system_test/validation_data'
    # print("test folder exist:", os.path.exists(test_folder))
    assert os.path.exists(test_folder)
    # prin = pprint.PrettyPrinter(indent=3)
    # print("test folder len:", os.listdir(test_folder))
    # print("The number of frames to infer:", sum([len(os.listdir(frames)) for frames in glob.glob(os.path.join(test_folder, "*"))]))

    #video_list = sorted(glob.glob(os.path.join(test_folder, "*")))

    # classifier = classifier_(
    #     model_path1="./classifier_ns_2/1114_1110/bg.pt",
    #     model_path2="./classifier_ns_2/1114_1110/falldown.pt",
    #     back_vs_person_padding=False, back_vs_person_normalize=False, normal_vs_falldown_padding=False,
    #     normal_vs_falldown_normalize=True)

    video_list = sorted(glob.glob(os.path.join(test_folder, "*")))
    basepath = os.path.dirname(os.path.realpath(__file__))
    classifier_sub_ = classifier_sub(
        model_path1= os.path.join(basepath, "classifier_ns_2/20201118/efficientnetb0_ns_fit_background_vs_person_test_ES_stage1stage2_back_vs_person_6_97.71505376344086_False_True.pt"),
        model_path2= os.path.join(basepath, "classifier_ns_2/20201211/efficientnetb0_ns_fit_normal_vs_falldown_stage1stage2_normal_vs_falldown_beforeafterjangdae_4_99.27884615384616_False_True.pt"),
        back_vs_person_padding=False, back_vs_person_normalize=True, normal_vs_falldown_padding=False,
        normal_vs_falldown_normalize=True)

   # classifier = classifier_(model_path=os.path.join(basepath, "classifier_ensemble/bestweight_1118"))
    classifier = classifier_h(0.7)
    detector = HumanDetector(os.path.join(basepath, 'yolov5_test/weight/last.pt'))
    OUTPUT_FILENAME = os.path.join(basepath, 't1_res_U0000000302.json')

    output_json = {'annotations':[]}

    for idx, video in enumerate(video_list):
        # image_list = sorted(glob.glob(os.path.join(video, "*.*")))
        image_list = sorted(glob.glob(os.path.join(video, "*.*")))
        tracker_list = []
        start_video = time.time()
        print(video)
        folder_name = video.split("/")[-1]
        #folder_name = video_index[idx]
        flag = False
        for idx, image in enumerate(image_list):

            if idx == 0: continue
            #if idx % GET_EVERY != 0: continue
            if idx % GET_EVERY != 0: continue

            if not flag:
                sub_images = [image_list[len(image_list) // 2 - 15], image_list[len(image_list) // 2 - 7], image_list[len(image_list) // 2 ], image_list[len(image_list) // 2 + 7], image_list[len(image_list) // 2 + 15]]

                for sub in sub_images:
                    sub_frame = cv2.imread(sub)
                    sub_out = detector.predict(sub_frame, IOU_NMS_THRESHOLD, CONF_THRESHOLD)
                    sub_patch_images = sub_out['img']
                    patch_images = [img[:, :, ::-1] for img in sub_patch_images]
                    output_class = classifier_sub_.predict(patch_images)
                    if sum(output_class) > 0:
                        flag = True

            if not flag:
                break

            frame = cv2.imread(image)
            out = detector.predict(frame, IOU_NMS_THRESHOLD, CONF_THRESHOLD)
            patch_images = out['img']
            coordinates = out['label'] # 확인 완료
            confidences = out['score']
            patch_images = [img[:, :, ::-1] for img in patch_images]
            output_class = classifier.predict(patch_images)


            swoon_coords = []
            swoon_confidence = []
            for id, (coord, class_, conf) in enumerate(zip(coordinates, output_class, confidences)): # 한 프레임 안에 쓰러진 사람 좌표 append
                if class_ == 1: # swoon case
                    swoon_coords.append(coord)
                    swoon_confidence.append(conf)


            if len(tracker_list) == 0: # 동영상 속 쓰러진 사람(들)이 처음 발견되면, tracker 생성
                for swoon_coord, swoon_conf in zip(swoon_coords, swoon_confidence):
                    tracker_list.append(SwoonTracker(swoon_coord, image, idx, swoon_conf))
            else:
                if len(swoon_coords) == 0: #tracker가 있지만, 쓰러진 사람이 탐지되지 않을 경우
                    for i, tracker in enumerate(tracker_list):
                        tracker_list[i].append([], 0, idx, image, 0) # 탐지되지 않은 정보를 모든 tracker에 append
                else: # tracker 가 있고, 쓰러진 사람이 탐지될 경우
                    swoon_matching = [False] * len(swoon_coords) # 쓰러짐 좌표가 매칭이 되면 True로 변경
                    tracker_matching = [False] * len(tracker_list)
                    for i, (swoon_coord, swoon_conf) in enumerate(zip(swoon_coords, swoon_confidence)):
                        max_iou = -1
                        max_index = -1
                        for j, tracker in enumerate(tracker_list):
                            # print(tracker.average_frame.tolist(), swoon_coord)
                            get_iou = cal_iou(tracker.bboxes[0], swoon_coord)
                            if max_iou < get_iou and not tracker_matching[j]:
                                max_iou = get_iou
                                max_index = j
                        if max_iou > IOU_THRESHOLD:
                            swoon_matching[i] = True
                            tracker_list[max_index].append(swoon_coord, 1, idx, image, swoon_conf)
                            tracker_matching[max_index] = True
                    for z, track_bool in enumerate(tracker_matching):
                        if not track_bool:
                            tracker_list[z].append([], 0, idx, image, 0)
                    for match_result, swoon, swoon_conf in zip(swoon_matching, swoon_coords, swoon_confidence): # 매칭되지 않은 쓰러진 좌표가 있다면, 그 좌표를 시작점으로 새로운 Tracker 생성
                        if not match_result:
                            tracker_list.append(SwoonTracker(swoon, image, idx, swoon_conf))

        if flag:
            print('Detected')
        else:
            for image_name in image_list:
                file_dict = {
                    'file_name': image_name.split("/")[-1],
                    'box': []
                }
                output_json['annotations'].append(file_dict)
            print('NotDetected')
            continue
        print("time to infer on one video:", time.time() - start_video)
        final_decision, final_decision_confidence = analysis_tracker(tracker_list, image_list)


        for img_out, tracker_out, conf_out in zip(image_list, final_decision, final_decision_confidence):
            if len(tracker_out) == 0:
                file_dict = {
                    'file_name': img_out.split("/")[-1],
                    'box': []
                }
            else:
                file_dict = {
                    'file_name': img_out.split("/")[-1],
                    'box': []
                }
                for cd, cf in zip(tracker_out, conf_out):
                    box_dict = {
                        'position': cd,
                        'confidence_score': str(cf)
                    }
                    file_dict['box'].append(box_dict)
            output_json['annotations'].append(file_dict)

    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(output_json, f)

if __name__ == "__main__":
    main()