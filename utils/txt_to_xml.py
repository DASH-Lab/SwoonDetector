from imageio import mimread
import numpy as np
import cv2
import glob
import os
import xml.etree.cElementTree as ET
import lxml.etree as etree
import json
import matplotlib.pyplot as plt
from YoloDetector import human_detector_2
import shutil
#type_lis = ['assault', 'burglary', 'datefight', 'drunken', 'dump', 'fight', 'kidnap', 'robbery', 'swoon', 'trespass', 'vandalism', 'wander']

def txt_to_xml(label, image_saved_path, xml_saved_path, img_filepath):

    f_list = label

    annotation = 'annotation'
    folder = 'folder'
    filename = 'filename'
    path = 'path'
    source = 'source'
    database = 'database'
    size = 'size'
    width = 'width'
    height = 'height'
    depth = 'depth'
    segmented = 'segmented'
    object = 'object'
    name = 'name'
    pose = 'pose'
    truncated = 'truncated'
    difficult = 'difficult'
    bndbox = 'bndbox'
    xmin = 'xmin'
    ymin = 'ymin'
    xmax = 'xmax'
    ymax = 'ymax'

    anomaly_type = "swoon"
    folder_name = "user_generate"

    root_ = ET.Element(annotation)
    folder_ = ET.SubElement(root_, folder)
    folder_.text = anomaly_type
    filename_ = ET.SubElement(root_, filename)
    filename_.text = img_filepath
    path_ = ET.SubElement(root_, path)
    saved_path = "T:" +image_saved_path.replace("/","\\")

    path_.text = saved_path + "\\" + img_filepath
    source_ = ET.SubElement(root_, source)
    database_ = ET.SubElement(source_, database)
    database_.text = 'Unknown'
    size_ = ET.SubElement(root_, size)
    segmented_ = ET.SubElement(root_, segmented)
    segmented_.text = '0'
    width_ = ET.SubElement(size_, width)
    width_.text = '1920'
    height_ = ET.SubElement(size_, height)
    height_.text = '1080'
    depth_ = ET.SubElement(size_, depth)
    depth_.text = '3'

    for line in f_list:
        line = line.replace('\n', '')
        label = line.split(' ')

        object_ = ET.SubElement(root_, object)
        name_ = ET.SubElement(object_, name)
        name_.text = '0'
        pose_ = ET.SubElement(object_, pose)
        pose_.text = 'Unspecified'
        truncated_ = ET.SubElement(object_, truncated)
        truncated_.text = '0'
        difficult_ = ET.SubElement(object_, difficult)
        difficult_.text = '0'
        bndbox_ = ET.SubElement(object_, bndbox)
        xmin_ = ET.SubElement(bndbox_, xmin)
        xmin_.text = str(round(((float(label[1]) - float(label[3]) / 2))*1920))
        #xmin_.text = str(label[1])
        ymin_ = ET.SubElement(bndbox_, ymin)
        ymin_.text = str(round(((float(label[2]) - float(label[4]) / 2))*1080))
        #ymin_.text = str(label[2])
        xmax_ = ET.SubElement(bndbox_, xmax)
        xmax_.text = str(round(((float(label[1]) + float(label[3]) / 2))*1920))
        #xmax_.text = str(label[3])
        ymax_ = ET.SubElement(bndbox_, ymax)
        ymax_.text = str(round(((float(label[2]) + float(label[4]) / 2))*1080))
        #ymax_.text = str(label[4])
    tree = ET.ElementTree(root_)

    tree.write(os.path.join(xml_saved_path, img_filepath.split(".")[0] + ".xml"))
    pretty = etree.parse(os.path.join(xml_saved_path, img_filepath.split(".")[0] + ".xml"))
    pret = etree.tostring(pretty, pretty_print=True)

    #print(pret)

    f = open(os.path.join(xml_saved_path, img_filepath.split(".")[0] + ".xml"), 'wb')
   # print(os.path.join(xml_saved_path, img_filepath.split(".")[0] + ".xml"))

    f.write(pret)
    f.close()

def mp4_to_xml(mp4_path, human_detector, extract_every=10, save_path='./'):
    #../gopro_4.mp4
    assert os.path.exists(mp4_path), 'File Not Found Error'
    print('hello')
    base_filename = mp4_path.split("/")[-1].split(".")[0]
    image_path = os.path.join(save_path, 'image')
    label_path = os.path.join(save_path, 'label')
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    cap = cv2.VideoCapture(mp4_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = list(range(0, length, extract_every))[10:]

    for idx, frame_no in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        if not ret or frame is None:
            continue
        resized_frame = cv2.resize(frame, (1920, 1080))
        ret = human_detector.predict(resized_frame)

        to_save_image_name = base_filename +"_"+str(frame_no).zfill(4) +".jpg"


        imgs = ret['img']
        coords = ret['label']
        print(coords)
        if len(imgs) == 0 or len(coords) == 0:
            continue

        cv2.imwrite(os.path.join(image_path, to_save_image_name), resized_frame)


        yolo_label = []
        for coord in coords:
            crd = ["0"]
            lis = list(map(str, coord))
            crd.extend(lis)
            st = " ".join(crd)
            yolo_label.append(st)
        print(os.path.join(image_path, to_save_image_name))
        txt_to_xml(yolo_label, image_path, label_path, to_save_image_name)


if __name__ == "__main__":
    # to_save_path = '/media/data2/AGC/iitp_validation'
    # os.makedirs(os.path.join(to_save_path, "images"), exist_ok=True)
    # os.makedirs(os.path.join(to_save_path, "labels"), exist_ok=True)
    #
    # video_frame_folder = '/media/data2/AGC/IITP_Track01_Sample'
    # video_label_folder = '/media/data2/AGC/Track1_Result_20201006'
    #
    # folders = os.listdir(video_frame_folder)
    #
    # for folder in folders:
    #     video_paths = sorted(glob.glob(os.path.join(video_frame_folder, folder, "*.jpg")))
    #     label_paths = sorted(glob.glob(os.path.join(video_label_folder, folder, "*.json")))
    #     to_extract = np.arange(0, len(video_paths), len(video_paths)/10)[1:].astype(np.int)
    #     video_paths = np.array(video_paths)[to_extract]
    #     label_paths = np.array(label_paths)[to_extract]
    #
    #     for frame, label in zip(video_paths, label_paths):
    #         assert frame.split("/")[-1].split(".")[0] == label.split("/")[-1].split(".")[0]
    #         file = open(label, 'rb')
    #         json_file = json.load(file)
    #         shutil.copy(frame, os.path.join(to_save_path, "images"))
    #         try:
    #             coord = json_file['box']
    #             print('hello')
    #             coord.insert(0, 0)
    #             coord = list(map(str, coord))
    #             str_coord = " ".join(coord)
    #             coord = [str_coord]
    #             txt_to_xml(coord, os.path.join(to_save_path, "images"), os.path.join(to_save_path, "labels"), frame.split("/")[-1].split(".")[0]+".jpg")
    #         except KeyError:
    #             continue

    txt_filepath = '/media/data2/AGC/NightData/NightDataCut/labels'
    xml_saved_folder = '/media/data2/AGC/NightData/NightDataCut/xmls_reconstruction'
    txts = glob.glob(os.path.join(txt_filepath, '*.txt'))
    for txt in txts:
        print(txt)
        f = open(txt, 'r')
        readlines = f.readlines()

        txt_to_xml(readlines, '/media/data2/AGC/NightData/NightDataCut/images', '/media/data2/AGC/NightData/NightDataCut/xmls_reconstruction', txt.split("/")[-1].split(".")[0]+".jpg")


