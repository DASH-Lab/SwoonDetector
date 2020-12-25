import glob
import os
import munch
import xmltodict

def make_txt(txt_save_path, xml_folder):
    paths = glob.glob(os.path.join(xml_folder, "*.xml"))
    #paths = [path for path in paths if "swoon" in path]
    for xml_path in paths:
        print(xml_path)
        txt_name = xml_path.split("/")[-1].split(".")[0]+".txt"
        # print('hello', mp4_name)
        file = open(xml_path)
        # if os.path.exists(os.path.join(to_save_path, txt_name)):
        #     continue
        save_file = open(os.path.join(txt_save_path, txt_name), 'w')
        doc = xmltodict.parse(file.read())
        doc = munch.munchify(doc)
        try:
            annot = doc.annotation.object
        except:
            continue
        if isinstance(annot, munch.Munch):
            annot = [annot]

        for anot in annot:
            coord = anot.bndbox
            label = anot.name
            center_x = ((float(coord.xmax) + float(coord.xmin)) / 2) / 1920.0
            center_y = ((float(coord.ymax) + float(coord.ymin)) / 2) / 1080.0
            width = (float(coord.xmax) - float(coord.xmin)) / 1920.0
            height = (float(coord.ymax) - float(coord.ymin)) / 1080.0
            save_file.write('0 {} {} {} {}\n'.format(center_x, center_y, width, height))
        save_file.close()


#os.makedirs('/media/data2/AGC/WorkDirectory/complete_data/txt_labels', exist_ok=True)
make_txt('/media/data2/AGC/WorkDirectory2/train/zips/labels', '/media/data2/AGC/WorkDirectory2/train/zips/xmls')
