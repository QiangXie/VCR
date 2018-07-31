import os
import cv2
import xml.etree.ElementTree as ET
import shutil
import config
from data.data_augmentation import DataAugmentation
from tqdm import tqdm

FLAGS = config.FLAGS

charset = u'0123456789ABCDEFGHJKLMNPRSTUVWXYZ'
class VinAnnotation(object):
    def __init__(self, vin_code, xmin, ymin, xmax, ymax):
        self.vin_code = vin_code
        self.xmin = xmin 
        self.ymin = ymin 
        self.xmax = xmax 
        self.ymax = ymax 

    def get_boundingbox(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def get_vin_code(self):
        return self.vin_code

class VinAnnotationsReader(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.img_list = list()
        self.xml_list = list()
        self.vin_annotations = list()
        imgs = os.listdir(self.data_path)
        self.img_num = 0
        for img in imgs:
            img_abs_path = os.path.join(self.data_path, img)
            if (os.path.isfile(img_abs_path) and 
                    (os.path.splitext(img_abs_path)[1] == ".jpg" or os.path.splitext(img_abs_path)[1] == ".png")):
                #trans png file to jpg file
                if os.path.splitext(img_abs_path)[1] == ".png":
                    new_img_path = os.path.splitext(img_abs_path)[0] + ".jpg"
                    cv2.imwrite(new_img_path, cv2.imread(img_abs_path))
                    os.remove(img_abs_path)
                    img_abs_path = new_img_path
                xml_abs_path = os.path.join(self.data_path, "Annotations", 
                        os.path.splitext(img)[0]+".xml")
                if os.path.exists(xml_abs_path):
                    tree = ET.parse(xml_abs_path)
                    root = tree.getroot()
                    annotation = root.find('object')
                    vin_code_ = annotation.find('name').text
                    #check vin code and fix it, vin code does not contain IOQ
                    if len(vin_code_) != 17:
                        print("Label length error: {}".format(xml_abs_path))
                        continue
                    vin_code_list = list(vin_code_)
                    for i, c in enumerate(vin_code_list):
                        if c == 'O':
                            vin_code_list[i] = '0'
                        if c == "I":
                            vin_code_list[i] = '1'
                        if c == "Q":
                            vin_code_list[i] = '0'
                        if vin_code_list[i] not in charset:
                            raise ValueError("Error char in label:{}".format(xml_abs_path))
                    vin_code_ = "".join(vin_code_list)


                    xmin_ = int(annotation.find('bndbox').find('xmin').text)
                    ymin_ = int(annotation.find('bndbox').find('ymin').text)
                    xmax_ = int(annotation.find('bndbox').find('xmax').text)
                    ymax_ = int(annotation.find('bndbox').find('ymax').text)
                    vin_annotation = VinAnnotation(vin_code_, 
                            xmin_,
                            ymin_, 
                            xmax_, 
                            ymax_)
                    self.vin_annotations.append(vin_annotation)
                    self.img_list.append(img_abs_path)
                    self.xml_list.append(xml_abs_path)
                    self.img_num += 1

    def print_annotation(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for i in range(self.img_num):
            img_cv = cv2.imread(self.img_list[i]) 
            xmin, ymin, xmax, ymax = self.vin_annotations[i].get_boundingbox()
            vin_code = self.vin_annotations[i].get_vin_code()
            cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(img_cv, vin_code, (ymin, xmin), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)
            save_path = os.path.join(save_dir, os.path.split(self.img_list[i])[1])
            cv2.imwrite(save_path, img_cv)

    def crop(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        else:
            print("Remove old crop imgs:{}".format(save_dir))
            shutil.rmtree(save_dir)
            os.mkdir(save_dir)

        for i in tqdm(range(self.img_num), ascii=True, desc="Crop Images"):
            img_cv = cv2.imread(self.img_list[i])
            xmin, ymin, xmax, ymax = self.vin_annotations[i].get_boundingbox()
            vin_code = self.vin_annotations[i].get_vin_code()
            vin_img_cv = img_cv[ymin:ymax, xmin:xmax]
            save_name = "{}_{}.jpg".format(i, vin_code)
            save_path = os.path.join(save_dir, save_name)
            cv2.imwrite(save_path, vin_img_cv)

    def data_augmentation(self, data_dir, save_dir, multiple=10):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        imgs = os.listdir(save_dir) 
        dict_counters = dict()
        for img in imgs:
            label = os.path.splitext(img)[0].split('_')[1]
            if label not in dict_counters.keys():
                dict_counters[label] = 1
            else:
                dict_counters[label] += 1

        ori_imgs = os.listdir(data_dir)
        counter = len(imgs)
        data_augmentator = DataAugmentation()
        for ori_img in tqdm(ori_imgs, ascii=True, desc="Data Augmentation"):
            label = os.path.splitext(ori_img)[0].split('_')[1]
            if label not in dict_counters.keys() or dict_counters[label] < 1+multiple:
                old_path = os.path.join(data_dir, ori_img)
                out_pathes = list()
                if label not in dict_counters.keys():
                    new_ori_path = os.path.join(save_dir, str(counter)+"_"+label+".jpg")
                    shutil.copy(old_path, new_ori_path)
                    counter += 1
                    for i in range(multiple):
                        new_path = os.path.join(save_dir, str(counter)+"_"+label+".jpg")
                        out_pathes.append(new_path)
                        data_augmentator.generate(old_path, out_pathes)
                        counter += 1
                else:
                    for i in range(multiple+1-dict_counters[label]):
                        counter += 1
                        new_path = os.path.join(save_dir, str(counter)+"_"+label+".jpg")
                        out_pathes.append(new_path)
                        data_augmentator.generate(old_path, out_pathes)

        def get_img_list(self):
            return self.img_list

    def get_vin_codes(self):
        return [self.vin_annotations[i].get_vin_code() for i in range(self.img_num)]

    def get_boundingboxs(self):
        return [self.vin_annotations[i].get_boundingbox() for i in range(self.img_num)]
