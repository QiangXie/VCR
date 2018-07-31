from skimage import io,util,exposure,color,transform
import os
import cv2
import numpy as np
from multiprocessing import Pool

class DataAugmentation(object):
    def __init__(self, gamma_min = 1.0, gamma_max = 1.6, 
            noise_type=["gaussian", "s&p", "salt", "pepper"],
            noise_type_p=[0.25, 0.25, 0.25, 0.25],
            rotation_angle_min = -1.5, rotation_angle_max = 1.5,
            grayscale_p = [0.05, 0.95]):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.noise_type = noise_type
        self.noise_type_p = noise_type_p
        self.rotation_angle_min = rotation_angle_min
        self.rotation_angle_max = rotation_angle_max
        self.grayscale_p = grayscale_p

    def single_generate(self, arg):
        im = arg[0]    
        out_path = arg[1]
        im_gamma = self.gamma(im)
        im_transform = self.transform(im_gamma)
        im_noise = self.random_noise(im_transform)
        im_gray = self.gray(im_noise)
        io.imsave(out_path, im_gray)

    def generate(self, input_path, output_pathes):
        im = io.imread(input_path)
        args = []
        for i in range(len(output_pathes)):
            args.append([im, output_pathes[i]])
        p = Pool(len(output_pathes))
        p.map(self.single_generate, args)
        p.close()
        

    def random_noise(self, im):
        noise_t = np.random.choice(self.noise_type, 1, p=self.noise_type_p)
        im_ = util.random_noise(im, mode=str(noise_t[0]))
        return im_

    def gamma(self, im):
        gamma_value = np.random.uniform(low=self.gamma_min, high=self.gamma_max)
        im_ = exposure.adjust_gamma(im, gamma=gamma_value)
        return im_

    def gray(self, im):
        gray_list = [True,False]
        gray_p = np.random.choice(gray_list, 1, self.grayscale_p)
        if gray_p:
            return color.rgb2gray(im)
        else:
            return im

    def transform(self, im):
        rotation_angle = np.random.uniform(self.rotation_angle_min, self.rotation_angle_max)
        im_ = transform.rotate(im, rotation_angle) 
        return im_



if __name__ == "__main__":
    img_path = "/home/new/Data/VIN_CHECKED/crop"
    multiple = 10
    save_dir = "./result"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    imgs = os.listdir(img_path)
    num = len(imgs)
    data_augmentator = DataAugmentation()
    for img in imgs:
        label = os.path.splitext(img)[0].split('_')[1]
        old_path = os.path.join(img_path, img)
        out_pathes = list()
        for i in range(multiple):
            num += 1
            new_path = os.path.join(save_dir, str(num)+"_"+label+".jpg")
            out_pathes.append(new_path)
        data_augmentator.generate(old_path, out_pathes) 


