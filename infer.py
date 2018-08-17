import os
from model.infer import Infer
from tqdm import tqdm

test_data_path = "/home/new/Data/vin_data_checked/val/augment"

test_imgs = os.listdir(test_data_path)

infer = Infer(ckpt_path = "./ckpt/")
infer.build_model()

counter = 0
right_counter = 0
pathes = []
for test_img in test_imgs:
    counter += 1
    img_abs_path = os.path.join(test_data_path, test_img)
    results = infer.recognize_from_path(img_abs_path)
    result = results[0] 
    gt = os.path.splitext(test_img)[0].split('_')[1]
    if gt == result:
        right_counter += 1
    #print("GT:{},Result:{}".format(gt, result))
    else:
        print("GT:{},Result:{}".format(gt, result))

print("Accuracy: {:.3f}".format(float(right_counter)/float(counter)*100))
