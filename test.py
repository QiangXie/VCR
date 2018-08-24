import os
from model.infer import Infer
import argparse

parser = argparse.ArgumentParser(description='Process argument')
parser.add_argument("--test_data_path", dest="test_data_path", type=str, default="./1")
args = parser.parse_args()

test_imgs = os.listdir(args.test_data_path)

infer = Infer(ckpt_path = "./ckpt/")
infer.build_model()

for test_img in test_imgs:
    img_abs_path = os.path.join(args.test_data_path, test_img)
    results = infer.recognize_from_path(img_abs_path)
    result = results[0] 
    print("{} rec result:{}".format(img_abs_path, result))
