import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="2"
import glob
import cv2
import time
from tqdm import tqdm
from jeongho.module_2class import classifier

img_paths = glob.glob("./test_img(test_falldown)/*")
img_paths.sort()
classifier = classifier(model_path1 = "./best_model/efficientnetb0_ns_fit_background_vs_person_stage1stage2_back_vs_person_7_97.11021505376344_False_False.pt",
                         model_path2 = "./best_model/efficientnetb0_ns_fit_normal_vs_falldown_stage1stage2_normal_vs_falldown_3_93.26923076923077_False_False.pt",
                        back_vs_person_padding = False, back_vs_person_normalize = False,
                        normal_vs_falldown_padding = False, normal_vs_falldown_normalize = False)

results = []
count = [0,0,0]
start = time.time()
for n, path in tqdm(enumerate(img_paths)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[np.newaxis, :, :, :]
    result = classifier.predict(img)
    #result.append((path, result_dict[0]["class"], result_dict[0]["confidence"]))
    results.append(result)
    #count[result_dict[0]["class"]] += 1
#print("time : {}".format(time.time() - start))
print(results)
#print(count)
