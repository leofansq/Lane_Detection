"""
Main Function
Code by leofsq
"""
import cv2
import numpy as np
import os
import time
from tqdm import tqdm

from basic_function import show_img, find_files
from unit_process import pre_process, find_line, detect_line

#**********************************************************#
#                         Option                           #
#**********************************************************#

# FILE PATH
SRC_PATH = "./img/"
SAVE_PATH = "./result/"

# IMAGE FORMAT
IMG_FORMAT = 'jpg'

#**********************************************************#
#                        Function                          #
#**********************************************************#
def main():
    time_cost = []
    for file_path in tqdm(find_files(SRC_PATH, '*.{}'.format(IMG_FORMAT))):
        _, file_name = os.path.split(file_path)
        print (file_name)        
        
        start_time = time.time()
        
        img = cv2.imread(file_path)
        show_img("origin", img)

        img_result = detect_line(img, debug=False)
        show_img("result", img_result)
        
        end_time = time.time()
        time_cost.append(end_time - start_time)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):break
        elif key == ord('s'):cv2.imwrite(SAVE_PATH+file_name,img_result)

    print ("Mean_time_cost: ", np.mean(time_cost))
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()
