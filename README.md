## About
This code is used for lane detection in driving environment.

![Result](./result/7.jpg)

## How to use
1.  Set the FILE_PATH & IMAGE_FORMAT in [main.py](./main.py)
2.  Run the main.py

> *tqdm* is used to generate the processbar in the terminal. It's Optional. If you don't want to use it, you can replace the code at *line_30* in [main.py](./main.py) as below.
>   ```Python
>   for file_path in find_files(SRC_PATH, '*.{}'.format(IMG_FORMAT)):
>   ```

## Code Info

* [main.py](./main.py) --- Main Function
* [basic_function](./basic_function.py) --- Some Basic Function
    * show_img(name, img) : Show the image
    * find_files(directory, pattern) : Method to find target files in one directory, including subdirectory
    * get_M_Minv() : Get Perspective Transform
    * draw_area(img_origin, img_line, Minv, left_fit, right_fit) : Draw the road area in the image
    * draw_demo(img_result, img_bin, img_line, img_bev_result, curvature, distance_from_center) : Generate the Demo image
* [unit_process](./unit_process.py) --- Unit Operation for Image
    * detect_line(img, debug=False) : Main Function
    * pre_process(img, debug=False) : Image Pre-process
    * find_line(img, debug=False) : Detect the lane using Sliding Windows Methods
    * calculate_curv_and_pos(img_line, left_fit, right_fit) : Calculate the curvature & distance from the center



