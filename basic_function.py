"""
Basic Function
Code by leofsq
"""
import cv2
import numpy as np
import os
import fnmatch

def show_img(name, img):
    """
    Show the image

    Parameters:    
        name: name of window    
        img: image
    """
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)

def find_files(directory, pattern):
    """
    Method to find target files in one directory, including subdirectory
    :param directory: path
    :param pattern: filter pattern
    :return: target file path list
    """
    file_list = []
    for root, _, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                file_list.append(filename)
    
    return file_list

def get_M_Minv():
    """
    Get Perspective Transform
    """
    src = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])
    dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M,Minv

def draw_area(img_origin, img_line, Minv, left_fit, right_fit):
    """
    Draw the road area in the image

    Parameters:
    img_origin: original iamge
    img_line: warp_line
    Minv: inverse parameteres for perspective transform
    left_fit: [a,b,c]
    right_fit: [a,b,c]

    Return:
    image with road area
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_line.shape[0]-1, img_line.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    mask_road_warp = np.zeros_like(img_line).astype(np.uint8)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(mask_road_warp, np.int_([pts]), (0, 255, 0))
    mask_road_warp = cv2.addWeighted(mask_road_warp, 1, img_line, 1, 0)    

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img_roadmask = cv2.warpPerspective(mask_road_warp, Minv, (img_origin.shape[1], img_origin.shape[0]))

    # Combine the result with the original image
    img_result = cv2.addWeighted(img_origin, 1, img_roadmask, 0.3, 0)
    return img_result

def draw_demo(img_result, img_bin, img_line, img_bev_result, curvature, distance_from_center):
    """
    Generate the Demo image

    Parameters:
    img_result: original image with road area
    img_bin: img_bin (Bin)
    img_line: img_line (Bin)
    img_bev_result: line result in bev (BGR) 
    curvature: radius of curvature
    distance_from_center: distance from center

    Return:
    Demo image
    """
    img_demo = img_result.copy()

    h, w = img_demo.shape[:2]
    ratio = 0.2
    show_h, show_w = int(h*ratio), int(w*ratio)
    offset_x, offset_y = 20, 15

    # Draw the highlight
    cv2.rectangle(img_demo, (0,0), (w, show_h+2*offset_y), (0,0,0), -1)
    img_demo = cv2.addWeighted(img_demo, 0.2, img_result, 0.8, 0)

    # Draw img_bin
    img_bin = cv2.resize(img_bin, (show_w, show_h))
    img_bin = np.dstack([img_bin, img_bin, img_bin])
    img_demo[offset_y:offset_y+show_h, offset_x:offset_x+show_w, :] = img_bin

    # Draw img_line
    img_line = cv2.resize(img_line, (show_w, show_h))
    img_line = np.dstack([img_line, img_line, img_line])
    img_demo[offset_y:offset_y+show_h, offset_x*2+show_w:(offset_x+show_w)*2, :] = img_line

    # Draw img_bev_result
    img_bev_result = cv2.resize(img_bev_result, (show_w, show_h))
    img_demo[offset_y:offset_y+show_h, offset_x*3+show_w*2:(offset_x+show_w)*3, :] = img_bev_result

    # Write the text
    pos_flag = 'right' if distance_from_center>0 else 'left'

    radius_text = "Radius of Curvature: %sm"%(round(curvature))
    center_text = "Vehicle is %.2fm %s of center"%(abs(distance_from_center),pos_flag)   
        
    cv2.putText(img_demo,radius_text,(860,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)
    cv2.putText(img_demo,center_text,(860,130), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)

    return img_demo