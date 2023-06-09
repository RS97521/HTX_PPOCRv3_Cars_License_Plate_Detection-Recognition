import cv2             # OpenCV computer vision for image processing
import os              # interacting with operating system
import json            # Javascript Mainly for config files, storing data or APIs (for working with JSON files)
from tqdm import tqdm  # For displaying progress bars during iterations
import numpy as np     # For numerical operations

def get_rotate_crop_image(img, points):
# This function takes two parameters: img and points & performs image cropping and rotation based on the provided points.
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    # checks if the length of the points array is equal to 4. If not, it raises an assertion error with the specified message. This ensures that points contains exactly four sets of coordinates.

    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                        [img_crop_width, img_crop_height],
                        [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

img_dir = '/Users/seanl/OneDrive/Desktop/Pytorch_Project/images' # Linux is forward slash /
save_gt_folder = '/Users/seanl/OneDrive/Desktop/Pytorch_Project/Groundtruth'

crop_img_save_dir = os.path.join(save_gt_folder, 'train', 'crop_imgs')
os.makedirs(crop_img_save_dir, exist_ok=True)

f_det = open(os.path.join(save_gt_folder, 'train', 'det.txt'), 'w', encoding='utf-8')
f_rec = open(os.path.join(save_gt_folder, 'train', 'rec.txt'), 'w', encoding='utf-8')
#f_det = open(os.path.join(save_gt_folder, 'val', 'det.txt'), 'w', encoding='utf-8')
#f_rec = open(os.path.join(save_gt_folder, 'val', 'rec.txt'), 'w', encoding='utf-8')
#f_det = open(os.path.join(save_gt_folder, 'test', 'det.txt'), 'w', encoding='utf-8')
#f_rec = open(os.path.join(save_gt_folder, 'test', 'rec.txt'), 'w', encoding='utf-8')
#f_det and f_rec are text files

i = 0 #image number
for filename in os.listdir(os.path.join(img_dir,'train')):
    print(filename)
    #print(type(filename))

    str_list = filename.split('-')
    print(str_list)

    if len(str_list) < 5: # only 5 instead of the full 7 as brightness and bluriness can be omitted
        continue
    print('pass')

    coord_list = str_list[3].split('_')
    print(coord_list)

    txt_list = str_list[4].split('_')
    print(txt_list)

    boxes = [] # Empty list to store coordinates.
    for coord in coord_list:
        boxes.append([int(x) for x in coord.split("&")])
    print(boxes)

    lp_number = provinces[int(txt_list[0])] + alphabets[int(txt_list[1])] + ''.join([ads[int(x)] for x in txt_list[2:]])
    print(lp_number)

    #detection
    det_info = [{'points':boxes, 'transcription':lp_number}] # Create a list of dictionaries (det_info) containing the 'points' (coordinates) and 'transcription' (license plate number) information.
    print(det_info)
    f_det.write('{}\t{}\n'.format(os.path.join('train', filename), json.dumps(det_info, ensure_ascii=False))) # json.dumps(obj) converts a python obj (such as dictionary or list) into a JSON String representation
    

    #RECOGNITION
    print(boxes)
    boxes = np.float32(boxes) # Convert the boxes list to a NumPy array of type float32.
    print(boxes)
    #img = cv2.imread(os.path.join(img_dir, phase, filename))
    # img = cv2.imread(os.path.join(img_dir, phase, filename)): Read the image from the specified file path (img_dir/phase/filename) using OpenCV's imread() function and store it in the img variable.

    img = cv2.imread(os.path.join(img_dir, 'train', filename))
    if img is not None:
        #display image dimensions (it shows height, width and number of colour channels (BGR))
        print('image dimensions: ',img.shape)
    else:
        print('image not loaded')

    #crop_img = img[int(boxes[:,1].min()):int(boxes[:,1].max()),int(boxes[:,0].min()):int(boxes[:,0].max())]
    crop_img = get_rotate_crop_image(img, boxes)
    #print(crop_img)
    #print('_'.join(txt_list))
    crop_img_save_filename = '{}_{}.jpg'.format(i,'_'.join(txt_list))
    print(crop_img_save_filename)

    crop_img_save_path = os.path.join(crop_img_save_dir, crop_img_save_filename)

    cv2.imwrite(crop_img_save_path, crop_img)

    f_rec.write('{}/crop_imgs/{}\t{}\n'.format('train', crop_img_save_filename, lp_number))
    # This line writes a formatted string to a file object f_rec. It uses the write() method to write the formatted string. The string is formatted with three values: phase, crop_img_save_filename, and lp_number. 
    i+=1
f_det.close() # To ensure that reading from or writing to the f_det file is no longer possible
f_rec.close()