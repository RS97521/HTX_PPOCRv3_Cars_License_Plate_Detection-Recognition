import cv2             # OpenCV computer vision for image processing
import os              # interacting with operating system
import json            # Javascript Mainly for config files, storing data or APIs (for working with JSON files)
from tqdm import tqdm  # For displaying progress bars during iterations
import numpy as np     # For numerical operations

#provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
#alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O'] # 'I' and 'O' are excluded in license plates to avoid confusion with 1 and 0' 
#ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']# other letters excluded are 'F', 'N', 'Q' and 'V'
universal_list = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','G','H','J','K','L','M','P','R','S','T','U','W','X','Y','Z'] # 30 characters, last index 29

# img_dir - The directory containing the images
# save_gt_folder - The folder where the GROUND TRUTH data is stored (The human annotated/labelled data for referencing)
# GROUND TRUTH data refers to the accurate and reliable information that serves as a reference or benchmark for evaluating the performance of a machine learning model or algorithm. 
# phase - 'train', 'val' or 'test'
def make_label(img_dir, save_gt_folder, phase): 
    crop_img_save_dir = os.path.join(save_gt_folder, phase, 'crop_imgs')
    os.makedirs(crop_img_save_dir, exist_ok=True)
    # os.path.join() creates a directory path for saving cropped images by joining the save_gt_folder, phase, and 'crop_imgs'. E.g. \home\aistudio\data\CCPD2020\PPOCR\train\crop_imgs
    # os.makedirs() then creates the directory if it doesn't already exist.


    f_det = open(os.path.join(save_gt_folder, phase, 'det.txt'), 'w', encoding='utf-8')
    f_rec = open(os.path.join(save_gt_folder, phase, 'rec.txt'), 'w', encoding='utf-8')
    # These lines open two files for writing: 'det.txt' and 'rec.txt' within the save_gt_folder and phase subdirectories. 
    # The files are opened in write mode ('w') and encoded with UTF-8.

    i = 0 # image number
    for filename in tqdm(os.listdir(os.path.join(img_dir, phase))):
        # os.path.join(img_dir, phase) E.g. /Users/seanl/OneDrive/Desktop/Pytorch_Project/images/train
        # os.listdir() # Creates a list of all the files in the directories
        # tqdm() is just the format to display a progress bar
        # filename is in string format (I CHECKED)
        
        # BEFORE CONTINUING, ENSURE THAT THE DATAs WITH THE APPROPRIATE NAMING CONVENTIONS ARE IN THE os.path.join(img_dir, phases) folders


        # *****************ADD IMAGE ADJUSTMMET CODE HERE!!!!*************


        str_list = filename.split('-')
        # Split the filename using the '-' character as a delimiter and store the resulting substrings in a list called str_list.

        if len(str_list) < 5: # only 5 instead of the full 7 as brightness and bluriness can be omitted
            continue 
        # if a filename in the directory being iterated does not have at least 5 parts when split by the '-' character, 
        # the remaining code in the current iteration is skipped, and the loop proceeds to the next filename in the directory.
        
        coord_list = str_list[3].split('_')
        # Split the fourth element of str_list using the '_' character as a delimiter and store the resulting substrings in a list called coord_list.
        txt_list = str_list[4].split('_')
        # Split the fifth element of str_list using the '_' character as a delimiter and store the resulting substrings in a list called txt_list.
        boxes = [] # Empty list to store coordinates.
        for coord in coord_list:
            boxes.append([int(x) for x in coord.split("&")]) 
        # Iterate over each coordinate in coord_list, split it using the '&' character as a delimiter, convert the resulting substrings to integers, and append them as a list to the boxes list.
        # E.g. boxes = [[386, 473], [177, 454], [154, 383], [363, 402]]
        
        boxes = [boxes[2], boxes[3], boxes[0], boxes[1]] # Rearrange the elements of boxes to a specific order [x1, y1, x2, y2].
        lp_number = provinces[int(txt_list[0])] + alphabets[int(txt_list[1])] + ''.join([ads[int(x)] for x in txt_list[2:]])
        #Create a license plate number (lp_number) by concatenating elements from provinces, alphabets, and ads lists based on the corresponding indices in txt_list.

        # ***DETECTION***
        det_info = [{'points':boxes, 'transcription':lp_number}] # Create a list with a dictionary containing the 'points' (coordinates) and 'transcription' (license plate number) information of the single image.
        f_det.write('{}\t{}\n'.format(os.path.join(phase, filename), json.dumps(det_info, ensure_ascii=False))) # json.dumps(obj) converts a python obj (such as dictionary or list) into a JSON String representation
        # Write the information to a text file called 'f_det'.
        # os.path.join(phase, filename): This is the first value that will replace the first {} in the string. os.path.join() is a function used to concatenate directory paths, and it is used here to join the phase and filename values into a single path.
        # json.dumps(det_info, ensure_ascii=False): This is the second value that will replace the second {} in the string. json.dumps() is a function that converts a Python object (in this case, det_info) into a JSON string representation. The ensure_ascii=False argument is used to ensure that non-ASCII characters are properly encoded in the JSON string.

        # E.g. HOW IT LOOKS INSIDE THE f_det.txt file [The json.dumps() part is in string format]
        # train\025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg	[{"points": [[386, 473], [177, 454], [154, 383], [363, 402]], "transcription": "皖AY339S"}]
        # train\033-97_113-154&383_386&473-386&473_177&454_154&383_363&402-1_0_23_28_29_30_16-45-19.jpg	[{"points": [[386, 473], [177, 454], [154, 383], [363, 402]], "transcription": "沪AZ456S"}]


        # ***RECOGNITION***
        boxes = np.float32(boxes) # Convert the boxes list to a NumPy array of type float32.
        # Initial format    [[386, 473], [177, 454], [154, 383], [363, 402]]
        # float32 format   [[386. 473.]
        #                   [177. 454.]
        #                   [154. 383.]
        #                   [363. 402.]]

        img = cv2.imread(os.path.join(img_dir, phase, filename))
        # img = cv2.imread(os.path.join(img_dir, phase, filename)): Read the image from the specified file path (img_dir/phase/filename) using OpenCV's imread() function and store it in the img variable.
        # Example output when we use img.shape
        # (1280, 598, 3) [It gives a tuple showing height, width and number of colour channels (BGR)]

        # [FROM SOURCE] crop_img = img[int(boxes[:,1].min()):int(boxes[:,1].max()),int(boxes[:,0].min()):int(boxes[:,0].max())]
        crop_img = get_rotate_crop_image(img, boxes)
        # Call a function called get_rotate_crop_image() with the img and boxes as arguments to obtain a cropped image.

        crop_img_save_filename = '{}_{}.jpg'.format(i,'_'.join(txt_list))
        # A string called crop_img_save_filename is created. It uses the format() method to format the string with two values: i and '_'.join(txt_list). i is the count, 
        # and '_' is used to join the elements of the txt_list list with underscores. The resulting string is assigned to crop_img_save_filename, which will have a format like 'i_value0_value1_value2_value3_value4_value5_value6.jpg'.

        crop_img_save_path = os.path.join(crop_img_save_dir, crop_img_save_filename)
        # A string called crop_img_save_path is created. It uses the os.path.join() function to join the crop_img_save_dir (a directory path) and crop_img_save_filename (a filename) together, creating a complete path to save the image. The resulting string is assigned to crop_img_save_path.

        cv2.imwrite(crop_img_save_path, crop_img)
        # The cv2.imwrite() function from the OpenCV library is used to save an image at a specific path. It takes two arguments: crop_img_save_path (the path where the image should be saved) and crop_img (the image data to be saved).

        f_rec.write('{}/crop_imgs/{}\t{}\n'.format(phase, crop_img_save_filename, lp_number))
        # This line writes a formatted string to a file object f_rec. It uses the write() method to write the formatted string. The string is formatted with three values: phase, crop_img_save_filename, and lp_number. 
        i+=1
    f_det.close() # To ensure that reading from or writing to the f_det file is no longer possible
    f_rec.close()


# DON'T EDIT THIS
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


# MAIN CODE STARTS HERE
img_dir = '/Users/seanl/OneDrive/Desktop/Pytorch_Project/images' # Linux is forward slash /
save_gt_folder = '/Users/seanl/OneDrive/Desktop/Pytorch_Project/Groundtruth'
# E.g. phase = 'train' # changes to val and test to make val dataset and test dataset
for phase in ['train','val','test']:
    make_label(img_dir, save_gt_folder, phase) #each iteration, phase variable takes on 'train', 'val', then 'test