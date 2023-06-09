import cv2             # OpenCV computer vision for image processing
import os              # interacting with operating system
import json            # Javascript Mainly for config files, storing data or APIs (for working with JSON files)
from tqdm import tqdm  # For displaying progress bars during iterations
import numpy as np     # For numerical operations

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result)

img_dir = '/Users/seanl/Downloads/Test_file' # Linux is forward slash /
#print(os.path.join(img_dir,'image2.jpg'))
img = cv2.imread(os.path.join(img_dir, 'image2.jpg')) # EDIT THIS CODE
adjusted_img = automatic_brightness_and_contrast(img)
#print(adjusted_img.shape)
cv2.imshow('Original image ', img)
cv2.imshow('Adjusted image ', adjusted_img)
cv2.waitKey(0) # this is necessary otherwise the image windows will close immediately upon launch
#waitkey(0) means that it will wait indefinitely until any key is pressed.
#waitkey(3000) means that it will wait for 3s until it closes the cv2 image window
cv2.destroyAllWindows()
save_path = os.path.join(img_dir, 'image2_adjusted.png') # EDIT THIS CODE
# the save_path must have a .jpg at the end
print(save_path)
cv2.imwrite(save_path, adjusted_img)


        # Example output when we use img.shape()
        # (1280, 598, 3) [It gives a tuple showing height, width and number of colour channels (BGR)]

#crop_img_save_path = os.path.join(crop_img_save_dir, crop_img_save_filename)
        # A string called crop_img_save_path is created. It uses the os.path.join() function to join the crop_img_save_dir (a directory path) and crop_img_save_filename (a filename) together, creating a complete path to save the image. The resulting string is assigned to crop_img_save_path.

#cv2.imwrite(crop_img_save_path, crop_img)
        # The cv2.imwrite() function from the OpenCV library is used to save an image at a specific path. It takes two arguments: crop_img_save_path (the path where the image should be saved) and crop_img (the image data to be saved).



