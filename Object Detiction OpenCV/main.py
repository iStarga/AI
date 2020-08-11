import cv2 as cv
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))

haystack_img = cv.imread('farm.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('cabbage.jpg', cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)


print(f'Best match top left position: {max_loc}')
print(f'Best match confidence: {max_val}')


threshold = 0.8
if max_val >= threshold:
    print('Found needle.')

    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

    cv.rectangle(haystack_img, top_left, bottom_right,
                 color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)

    cv.imwrite('result.jpg', haystack_img)

else:
    print('Needle not found.')
