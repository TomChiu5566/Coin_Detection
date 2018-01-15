# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 01:55:49 2017

@author: Tom
"""

import cv2 
import numpy as np

def displayIMG(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def gray_and_Gaussian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        ## Gaussian filter eliminates noises
    gray = cv2.GaussianBlur(gray, (15,15), 0)
    return gray

def find_std(all_coins):
    maxi = 0
    idx = 0
    std_radius = 0
    for (i ,masked) in enumerate(all_coins):
        gray = gray_and_Gaussian(masked)
        if np.mean(gray) > maxi:
            maxi = np.mean(gray)
            idx = i
            std_radius = np.mean(np.shape(gray))
    return std_radius - 3, idx ### try

def repeat_check(M_now, M_list):
    for M_pre in M_list:
        if distance(M_now, M_pre) < 50:
            return False
    return True

def distance(loc1, loc2):
    d = np.sqrt((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2)
    return d

def tell_kind(crop, kinds, std):

    score = [0, 0, 0, 0]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    I = 0
    num = 0
    for i in gray:
        for j in i:
            if j != 0:
                num += 1
                I += j
### size: one:20mm, five:22mm, ten:26mm, fifty:28mm
    radius = np.mean(np.shape(gray))
    not_coin = std * 7/13
    one_th   = std * 10.8/13
    five_th  = std * 12.25/13
    ten_th   = std * 13.8/13 
    fifty_th = std * 16/13

    if radius < not_coin:
        print('not a coin')
        return -1
    elif radius >= not_coin and radius < one_th:
        score[3] += 200
        print('one')
    elif radius >= one_th and radius < five_th:
        score[2] += 200
        print('five')
    elif radius >= five_th and radius < ten_th:
        score[1] += 200
        print('ten')
    elif radius >= ten_th and radius < fifty_th:
        score[0] += 200
        print('fifty')
    else:
        print('not a coin')
        return -1
    kinds = np.argmax(score)
            
    return kinds

def output_result(result):
    total_sum = result[0] * 50 + result[1] * 10 + result[2] * 5 + result[3]    
    if result[0] == 0:
        print('There is no fifty-dollar-coin!')
    elif result[0] == 1:
        print('There is 1 fifty-dollar-coin!')
    else:
        print('There are ' + str(result[0]) + ' fifty-dollar-coins!')
        
    if result[1] == 0:
        print('There is no ten-dollar-coin!')
    elif result[1] == 1:
        print('There is 1 ten-dollar-coin!')
    else:
        print('There are ' + str(result[1]) + ' ten-dollar-coins!')
        
    if result[2] == 0:
        print('There is no five-dollar-coin!')
    elif result[2] == 1:
        print('There is 1 five-dollar-coin!')
    else:
        print('There are ' + str(result[2]) + ' five-dollar-coins!')
        
    if result[3] == 0:
        print('There is no one-dollar-coin!\n')
    elif result[3] == 1:
        print('There is 1 one-dollar-coin!\n')
    else:
        print('There are ' + str(result[3]) + ' one-dollar-coins!\n')
   
    print('There are ' + str(total_sum) + ' dollars in total!!!!')     
#%% Parameters
filename = 'coins_089.jpg'
img = cv2.imread(filename)
displayIMG('IMG', img)

r = 450.0 / img.shape[0]
dim = (int(img.shape[1] * r), 450)
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


gray = gray_and_Gaussian(resized)        ## Gaussian filter eliminates noises
size_x, size_y = np.shape(gray)
displayIMG('gray', gray)


edged = cv2.Canny(gray, 30, 100)
displayIMG('edged', edged)
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
coins = resized.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)

total   = 0
kinds   = [0, 0, 0, 0] ## [ fifties, tens, fives, ones]
#bg_img = resized.copy() ## background
coins = resized.copy()
all_coins= [] 
M_list = []
for (i, c) in enumerate(cnts):
    area = cv2.contourArea(c)
    if area > 20:       
        M = cv2.moments(c)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        M_now = (cX, cY)
        check = repeat_check(M_now, M_list)
        if check == False:
            continue
        M_list.append(M_now)
        cv2.circle(resized, (cX, cY), 5, (1, 227, 254), -1)
        
        (x, y, w, h) = cv2.boundingRect(c)
        if w < 10 or h < 10 or w > 130 or h > 130:
            M_list.pop()
            continue
        total += 1
        cv2.drawContours(coins, c, -1, (0, 255, 0), 2)
        

        coin = resized[y:y + h, x:x + w]
        mask = np.zeros(resized.shape[:2], dtype = 'uint8')
        ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(mask, (int(centerX), int(centerY)), int(radius-2), 255, -1)
        mask = mask[y:y + h, x:x + w]
        masked = cv2.bitwise_and(coin, coin, mask = mask)
        all_coins.append(masked)
#        bg_img[y:y+h, x:x+w] = [0, 0 ,0]
displayIMG('resized', coins)
#bg = np.mean(bg_img)
std, std_idx = find_std(all_coins)

result = [0, 0, 0, 0]
for i, masked in enumerate(all_coins):

    if i == std_idx:
        print('std image')
        displayIMG('std', masked)
        continue
    kinds = tell_kind(masked, kinds, std)
    if kinds == 0:
        result[0] += 1
    elif kinds == 1:
        result[1] += 1
    elif kinds == 2:
        result[2] += 1
    elif kinds == 3:
        result[3] += 1
    displayIMG('Masked Coins', masked)

output_result(result)
displayIMG('result', coins)
