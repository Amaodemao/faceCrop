import dlib
import numpy as np
import cv2
import math
import os

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 

def arc_points(point1, point2, num_of_points):
    points = []
    center_x = (point1[0] + point2[0])/2
    center_y = (point1[1] + point2[1])/2
    radius = abs((point1[0] - point2[0])/2)
    for i in range(num_of_points):
        if i == 0:
            continue
    
        point = []
        x = center_x + radius * math.cos(math.pi + i * math.pi / num_of_points)
        y = center_y + radius * math.sin(math.pi + i * math.pi / num_of_points)
        point.append(x)
        point.append(y)
        
        points.append(point)
    
    return points

def get_landmarks(img):
    dets = detector(img, 1)
    landmarks = np.zeros((34, 2))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        for i in range(17):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)
        
        point1 = [shape.part(0).x, shape.part(0).y]
        point2 = [shape.part(16).x, shape.part(16).y]
        points = arc_points(point1, point2, 18)
        for i in range(len(points)):
            landmarks[33 - i] = (points[i][0], points[i][1])
    
    return landmarks

def inside(X,Y,Region): 
    j=len(Region)-1
    flag=False
    for i in range(len(Region)):
        if (Region[i][1]<Y and Region[j][1]>=Y or Region[j][1]<Y and Region[i][1]>=Y):  
            if (Region[i][0] + (Y - Region[i][1]) / (Region[j][1] - Region[i][1]) * (Region[j][0] - Region[i][0]) < X):
                flag = not flag
        j=i
    return flag

def main():
    # Recursively search for the image in the directory
    for root, dirs, files in os.walk(".\inputs"): # Input directory here. Modify it before using.
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                path = os.path.join(root, file)
                print(path)
                img = cv2.imread(path).astype(np.uint8)
                region = get_landmarks(img)
                shape = list(img.shape) 
                cropped_img = img.copy()
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        if not inside(j, i, region): 
                            cropped_img[i, j] = (255, 255, 255) # Set the pixel to white if it is outside the region.
                            # I didn't mask the picture and fill the exterior region with transparent pixels because I am lazy :)
                            # You can try to do so if you want.
                output_dir = os.path.dirname(path.replace('inputs', 'outputs'))
                print(output_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    cv2.imwrite(os.path.join(output_dir, file), cropped_img)
                else:
                    cv2.imwrite(os.path.join(output_dir, file), cropped_img)



if __name__ == "__main__":
    main()