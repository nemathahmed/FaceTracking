import numpy as np
import cv2
import time

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    count=0
    firstface=0
    while True:
        now=time.time()

        counte=0
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
        ret_val, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            if firstface==0:
                ix=x
                iy=y
                iw=w
                ih=h
                firstface=1
                count=1
                future=now+10
                print(now,future)
        print(now)
        if count==1 and int(time.time())==int(future):
            count=0
            firstface=0
        if count==1:
            
            img = cv2.rectangle(img,(ix,iy),(ix+iw,iy+ih),(255,0,0),2)
            roi_gray = gray[iy:iy+ih, ix:ix+iw]
            roi_color = img[iy:iy+ih, ix:ix+iw]
            #eyes = eye_cascade.detectMultiScale(roi_gray)
        '''
                       for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                counte=1

            count=1
        '''

        if count==0:
           img = np.zeros([512,512,3])
        '''    
        if count==1:
            if counte==1:
                for i in range(ex,ex+ew):
                    for j in range(ey,ey+eh):
                        img[i,j,0]=0
                        img[i,j,1]=0
                        img[i,j,2]=0
            img=img[y:y+h, x:x+w]
            img=cv2.resize(img,(512,512))
        ''' 
        if count==1:  
            img=img[iy:iy+ih,ix:ix+iw]
            img=img[:,:,2]
            img=cv2.resize(img,(512,512))
           # print(ix,iy,iw,ih)
        cv2.imshow('Photo Video Camera Stream', img)
        
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

show_webcam()