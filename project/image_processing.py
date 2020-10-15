#!/usr/bin/env python3
import rclpy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from rclpy.node import  Node
from example_interfaces.msg import String
import threading
class Img_proc(Node):
    def callback(self,directions):
        self.get_logger().info(directions.data)

    def __init__(self):
        super().__init__("Image_processing")
        self.subscriber_=self.create_subscription(String,"DIRECTION",self.callback,10)
        self.publisher_=self.create_publisher(String,"GATE_POSITION",10)
        t1=threading.Thread(target=process_img,args=[self])
        t1.start()
        
        
def process_img(self):
    cap = cv2.VideoCapture('/home/islam/Videos/right_log11.avi')

 
    fps = 5
    cnt=0
    # cv2.namedWindow('HSV')
    # cv2.resizeWindow('HSV',640,480)
    # cv2.createTrackbar('th1','HSV',80,180,empty)
    # cv2.createTrackbar('th2','HSV',150,180,empty)

    # cv2.createTrackbar('hue min','HSV',0,180,empty)
    # cv2.createTrackbar('hue max','HSV',84,180,empty)
    # cv2.createTrackbar('Min sat','HSV',0,255,empty)
    # cv2.createTrackbar('Max sat','HSV',143,255,empty)
    # cv2.createTrackbar('Min val','HSV',0,255,empty)
    # cv2.createTrackbar('Max val','HSV',255,255,empty)
    
    
    if cap.isOpened()== False: 
        print("Error opening the video file. Please double check your file path for typos. Or move the movie file to the same location as this script/notebook")
    
    while cap.isOpened():
        success, frame = cap.read() 
        
        if success==True:
        
            framehsv =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            # th1=cv2.getTrackbarPos('th1','HSV')
            # th2=cv2.getTrackbarPos('th2','HSV')
            # x=cv2.getTrackbarPos('hue min','HSV')
            # h_max=cv2.getTrackbarPos('hue max','HSV')
            # s_min=cv2.getTrackbarPos('Min sat','HSV')
            # s_max=cv2.getTrackbarPos('Max sat','HSV')
            # v_min=cv2.getTrackbarPos('Min val','HSV')
            # v_max=cv2.getTrackbarPos('Max val','HSV')
            
            # if x%2!=1 and x!=100:
            #     cv2.setTrackbarPos("hue min","HSV",x+1)
            #     x+=1
            # elif x==100:
            #     x-=1
    
            #lower=np.array([h_min,s_min,v_min])
            #upper=np.array([h_max,s_max,v_max])

            mask_orange1=cv2.inRange(framehsv,np.array([0,0,0]),np.array([90,136,148]))
            result_mask=cv2.bitwise_and(frame,frame,mask=mask_orange1)
            result=np.array(result_mask)
            result=getcontours(1,result,[80,150])
            
            contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            Rects=[]
            areas=[]
            for c in contours:
                peri=cv2.arcLength(c,True)       
                approx =cv2.approxPolyDP(c,0.02*peri,True)
                if cv2.contourArea(c)>1000 and len(approx)>2 and len(approx)<7:
                    (x, y, w, h) = cv2.boundingRect(c)
                    
                    (win_w,win_h,j)=frame.shape
                    flag=0
                    
                    if y+h>150 and y<.8*win_h:
                        Rects.append((x,y,w,h))  
                        cv2.rectangle(frame, (x,y),(x + w, y + h) , (255, 0, 255), 2)
        
            Rects.sort()
            #print(Rects)
            if len(Rects)==0:
                pos=0
                
            elif len(Rects)==1:
                (x, y, w, h)=Rects[0]
                pos=0
            else:
                (x, y, w, h)=Rects[0]
                (x2, y2, w2, h2)=Rects[len(Rects)-1]
                p1=(x,y);p2=(x2 + w2, y2 + h2)
                area=(x2-x+w2)*(y2+h2-y)
                if area>8000:
                    if len(Rects)==2 or len(Rects)>3:
                        if y<y2:
                            cv2.rectangle(frame, (x,y),(x2 + w2, y2 + h2) , (255, 255, 0), 2)                   
                                #elf.publisher_.publish(pos)
                        else:
                            p1=(x,y2);p2=(x2+w2,y+h)
                            cv2.rectangle(frame, (x,y2),(x2 + w2, y + h) , (255, 255, 0), 2)
                        pos=getPos(frame.copy(),p1,p2)
                    
                    elif len(Rects)==3:
                        (x1, y1, w1, h1)=Rects[0]
                        (x2, y2, w2, h2)=Rects[1]
                        (x3, y3, w3, h3)=Rects[2]
                        if x2+int(w2/2)-x3-int(w3/2)>x1+int(w1/2)-x2-int(w2/2):
                    
                            if y1>y3:
                                p1=(x1,y1);p2=(x3 + w3, y3 + h3)
                            else:
                                p1=(x,y3);p2=(x3+w3,y1+h1)
                            p3,p4=(x2+w2,y2),(x3 + w3, y3 + h3)    
                            cv2.rectangle(frame, (x2+w2,y2),(x3 + w3, y3 + h3) , (0, 255, 255), 2)
                            cv2.putText(frame,'Bonus',(x2 +w2, y2),cv2.FONT_HERSHEY_COMPLEX,.7,(0,255,255),2)
                        else :
                            if y1>y3:
                                p1 = (x1, y1); p2 = (x3 + w3, y3 + h3)
                            
                            else:
                                p1 = (x, y3); p2 = (x3 + w3, y1 + h1)
                            p3,p4=(x2+w2,y2),(x1 , y1 + h1)
                            cv2.rectangle(frame, (x2+w2,y2),(x1 , y1 + h1) , (0, 255, 255), 2)
                            cv2.putText(frame,'Bonus',(x2 +w2+20, y2),cv2.FONT_HERSHEY_COMPLEX,.7,(0,255,255),2)
                        pos=getPos(frame.copy(),p3,p4)
                        cv2.rectangle(frame,p1,p2,(255,255,0),2)
                
                    
            msg=String()
            msg.data=str(pos)
            self.publisher_.publish(msg)
            frame=np.array(frame)
            #imgs=stackImages(.8,[frame,result_mask,result])
            cv2.imshow('frame',frame)
            cv2.resizeWindow('frame',1000,1000)
        
            
            
            time.sleep(1/fps)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1)
            
        else:
            break
            
    cap.release()
    
    cv2.destroyAllWindows()
    #self.publisher_.publish(pos) to be used when gate detected     
def getcontours(x,frame,cThr=[0,0],showCanny=False):
        frameGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #frameGray=cv2.GaussianBlur(frameGray,(9,9),1)
        frameGray=cv2.medianBlur(frame,x)
        #frameGray=cv2.adaptiveThreshold(frameGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        imgcanny=cv2.Canny(frameGray,cThr[0],cThr[1])
        kernel=np.ones((5,5))
        imgdial=cv2.dilate(imgcanny,kernel,iterations=2)
        #imgdial=cv2.erode(imgdial,kernel,iterations=2)
        #cv2.imshow('canny',imgcanny)
        return imgdial
def getRow(center,height):
    if (center[1] <= height // 3):
        return 0
    elif (center[1] <= 2*height // 3):
        return 1
    else:
        return 2

def getCol(center,width):
    if (center[0] <= width // 3):
        return 0
    elif (center[0] <= 2*width // 3):
        return 1
    else:
        return 2

def getPos(frame,p1,p2):
    height,width,x=frame.shape
    cv2.line(frame, (width // 3, 0), (width // 3, height), (0, 255, 0), thickness=1)
    cv2.line(frame, (2*width // 3, 0), (2*width // 3, height), (0, 255, 0), thickness=1)
    cv2.line(frame, (0,height//3), (width,height//3), (0, 255, 0), thickness=1)
    cv2.line(frame, (0,2*height//3), (width,2*height//3), (0, 255, 0), thickness=1)
    center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    row=getRow(center,height)
    col=getCol(center,width)
    cv2.putText(frame, 'X', (width//6+col*width//3,height//6+row*height//3), cv2.FONT_HERSHEY_COMPLEX,1, (255, 255, 255), 2)
    cv2.imshow("DividedFrame",frame)
    pos = 3 * row + col + 1
    return pos        
def main(args=None):
    rclpy.init(args=args)  
    node=Img_proc()

    rclpy.spin(node)
    rclpy.shutdown()

if __name__=="__main__":
    main()