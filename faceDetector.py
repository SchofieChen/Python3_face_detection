import cv2
import datetime
import threading
from faced import FaceDetector
from faced.utils import annotate_image
import os
face_detector = FaceDetector()
video_capture = cv2.VideoCapture(0)

thresh = 0.85
cnt = 0
savePerson = 'Wei'
saveTrainFile = 'train'

def storeImageData(image):
    global cnt
    cnt = cnt+1
    dateInfo = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cv2.imwrite(saveTrainFile + '/' + savePerson+'/output'+dateInfo+str(cnt)+'.png', image)

def resizeImg(image,bboxes):
    imgResize = cv2.resize(image,(224,224))

while True:
    try:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Receives RGB numpy image (HxWxC) and
        # returns (x_center, y_center, width, height, prob) tuples. 
        bboxes = face_detector.predict(rgb_img, thresh)
        print(bboxes)
        
        # Use this utils function to annotate the image.
        ann_img = annotate_image(rgb_img, bboxes)
    except:
        pass
    
    '''
    auto focus images
    '''
    try:
        # Save image when face detected
        if bboxes != []:
            x_center = bboxes[0][0]
            y_center = bboxes[0][1]
            width = bboxes[0][2]
            height = bboxes[0][3]
            pic = frame[y_center-int(height/1.2):y_center+int(height/1.2),x_center-int(width/1.5):x_center+int(width/1.5)]
            cv2.imshow('Video', pic)
       
            #store Image
            #t = threading.Thread(target = storeImageData, args = (pic,))
            # 執行該子執行緒
            #t.start()
            #storeImageData(ann_img)
        else:
            cv2.imshow('Video', frame)
    except:
        pass
    # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
     
    # Display the resulting frame
    
    # #cv2.imshow('My Image', ann_img)
    
   
    # 按下任意鍵則關閉所有視窗271-146:271+146
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()