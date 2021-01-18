 
import cv2 
import os
import image_demo

import shutil



# function to display the coordinates of 
# of the points clicked on the image  
  
# driver function
count = 0
totalCount = 0

def extractFramesAndCallImageDemo(path):
    cap = cv2.VideoCapture(path)  # video_name is the video being called
    print(path)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    videoname = path.split('/')[-1]
    size = totalFrames / 3
    frames = []
    frames.append(int(0.5 * size))
    frames.append(int(1.5 * size))
    frames.append(int(2.5 * size))
    os.mkdir(f"./images")

    for i in frames:
        cap.set(1,i)
        ret, frame = cap.read() # Read the frame
        cv2.imwrite(f"./images/{videoname.split('.')[0]}"+str(i)+'.jpg', frame)
    wordValidity = image_demo.word(videoname.split('.')[0])
    if(wordValidity):
        count += 1
    try:
        shutil.rmtree(f"./images")
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))
    
if __name__ == "__main__":
    

    for file in os.listdir(os.getcwd()+'/input'):
        if file.endswith(".mp4"):
            totalCount += 1
            path=os.path.join(os.getcwd()+'/input', file)
            extractFramesAndCallImageDemo(path)
    
    
    print("accuracy:")
    print(count/totalCount)
    cv2.waitKey(0) 
  
    # close the window 
    cv2.destroyAllWindows() 
