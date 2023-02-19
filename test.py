import cv2
import mediapipe as mp
import time
import serial

# ser = serial.Serial(port='COM3', baudrate=9600, timeout=.1)  # open serial port
# grab opened serial port
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

def dist(x1,y1,x2,y2):
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    trig = 0
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            if ( dist(handLms.landmark[8].x,handLms.landmark[8].y,handLms.landmark[4].x,handLms.landmark[4].y) < 0.1):
                trig = 1
                # ser.write(b'ON ')
            else:
                trig = 0
            #     # ser.write(b'OFF')
                
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
                if id == 8 and trig == 1:
                    cv2.circle(img, (cx, cy), 25, (255, 255, 0), cv2.FILLED)
                if id == 4 and trig == 1:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    img = cv2.flip(img,1)
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)