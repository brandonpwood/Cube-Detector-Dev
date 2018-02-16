import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_limit = np.array([30, 25, 0])
upper_limit = np.array([70, 70, 255])


while True:
	# Capture frame and convert colorspace 
	ret, frame = cap.read()	
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
#	cv2.imshow("mic check", frame)

	# Threshold
	mask = cv2.inRange(hsv, lower_limit, upper_limit)
	
	hold = cv2.bitwise_and(hsv, hsv, mask = mask)

	# Return to RGB
	color = cv2.cvtColor(hold, cv2.COLOR_HSV2BGR) 

	# Display
	cv2.imshow("test", color)

	if cv2.waitKey(1) == 27:
		break
cap.release()
cv2.destroyAllWindows()


