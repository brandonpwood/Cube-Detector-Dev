import cv2
import numpy as np

class Vis:
	def __init__(self, image = '', verbose = False):
		# Storage			
		self.file_name = image
		self.SQUARENESS = 40 # Scaled down by 1000 before input
		self.MIN_PERIM = 10 # Scaled up by 10 before input
	
		# Settings
		self.verbose = verbose		
		if image != '':
			self.use_image = True
		else:
			self.use_image = False 
	def tune(self):
		# Init window
		cv2.namedWindow('Capture')
		
		# Add Sliders	
		def nothing(a):
			pass			
		cv2.createTrackbar('H', 'Capture', 0, 179, nothing)
		cv2.createTrackbar('S', 'Capture', 0, 255, nothing)
		cv2.createTrackbar('V', 'Capture', 0, 255, nothing)

		cv2.createTrackbar('HL', 'Capture', 0, 179, nothing)
		cv2.createTrackbar('SL', 'Capture', 0, 255, nothing)
		cv2.createTrackbar('VL', 'Capture', 0, 255, nothing)

		# Use image or video
		if not self.use_image:
			cap = cv2.VideoCapture(0)				
			while True:
				# Grab image and convert colorspace
				ret, frame = cap.read()	
				hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
				
				# Read limits from sliders and threshold
				UPPER_BOUNDS = np.array([cv2.getTrackbarPos('H', 'Capture'),cv2.getTrackbarPos('S', 'Capture'), 						cv2.getTrackbarPos('V', 'Capture')]) 
				LOWER_BOUNDS = np.array([cv2.getTrackbarPos('HL', 'Capture'),cv2.getTrackbarPos('SL', 'Capture'), 						cv2.getTrackbarPos('VL', 'Capture')]) 

				mask = cv2.inRange(hsv, LOWER_BOUNDS, UPPER_BOUNDS)
				hold = cv2.bitwise_and(hsv, hsv, mask = mask)
				
				# Convert back into RGB color space, then display
				color = cv2.cvtColor(hold, cv2.COLOR_HSV2BGR)
				cv2.imshow('Capture', color)
				
				# Break on ESC
				if cv2.waitKey(1) == 27:
					break	
			# Clean				
			cap.release()
			cv2.destroyAllWindows()
	
	def detect_cubes(self, LOWER_BOUNDS, UPPER_BOUNDS):
		cap = cv2.VideoCapture(0)
		while True:
			# Grab image and convert colorspace
			ret, frame = cap.read()
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			
			# Threshold			
			mask = cv2.inRange(hsv, LOWER_BOUNDS, UPPER_BOUNDS)
			hold = cv2.bitwise_and(hsv, hsv, mask = mask)
			
			# Find squares in the contours
			coords, img = self.find_cubes(hold)			

			# Convert back into RGB color space, then display
			if self.verbose:
				cv2.imshow('Capture', img)
			
			# Break on ESC
			if cv2.waitKey(1) == 27:
				break
		# Clean
		cap.release()
		cv2.destroyAllWindows()

	def find_cubes(self, hsv):
		# Expects image in HSV color space, with threshold applied
		# Find contours
		color= cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
		gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
		img, cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, 								   							cv2.CHAIN_APPROX_SIMPLE)		
		# Find COM for each contour
		coords = []
		for c in img:
			# Find moment of contour				
			M = cv2.moments(c)
			if M["m00"] > 0 and self.detect(c):  
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
			
				# Draw
				cv2.drawContours(color, [c], -1, (0, 255, 0), 2)
				cv2.putText(color, str(cX) + ',' + str(cY), (cX + 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 2)
				cv2.circle(color, (cX, cY), 7, (0, 255, 0), -1)
			
				coords.append([cX, cY])
	
		return coords, color

	def detect(self, c):
		peri = cv2.arcLength(c, False)
		#self.MIN_PERIM * 10	self.SQUARENESS/1000		
		if peri < 100:
			return False
		else:
			# Approximate number of sides			
			approx = cv2.approxPolyDP(c, .04* peri, True)
			return len(approx) == 4
	
	def tuna(self, LOWER_BOUNDS, UPPER_BOUNDS):
		''' Tune with square detection.
		'''
		# Init window
		cv2.namedWindow('Capture')
		
		# Add Sliders	
		def nothing(a):
			pass			
		cv2.createTrackbar('H', 'Capture', 0, 179, nothing)
		cv2.createTrackbar('S', 'Capture', 0, 255, nothing)
		cv2.createTrackbar('V', 'Capture', 0, 255, nothing)

		cv2.createTrackbar('HL', 'Capture', 0, 179, nothing)
		cv2.createTrackbar('SL', 'Capture', 0, 255, nothing)
		cv2.createTrackbar('VL', 'Capture', 0, 255, nothing)
		
		cv2.createTrackbar('Min Perim', 'Capture', 1, 500, nothing)
		cv2.createTrackbar('Squareness', 'Capture', 1, 1000, nothing)

		# Use image or video
		if not self.use_image:
			cap = cv2.VideoCapture(0)				
			while True:
				# Grab image and convert colorspace
				ret, frame = cap.read()	
				hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
				
				# Read limits from sliders and threshold
				UPPER_BOUNDS = np.array([cv2.getTrackbarPos('H', 'Capture'),cv2.getTrackbarPos('S', 'Capture'), 						cv2.getTrackbarPos('V', 'Capture')]) 
				LOWER_BOUNDS = np.array([cv2.getTrackbarPos('HL', 'Capture'),cv2.getTrackbarPos('SL', 'Capture'), 						cv2.getTrackbarPos('VL', 'Capture')]) 
				

				mask = cv2.inRange(hsv, LOWER_BOUNDS, UPPER_BOUNDS)
				hold = cv2.bitwise_and(hsv, hsv, mask = mask)
				
				# Adjsut metaparameters and find squares				
				self.MIN_PERIM = cv2.getTrackbarPos('Min Perim', 'Capture')
				self.SQUARENESS = cv2.getTrackbarPos('Squarness', 'Capture')				
				coords, color = self.find_cubes(hold)

				# Display with contours drawn
				cv2.imshow('Capture', color)
				
				# Break on ESC
				if cv2.waitKey(1) == 27:
					break	
			# Clean				
			cap.release()
			cv2.destroyAllWindows()		

if __name__ == '__main__':
	v = Vis(verbose = True)
	# Detect Cubes
	UPPER_BOUNDS = np.array([78, 177, 237])
	LOWER_BOUNDS = np.array([25, 120, 171])
	v.detect_cubes(LOWER_BOUNDS, UPPER_BOUNDS)
	#v.tuna(LOWER_BOUNDS, UPPER_BOUNDS)
