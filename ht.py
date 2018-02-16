'''
	Brandon Wood
	FRC 334
	Power Cube Detection
	2/11/18
	
	Power Cube detector for FRC 2018: Power Up using Hough transform
'''
import cv2
import numpy as np
import math

class HT:
	def __init__(self, verbose = False, show_houghs = False):
		# Storage			
		self.RHO_ACC = 400 # Scaled by 1/10
		self.THETA_ACC = 2 # inputs as np.pi/THETA_ACC 		
		self.MIN_LENGTH = 200
		self.THETA_THRESH = 10 # Number of degrees from horizontal lines are called horizontal
		self.NUM_CUBES = 6
		self.NUM_ITERS = 1
		self.EPSILON = 500 # Scaled by 1/1000
	
		# Settings
		self.verbose = verbose
		self.show_houghs = show_houghs
	
	# Tuning
	def tune_hough(self, LOWER_BOUNDS, UPPER_BOUNDS):
		# Init window
		cv2.namedWindow('Capture')
		
		# Add Sliders	
		def nothing(a):
			pass

		# Make trackbars
		cv2.createTrackbar('RHO_ACC', 'Capture', 1, 100, nothing)
		cv2.createTrackbar('THETA_ACC', 'Capture', 1, 20, nothing)
		cv2.createTrackbar('MIN_LENGTH', 'Capture', 1, 255, nothing)



		# Grab them videos
		cap = cv2.VideoCapture(0)				
		while True:
			# Grab image and convert colorspace
			ret, frame = cap.read()	
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

			# Threshold
			mask = cv2.inRange(hsv, LOWER_BOUNDS, UPPER_BOUNDS)
			hold = cv2.bitwise_and(hsv, hsv, mask = mask)
			
			# Get Hyperparameters from sliders and run
			self.RHO_ACC = cv2.getTrackbarPos('RHO_ACC', 'Capture')
			self.THETA_ACC = cv2.getTrackbarPos('THETA_ACC', 'Capture')
			self.MIN_LENGTH = cv2.getTrackbarPos('MIN_LENGTH', 'Capture')
				
			# Run Hough's transform and find lines
			before = self.show_houghs
			self.show_houghs = True

			coords, img = self.find_houghs(hold)
			cv2.imshow('Capture', img)
			
			self.show_houghs = before

			# Break on ESC
			if cv2.waitKey(1) == 27:
				break	
		# Clean				
		cap.release()
		cv2.destroyAllWindows()

	def tune_kmeans(self, LOWER_BOUNDS, UPPER_BOUNDS):
		# Init window
		cv2.namedWindow('Capture')
		
		# Add Sliders	
		def nothing(a):
			pass			

		cv2.createTrackbar('THETA_THRESHHL', 'Capture', 1, 179, nothing)
		cv2.createTrackbar('NUM_CUBES', 'Capture', 1, 12, nothing)
		cv2.createTrackbar('NUM_ITERS', 'Capture', 1, 5, nothing)
		cv2.createTrackbar('EPSILON', 'Capture', 1, 1000, nothing)


		# Grab them videos
		cap = cv2.VideoCapture(0)				
		while True:
			# Grab image and convert colorspace
			ret, frame = cap.read()	
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

			# Threshold
			mask = cv2.inRange(hsv, LOWER_BOUNDS, UPPER_BOUNDS)
			hold = cv2.bitwise_and(hsv, hsv, mask = mask)
			
			# Get Hyperparameters from sliders and run

			self.THETA_THRESH = cv2.getTrackbarPos('THETA_THRESH', 'Capture')
			self.NUM_CUBES = cv2.getTrackbarPos('NUM_CUBES', 'Capture')
			self.NUM_ITERS = cv2.getTrackbarPos('NUM_ITERS', 'Capture')
			self.EPSILON = cv2.getTrackbarPos('EPSILON', 'Capture')
			
			# Run Hough's transform and find points
			coords, img = self.find_houghs(hold)
			cv2.imshow('Capture', img)
			
			# Break on ESC
			if cv2.waitKey(1) == 27:
				break	
		# Clean				
		cap.release()
		cv2.destroyAllWindows()

	# Robot Methods	
	def detect_cubes_with_houghs(self, LOWER_BOUNDS, UPPER_BOUNDS):
		cap = cv2.VideoCapture(0)
		while True:
			# Grab image and convert colorspace
			ret, frame = cap.read()
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			
			# Threshold			
			mask = cv2.inRange(hsv, LOWER_BOUNDS, UPPER_BOUNDS)
			hold = cv2.bitwise_and(hsv, hsv, mask = mask)
			
			# Find squares in the contours
			coords, img = self.find_houghs(hold)			

			# Convert back into RGB color space, then display
			if self.verbose:
				cv2.imshow('Capture', img)
			
			# Break on ESC
			if cv2.waitKey(1) == 27:
				break
		# Clean
		cap.release()
		cv2.destroyAllWindows()


	# Utility Methods
	def find_houghs(self, img):
		# Cast to 1-Channel
		color = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
		gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
		
		# Run Canny edge detection
		#edgy = cv2.Canny(gray, 100, 200) 		

		# Compute and Draw Hough Lines
		coords = []
		lines = cv2.HoughLines(gray, self.RHO_ACC * .1, np.pi/self.THETA_ACC, self.MIN_LENGTH)
		if type(lines) != type(None):
			if self.show_houghs:
				for rho, theta in lines[0]:
					print(rho, theta)
					a = np.cos(theta)
					b = np.sin(theta)
					x0 = a*rho
					y0 = b*rho
					x1 = int(x0 + 1000*(-b))
					y1 = int(y0 + 1000*(a))
					x2 = int(x0 - 1000*(-b))
					y2 = int(y0 - 1000*(a))			
				
					cv2.line(color,(x1,y1),(x2,y2),(0,0,255),2)
				print('--------------------')		
			else:
				hits = self.find_hits(lines)
				coords = self.collect_points(hits)				
				for point in coords:
					if point[0]> 0 and point[1] > 0:
						cv2.circle(color,(point[0],point[1]), 5, (0, 255, 0), -1)
						cv2.putText(color, str(int(point[0])) + ', ' + str(int(point[1])), (int(point[0]) + 10, int(point[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 2)					

		return coords, color

	def find_hits(self, lines):
		horz = []			
		vert = []
		for rho, theta in lines[0]:
			if theta < np.pi/180*self.THETA_THRESH or theta > np.pi/180*(180-self.THETA_THRESH):
				horz.append(rho)
			else:
				vert.append(rho)
		hits = []

		for x in horz:
			for y in vert:
				hits.append([x, y])			


		return np.array(hits, dtype = np.float32)
		
	def collect_points(self, hits):
		centers = [[0, 0]]		
		if len(hits) > 0:
			compactness, labels, centers = cv2.kmeans(hits, int(math.ceil(len(hits)/self.NUM_CUBES)) + 1 , (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.NUM_ITERS, self.EPSILON*.001), 2, cv2.KMEANS_RANDOM_CENTERS)
		return centers

if __name__ == '__main__':
	v = HT(verbose = True, show_houghs = True)
	# Detect Cubes
	UPPER_BOUNDS = np.array([47, 140, 255])
	LOWER_BOUNDS = np.array([20, 81, 118])
	#v.detect_cubes_with_houghs(LOWER_BOUNDS, UPPER_BOUNDS)
	v.tune_hough(LOWER_BOUNDS, UPPER_BOUNDS)
