import cv2

# Load pre-trained pedestrian detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Start camera
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale for lane detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection (Lane detection)
    edges = cv2.Canny(gray, 50, 150)

    # Hough Transform for lane lines
    lines = cv2.HoughLinesP(edges, 1, 3.14/180, 100, minLineLength=100, maxLineGap=50)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Pedestrian detection
    boxes, _ = hog.detectMultiScale(frame, winStride=(8,8))

    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "Pedestrian", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # Show output
    cv2.imshow("In-Vehicle Vision System 😏", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()