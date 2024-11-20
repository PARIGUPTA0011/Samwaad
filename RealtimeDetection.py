from keras.models import model_from_json
import cv2
import numpy as np
import sys

# Set default encoding to UTF-8 to avoid UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

# Load the model from JSON and weights
json_file = open("signlanguagedetectionmodel48x48.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagedetectionmodel48x48.h5")

# Function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape to match model input
    return feature / 255.0  # Normalize the image

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Define the labels for the 26 classes (A-Z)
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

while True:
    # Capture frame-by-frame
    _, frame = cap.read()
    
    # Draw a rectangle to mark the region of interest (ROI) for sign language detection
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    
    # Crop the frame to the ROI (from 40 to 300 in both directions)
    cropframe = frame[40:300, 0:300]
    
    # Convert the cropped frame to grayscale
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    
    # Resize the cropped frame to match the input shape of the model
    cropframe = cv2.resize(cropframe, (48, 48))
    
    # Extract features from the preprocessed image
    cropframe = extract_features(cropframe)
    
    # Make predictions with the model
    pred = model.predict(cropframe)
    
    # Get the class label with the highest probability
    prediction_label = label[pred.argmax()]
    
    # Display a rectangle on top of the frame to show the prediction result
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    
    # Show the prediction label and the accuracy
    accu = "{:.2f}".format(np.max(pred) * 100)  # Get the accuracy in percentage
    cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show the frame with the prediction and the accuracy
    cv2.imshow("output", frame)
    
    # Break the loop if the user presses 'ESC' key (key code 27)
    if cv2.waitKey(27) & 0xFF == 27:
        break

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
