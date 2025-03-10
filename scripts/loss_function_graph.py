import pickle
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model from a pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand solution and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the MediaPipe Hands object
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map model predictions to labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'L'}
target_label = 0  # Target class index for "A"

loss_values = []

print("Press 'q' to stop capturing and plot the loss graph.")

while True:
    data_aux = []  # Auxiliary data for model prediction
    x_ = []  # List to store x-coordinates of landmarks
    y_ = []  # List to store y-coordinates of landmarks

    # Capture a frame from the camera
    ret, frame = cap.read()

    # Get the dimensions of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Extract landmark coordinates
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize landmark coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Ensure data_aux has the correct number of features
        if len(data_aux) == 42:
            data_aux = np.concatenate([data_aux, data_aux])  # Duplicate the features to match the expected input size

        # Predict the probabilities using the model
        prediction_probabilities = model.predict_proba([np.asarray(data_aux)])[0]

        # Calculate the loss for the target label
        real_value = 1 if target_label == np.argmax(prediction_probabilities) else 0
        predicted_value = prediction_probabilities[target_label]
        loss = (real_value - predicted_value) ** 2
        loss_values.append(loss)

        # Display information on the frame
        cv2.putText(frame, f"Loss: {loss:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Plot the loss graph
plt.figure(figsize=(10, 6))
plt.plot(loss_values, color='gray', label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Function Over Iterations')
plt.legend()
plt.grid(True)
plt.show()
