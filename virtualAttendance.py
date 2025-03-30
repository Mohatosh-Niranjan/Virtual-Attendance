import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime

# Step 1: Database Setup
def setup_database():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        roll_number TEXT NOT NULL UNIQUE
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    conn.commit()
    conn.close()

# Step 2: Train the LBPHFaceRecognizer Model with All Users
def train_model_with_all_users():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = []
    labels = []

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, roll_number FROM users')
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        user_id, roll_number = row
        user_folder = os.path.join("users", roll_number)

        if os.path.exists(user_folder):
            for image_file in os.listdir(user_folder):
                image_path = os.path.join(user_folder, image_file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in detected_faces:
                    face_roi = img[y:y+h, x:x+w]
                    faces.append(face_roi)
                    labels.append(user_id)

    if not faces or not labels:
        print("No faces or labels found for training. Aborting.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save("trained_model.yml")
    print("Model trained successfully with all users.")

# Step 3: Register New User
def register_user(name, roll_number):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    # Create the "users" directory if it doesn't exist
    if not os.path.exists("users"):
        os.makedirs("users")

    # Create a subfolder for the user using their roll number
    user_folder = os.path.join("users", roll_number)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    print("Capturing 10 images for training. Please stay still...")
    faces = []
    labels = []

    # Assign a unique label to the user based on their database ID
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (name, roll_number) VALUES (?, ?)', (name, roll_number))
        conn.commit()
        # Get the ID of the newly registered user
        cursor.execute('SELECT id FROM users WHERE roll_number = ?', (roll_number,))
        user_id = cursor.fetchone()[0]
    except sqlite3.IntegrityError:
        print(f"Roll number {roll_number} already exists. Please use a unique roll number.")
        conn.close()
        return

    count = 0
    while count < 10:  # Capture 10 images
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            face_roi = gray[y:y+h, x:x+w]
            faces.append(face_roi)
            labels.append(user_id)  # Use the user's database ID as the label

            # Save the captured image to the user's folder
            image_path = os.path.join(user_folder, f"image_{count+1}.jpg")
            cv2.imwrite(image_path, face_roi)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured {count+1}/10", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            count += 1
            cv2.imshow("Registration", frame)
            cv2.waitKey(500)  # Wait 500ms between captures

        cv2.imshow("Registration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

    # Train the model with all registered users' data
    train_model_with_all_users()

# Step 4: Face Recognition and Attendance Logging
def log_attendance():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trained_model.yml")

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # Load user details from the database
    cursor.execute('SELECT id, name, roll_number FROM users')
    rows = cursor.fetchall()
    label_dict = {row[0]: (row[1], row[2]) for row in rows}

    if not label_dict:
        print("No users found in the database. Please register users first.")
        return

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_roi)

            name, roll_number = "Unknown", "N/A"
            if confidence < 70:  # Confidence threshold (lower is better)
                user_details = label_dict.get(label)
                if user_details:
                    name, roll_number = user_details

            # Check if attendance is already logged for today
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('SELECT * FROM attendance WHERE user_id = ? AND date = ?', (label, today))
            existing_record = cursor.fetchone()

            # Determine the message to display
            if name != "Unknown":
                if not existing_record:
                    # Log attendance
                    cursor.execute('INSERT INTO attendance (user_id, date) VALUES (?, ?)', (label, today))
                    conn.commit()
                    message = f"Logged: {name} ({roll_number})"
                    color = (0, 255, 0)  # Green for success
                else:
                    message = f"Already Logged: {name} ({roll_number})"
                    color = (0, 0, 255)  # Red for already logged
            else:
                message = "Unknown User"
                color = (0, 0, 255)  # Red for unknown

            # Display the message on the camera feed
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, message, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the camera feed
        cv2.imshow("Attendance System", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    conn.close()
    cv2.destroyAllWindows()

# Step 5: View Attendance Logs
def view_attendance():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT users.name, users.roll_number, attendance.date FROM attendance
    JOIN users ON attendance.user_id = users.id
    ORDER BY attendance.date DESC
    ''')
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No attendance logs found.")
    else:
        print("\nAttendance Logs:")
        for row in rows:
            print(f"Name: {row[0]}, Roll No: {row[1]}, Date: {row[2]}")

# Main Menu
if __name__ == "__main__":
    setup_database()  # Initialize the database

    while True:
        print("\n--- Attendance System ---")
        print("1. Register New User")
        print("2. Start Attendance Logging")
        print("3. View Attendance Logs")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            name = input("Enter the name of the user to register: ")
            roll_number = input("Enter the roll number of the user: ")
            register_user(name, roll_number)
        elif choice == "2":
            log_attendance()
        elif choice == "3":
            view_attendance()
        elif choice == "4":
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")