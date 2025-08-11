#  ------------------------------------------

import mysql.connector
import cv2
import datetime
import face_recognition
import os

# --------------------------------------

criminal_image = r"D:\Criminal_detection_project\criminal_image"

def insert_criminal_data():

    conn = None
    cursor = None

    try:

        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root@123",
            database='criminal_database',
            port = 3306
        )

        if conn.is_connected():
            print("Database connected successfully.")


            cursor = conn.cursor()


            criminal_id = input("Enter Criminal ID: ").strip()
            criminal_name = input("Enter Criminal Name: ").strip()
            crime_description = input("Enter Crime Description: ").strip()
            time_of_attempting_crime = input("Enter Time of Attempting Crime (YYYY-MM-DD HH:MM:SS): ").strip()
            location_of_crime = input("Enter Location of Crime: ").strip()


            try:
                datetime.datetime.strptime(time_of_attempting_crime, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                print("Invalid datetime format. Please use YYYY-MM-DD HH:MM:SS.")
                return None


            insert_query = '''
            INSERT INTO criminals (criminal_id, criminal_name, crime_description, time_of_attempting_crime, location_of_crime)
            VALUES (%s, %s, %s, %s, %s)
            '''


            cursor.execute(insert_query, (criminal_id, criminal_name, crime_description, time_of_attempting_crime, location_of_crime))


            conn.commit()
            print("Criminal data inserted successfully!")
            return criminal_id

    except mysql.connector.Error as err:
        print(f"Database error: {err}")

    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

    finally:

        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("Database connection closed.")

    return None

# ----------------------------------------------------------------------

def capture_criminal_image(criminal_id):

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'Space' to capture the image and 'Esc' to exit.")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break


        cv2.imshow("Capture Criminal Image", frame)


        key = cv2.waitKey(1) & 0xFF
        if key == 32:

            os.makedirs(criminal_image, exist_ok=True)


            image_path = os.path.join(criminal_image, f"{criminal_id}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            break
        elif key == 27:
            print("Exiting without capturing an image.")
            break


    camera.release()
    cv2.destroyAllWindows()

#----------------------------------------------

def scan_for_criminals():

    known_face_encodings = []
    known_criminal_ids = []

    print("Loading criminal images from storage...")

    for file in os.listdir(criminal_image):
        if not file.endswith(".jpg"):
            continue

        try:
            img_path = os.path.join(criminal_image, file)
            img = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(img)

            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                criminal_id = os.path.splitext(file)[0]
                known_criminal_ids.append(criminal_id)
        except Exception as e:
            print(f"Error processing image {file}: {e}")

    if not known_face_encodings:
        print("No criminal images found or unable to encode faces.")
        return

    print("Criminal images loaded successfully. Starting face scanning...")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Scanning for criminals... Press 'Esc' to exit.")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()

                if matches[best_match_index]:  # If a match is found
                    criminal_id = known_criminal_ids[best_match_index]
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Criminal with ID '{criminal_id}' found in Pune at {current_time}")
                    break  # Stop scanning further for matches

        cv2.imshow("Scanning for Criminals", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key
            print("Exiting...")
            break

    camera.release()
    cv2.destroyAllWindows()

# --------------------------------------------------

def main():
    while True:
        print("\n--- Criminal Detection System ---")
        print("1. Insert Criminal Data & Capture Image")
        print("2. Scan for Criminals")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":
            criminal_id = insert_criminal_data()
            if criminal_id:
                capture_criminal_image(criminal_id)
            else:
                print("Failed to insert criminal data. Image capture skipped.")

        elif choice == "2":
            scan_for_criminals()

        elif choice == "3":
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

