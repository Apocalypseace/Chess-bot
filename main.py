
import serial
import time
import chess.engine
from inference import get_model
import supervision as sv
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

arduino_port = 'COM7'  # Replace with your Arduino's serial port
baud_rate = 9600  # Ensure this matches the Arduino's baud rate
ser = serial.Serial(arduino_port, baud_rate, timeout=1)
# Set the API key (if necessary)
os.environ["ROBOFLOW_API_KEY"] = "enter your api"
cell_width, cell_height = 0, 0  # To store dimensions of each cell
x1, y1, x2, y2 = 0, 0, 0, 0     # To store bounding box coordinates of the chessboard
stockfish_path = "path to your stockfish"
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
# Function to map a point to its chessboard cell
def get_chessboard_cell(center_x, center_y, x1, y1, cell_width, cell_height):
    # Calculate the column and row indices
    col = (center_x - x1) // cell_width
    row = (center_y - y1) // cell_height

    # Ensure indices are within bounds
    if 0 <= col < 8 and 0 <= row < 8:
        return f"{columns[int(col)]}{rows[int(row)]}"
    return "Out of bounds"

# Open the webcam
cap = cv2.VideoCapture(0)
board = chess.Board()
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    # Capture a single frame
    ret, frame = cap.read()

    if ret:
        # Load a pre-trained YOLOv8n model using the API key
        model = get_model(model_id="chessboard-1hk4y/3", api_key=os.getenv("ROBOFLOW_API_KEY"))

        frame1 = frame.copy()

        # Run inference on the captured frame with a lower confidence threshold
        results = model.infer(frame, conf=0.1,iou=0.2)[0]

        # Load the results into the supervision Detections API
        detections = sv.Detections.from_inference(results)

        for detection in detections:
            a = detection[0]
            x1 = a[0].astype(int)
            y1 = a[1].astype(int)
            x2 = a[2].astype(int)
            y2 = a[3].astype(int)

            cell_x_coords = [0 for _ in range(9)]
            cell_y_coords = [0 for _ in range(9)]


            # Draw outer bounding box
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Calculate width and height for each cell in the 8x8 grid
            cell_width = (x2 - x1) // 8
            cell_height = (y2 - y1) // 8



            # Chessboard notation labels for an 8x8 board
            columns = "HGFEDCBA"
            rows = "87654321"

            # Draw grid lines and label each cell
            for i in range(9):
                for j in range(9):
                    # Calculate top-left corner of each cell
                    cell_x1 = x1 + j * cell_width
                    cell_y1 = y1 + i * cell_height
                    cell_x_coords[j] = cell_x1
                    cell_y_coords[i] = cell_y1




                    # Draw the cell label in the center of each cell
                    if i < 8 and j < 8:  # Ensure we're within bounds for columns and rows
                        label = f"{columns[j]}{rows[7 - i]}"
                        cv2.putText(frame1, label, (cell_x1 + cell_width // 2 - 10, cell_y1 + cell_height // 2 + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                    # Draw vertical and horizontal lines for the grid
                    if j < 8:  # Vertical lines
                        cv2.line(frame1, (cell_x1 + cell_width, y1), (cell_x1 + cell_width, y2), (255, 0, 0), 1)
                    if i < 8:  # Horizontal lines
                        cv2.line(frame1, (x1, cell_y1 + cell_height), (x2, cell_y1 + cell_height), (255, 0, 0), 1)

        # Annotate the image with inference results
        annotated_image = sv.BoundingBoxAnnotator().annotate(scene=frame, detections=detections)
        annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)

        # Display grid box window with labels
        #plt.figure("64-cell grid with Chess Notation")
        plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
       # plt.axis('off')

        # Display annotated image window
        #sv.plot_image(annotated_image)
        print(detections)
        print(cell_x_coords)
        print(cell_y_coords)

        plt.show()

    else:
        print("Error: Could not read frame.")

    # Release the capture object
    cap.release()

# Continue with the rest of your code...









#The below code runs the chess piece detection model




centers = []
detected=[]
flag=False
# Wait for user input before running the second model
#input("Press Enter to run the second model...")
move1=[]
while True:
    columns = "hgfedcba"
    rows = "12345678"
    key = input("enter to run model")
    if key != 'q':

        model = get_model(model_id="chess-tipra/1", api_key=os.getenv("ROBOFLOW_API_KEY"))

        # Initialize the webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        # Set the frame dimensions (optional, depends on your setup)
        frame_width = 470
        frame_height = 470
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        print("Starting detection... Press 'q' to quit.")

        # Start continuous detection for 3 seconds
        start_time = time.time()
        while time.time() - start_time < 3:  # Run for 3 seconds
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Perform inference on the frame with the Roboflow model
            results = model.infer(frame, conf=0.1)[0]

            # Convert results into a format usable by supervision
            detections = sv.Detections.from_inference(results)

            # Optionally annotate the frame with bounding boxes
            annotated_frame = sv.BoundingBoxAnnotator().annotate(scene=frame, detections=detections)

            # Display the annotated frame
            cv2.imshow('Continuous Detection', annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            k = 0
            for detection in detections.xyxy:
                # Extract bounding box coordinates
                x1 = int(detection[0])
                y1 = int(detection[1])
                x2 = int(detection[2])
                y2 = int(detection[3])

                # Calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centers.append((center_x, center_y))

                # Extract the piece label
                piece_label = detections.data['class_name'][k]

                # Ensure piece_label is an integer for comparison
                #try:
                    #if int(piece_label) in {"piece"}:
                        #centers.append((center_x, center_y))

                        # Draw bounding box and label on the frame
                        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        #label_text = f"Piece: {piece_label}"
                        #cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                #except ValueError:
                    #print(f"Invalid label encountered: {piece_label}")

                k += 1

                # Calculate the center of the bounding box

                found = False
        # Process the detected centers
        move = set()  # To avoid duplicates
        for center_x, center_y in centers:
            found = False
            for i in range(len(cell_x_coords) - 1):
                for j in range(len(cell_y_coords) - 1):
                    if center_x <= cell_x_coords[i + 1] and center_y <= cell_y_coords[j + 1]:
                        position = f"{columns[i]}{rows[j]}"
                        detected.append((position))
                        found = True
                        break
                if found:
                    break
        from collections import Counter

        counter = Counter(detected)
        # Convert the set back to a list
        mymov = []
        move = set(detected)
        move = {position for position in move if counter[position] >= 3}
        # Calculate the new moves
        mymov = [m for m in move if m not in move1]  # Moves from move not in move1
        removed_moves = [m for m in move1 if m not in move]  # Moves removed from move1

        # Print the board state and moves
        print("Board state:", move)
        print("Move1 (previous moves):", move1)
        print("Current moves:", mymov)  #ok
        print("Removed moves:", removed_moves)
        print(detections)
        print("count",counter)
        # Combine moves (removed and current)
        final_move = ''.join(removed_moves + mymov)
        move1 = move.copy() # Update move1 with the current moves
        move.clear()
        centers.clear()  # Reset centers
        detected.clear()
        # Clear the move list for the next iteration

        if flag and final_move:
            try:
                # Push the move to the board
                user_move = chess.Move.from_uci(final_move)
                if user_move in board.legal_moves:
                    board.push(user_move)
                    print("Move pushed to the board:", final_move)
                else:
                    print("Illegal move:", final_move)

                # Get Stockfish's best move
                stockfish_move = engine.play(board, chess.engine.Limit(time=0.1)).move
                print("Stockfish recommends move:", stockfish_move)
                #first_two_chars = str(stockfish_move)[:2]

                # Remove the first two characters from move1 if they exist
                #if first_two_chars in move1:
                    #move1.remove(first_two_chars)
                destination_square = str(stockfish_move)[2:]
                print("dest",destination_square)
                if destination_square  in move1:
                    move1.remove(destination_square)  # Add the destination square to move
                    print(f"Added {destination_square} to move1.")

                # Push Stockfish's move
                board.push(stockfish_move)

                source_square = str(stockfish_move)[:2]  # First two characters represent the source
                destination_square = str(stockfish_move)[2:]  # Last two characters represent the destination

                print("Source square:", source_square)
                print("Destination square:", destination_square)

                box_size = 5

                # Define the chessboard notation
                rows = ['8', '7', '6', '5', '4', '3', '2', '1']
                columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

                # The origin is between e8 and d8, so calculate the offset
                origin_x = (columns.index('e') + 0.5) * box_size
                origin_y = (rows.index('8') + 0.5) * box_size


                def get_coordinates(position):
                    col = position[0]
                    row = position[1]

                    if col in columns and row in rows:
                        # Calculate the x and y coordinates relative to the origin
                        y = (columns.index(col) + 0.5) * box_size - origin_x+1
                        x = (rows.index(row) + 0.5) * box_size - origin_y+1


                        return x, y
                    else:
                        return None


                # Input loop

                user_input = source_square

                if len(user_input) == 2 and user_input[0] in columns and user_input[1] in rows:
                    coordinates = get_coordinates(user_input)
                    if coordinates:
                        print(f"{user_input}: (x={coordinates[0]:.2f}, y={coordinates[1]:.2f})")
                    else:
                        print("Invalid position. Please try again.")
                else:
                    print("Invalid input. Please enter a valid chessboard position.")
                x = coordinates[0] + 11 # offset on how much the board is away
                y = coordinates[1]
                print(coordinates, x, y)

                # HERE LIES THE LENGTH OF THE LINKS!!!!

                z = 1.5 #height to grab the piece from
                link1= 21
                link2= 26+6  # gripper parameters added
                offset_on_y=9
                gnd_to_base=14.5

                th1 = np.degrees(np.arctan2(x, y))

                dist = np.sqrt(x ** 2 + y ** 2)

                base = np.sqrt((gnd_to_base - z) ** 2 + (dist ** 2))
                b = base

                th2 = np.degrees(np.arccos((link2 ** 2 - link1 ** 2 - b ** 2) / (2 * b * link1)))
                th3 = np.degrees(np.arccos((link1 ** 2 - link2 ** 2 - b ** 2) / (2 * b * link2)))

                p = np.degrees(np.arccos(((gnd_to_base - z) ** 2 - dist ** 2 - b ** 2) / (2 * b * dist)))
                th2 = p - th2
                th3 = th3 + p - 180 - th2

                offset1 = np.degrees(np.arccos((dist ** 2 - (dist ** 2 + offset_on_y ** 2) - offset_on_y ** 2) / (2 * (np.sqrt(dist ** 2 + offset_on_y ** 2)) * offset_on_y)))
                offset1 = offset1 - 90

                th1 = th1 - offset1
                print(dist)
                print('source theta 1:', th1)
                print('source theta 2:', th2)
                print('source theta 3:', th3)






                user_input = destination_square

                if len(user_input) == 2 and user_input[0] in columns and user_input[1] in rows:
                    coordinates = get_coordinates(user_input)
                    if coordinates:
                        print(f"{user_input}: (x={coordinates[0]:.2f}, y={coordinates[1]:.2f})")
                    else:
                        print("Invalid position. Please try again.")
                else:
                    print("Invalid input. Please enter a valid chessboard position.")
                x = coordinates[0] + 11
                y = coordinates[1]
                print(coordinates, x, y)


                theta1 = np.degrees(np.arctan2(x, y))

                dist = np.sqrt(x ** 2 + y ** 2)

                base = np.sqrt((gnd_to_base - z) ** 2 + (dist ** 2))
                b = base

                theta2 = np.degrees(np.arccos((link2 ** 2 - link1 ** 2 - b ** 2) / (2 * b * link1)))
                theta3 = np.degrees(np.arccos((link1 ** 2 - link2 ** 2 - b ** 2) / (2 * b * link2)))

                p = np.degrees(np.arccos(((gnd_to_base - z) ** 2 - dist ** 2 - b ** 2) / (2 * b * dist)))
                theta2 = p - theta2
                theta3 = theta3 + p - 180 - theta2

                offset1 = np.degrees(
                    np.arccos((dist ** 2 - (dist ** 2 + offset_on_y ** 2) - offset_on_y ** 2) / (2 * (np.sqrt(dist ** 2 + offset_on_y ** 2)) * offset_on_y)))
                offset1 = offset1 - 90

                theta1 = theta1 - offset1
                print(dist)
                print('destination theta 1:', theta1)
                print('destination theta 2:', theta2)
                print('destination theta 3:', theta3)


                def send_angles(th1, th2, th3, theta1, theta2, theta3):
                    """
                    Sends angle values to Arduino via serial.
                    """
                    try:
                        # Create a formatted string to send
                        data = f"{round(th1)},{round(th2)},{round(th3)},{round(theta1)},{round(theta2)},{round(theta3)}\n"
                        ser.write(data.encode())  # Send the data
                        print(f"Sent: {data.strip()}")
                        time.sleep(0.1)

                        # Short delay to allow Arduino to process the data
                    except Exception as e:
                        print(f"Error: {e}")

                send_angles(th1,th2,th3,theta1,theta2,theta3)


                # Print the updated board
                print(board)

            except Exception as e:
                print("An error occurred:", e)

        # Visualization with Matplotlib (optional)
        #plt.figure("64-cell grid")
        #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        #plt.axis('off')
        plt.pause(0.001)


        # Display the annotated image window (Optional)
        # sv.plot_image(annotated_image)  # Uncomment if you have a function for annotated_image
        flag = True

        # Release the capture object
        cap.release()
        cv2.destroyAllWindows()
