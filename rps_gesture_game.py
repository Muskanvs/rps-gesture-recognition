import cv2
import mediapipe as mp
import random
import time

# Initialize Mediapipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Game variables
moves = ['rock', 'paper', 'scissors']
user_move = None
comp_move = None
result = ""
score_user = 0
score_comp = 0
start_time = None

# Helper function to count which fingers are up
def get_finger_status(hand_landmarks):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    
    # Get all landmark positions
    lm_list = []
    for id, lm in enumerate(hand_landmarks.landmark):
        h, w = 480, 640
        lm_list.append((int(lm.x * w), int(lm.y * h)))

    # Thumb (special case: x comparison)
    if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers (y comparison)
    for id in range(1, 5):
        if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

# Convert finger status to gesture
def get_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "rock"
    elif fingers == [1, 1, 1, 1, 1]:
        return "paper"
    elif fingers == [0, 1, 1, 0, 0]:
        return "scissors"
    else:
        return "unknown"

# Game loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(img_rgb)

    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = get_finger_status(hand_landmarks)
            user_move = get_gesture(fingers)

            if start_time is None:
                start_time = time.time()

            if time.time() - start_time > 3:
                comp_move = random.choice(moves)

                # Determine winner
                if user_move == comp_move:
                    result = "Draw"
                elif (user_move == "rock" and comp_move == "scissors") or \
                     (user_move == "scissors" and comp_move == "paper") or \
                     (user_move == "paper" and comp_move == "rock"):
                    result = "You Win!"
                    score_user += 1
                else:
                    result = "Computer Wins!"
                    score_comp += 1

                start_time = None

    # Display game info
    cv2.putText(img, f"Your Move: {user_move}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(img, f"Computer: {comp_move}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    cv2.putText(img, f"Result: {result}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(img, f"Score - You: {score_user} | Comp: {score_comp}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Rock Paper Scissors - Rindo's Project", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
