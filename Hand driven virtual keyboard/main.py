import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define keyboard layout
keys = [['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', '<'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M', 'Enter', '>'],
        ['CapsLock', 'Space', 'Backspace']]

# Key dimensions
key_width = 60
key_height = 60
start_x = 650
start_y = 320

# Special key widths
special_keys = {'Space': 300, 'Backspace': 180, 'CapsLock': 120, 'Enter': 120, '>': 60, '<': 60}

# State variables
hover_start_time = 0
hovered_key = None
typed_text = ""
debounce_time = 1.0  # seconds
caps_lock = False
cursor_pos = 0  # Cursor position in text

def draw_keyboard(img):
    for i, row in enumerate(keys):
        x_offset = start_x
        for key in row:
            width = special_keys.get(key, key_width)
            x = x_offset
            y = start_y + i * key_height
            cv2.rectangle(img, (x, y), (x + width, y + key_height), (0, 0, 0), 2)

            # Label adjustment
            if key == 'Space':
                label = 'Space'
                text_x = x + 110
            elif key == 'Backspace':
                label = 'Backspace'
                text_x = x + 10
            elif key == 'CapsLock':
                label = 'CAPS' if caps_lock else 'Caps'
                text_x = x + 10
            elif key == 'Enter':
                label = 'Enter'
                text_x = x + 10
            elif key in ['<', '>']:
                label = key
                text_x = x + 20
            else:
                label = key
                text_x = x + 10

            cv2.putText(img, label, (text_x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            x_offset += width

def get_key_under_finger(finger_x, finger_y):
    for i, row in enumerate(keys):
        x_offset = start_x
        for key in row:
            width = special_keys.get(key, key_width)
            x = x_offset
            y = start_y + i * key_height
            if x < finger_x < x + width and y < finger_y < y + key_height:
                return key
            x_offset += width
    return None

def draw_text_box(img, text, cursor_pos):
    lines = text[:cursor_pos].split('\n')
    current_line = len(lines) - 1
    cursor_index = len(lines[-1])

    lines = text.split('\n')
    box_height = 40 * len(lines) + 40
    cv2.rectangle(img, (100, 100), (900, 100 + box_height), (0, 0, 255), 5)

    y = 140
    for i, line in enumerate(lines):
        if i == current_line:
            line_to_draw = line[:cursor_index] + ('|' if int(time.time() * 2) % 2 == 0 else ' ') + line[cursor_index:]
        else:
            line_to_draw = line
        cv2.putText(img, line_to_draw, (110, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        y += 40

def type_in_application(key):
    global typed_text, cursor_pos, caps_lock
    if key == 'Space':
        typed_text = typed_text[:cursor_pos] + ' ' + typed_text[cursor_pos:]
        cursor_pos += 1
    elif key == 'Backspace':
        if cursor_pos > 0:
            typed_text = typed_text[:cursor_pos - 1] + typed_text[cursor_pos:]
            cursor_pos -= 1
    elif key == 'CapsLock':
        caps_lock = not caps_lock
    elif key == 'Enter':
        typed_text = typed_text[:cursor_pos] + '\n' + typed_text[cursor_pos:]
        cursor_pos += 1
    elif key == '>':
        if cursor_pos < len(typed_text):
            cursor_pos += 1
    elif key == '<':
        if cursor_pos > 0:
            cursor_pos -= 1
    else:
        letter = key.upper() if caps_lock else key.lower()
        typed_text = typed_text[:cursor_pos] + letter + typed_text[cursor_pos:]
        cursor_pos += 1

# Start webcam and set resolution
desired_width = 1400
desired_height = 1080
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Confirm actual resolution
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Camera resolution: {actual_width}x{actual_height}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Optional: force resize if camera didn't match requested resolution
    if (actual_width, actual_height) != (desired_width, desired_height):
        frame = cv2.resize(frame, (desired_width, desired_height))

    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    index_finger_pos = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_tip = int(hand_landmarks.landmark[8].x * w)
            y_tip = int(hand_landmarks.landmark[8].y * h)
            x_pip = int(hand_landmarks.landmark[6].x * w)
            y_pip = int(hand_landmarks.landmark[6].y * h)

            index_finger_pos = (x_tip, y_tip)

            if y_tip < y_pip:
                key = get_key_under_finger(x_tip, y_tip)
                if key:
                    if key == hovered_key:
                        if time.time() - hover_start_time > debounce_time:
                            type_in_application(key)
                            hover_start_time = time.time()
                    else:
                        hovered_key = key
                        hover_start_time = time.time()
                else:
                    hovered_key = None
                    hover_start_time = time.time()

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    draw_keyboard(frame)
    draw_text_box(frame, typed_text, cursor_pos)

    if index_finger_pos and hovered_key:
        for i, row in enumerate(keys):
            x_offset = start_x
            for key in row:
                width = special_keys.get(key, key_width)
                if key == hovered_key:
                    x = x_offset
                    y = start_y + i * key_height
                    cv2.rectangle(frame, (x, y), (x + width, y + key_height), (0, 0, 255), 3)
                x_offset += width

    cv2.imshow("Gesture Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()