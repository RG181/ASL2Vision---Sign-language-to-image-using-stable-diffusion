import cv2
import tensorflow as tf
import numpy as np
import json
import requests
import threading
import os

# ── CONFIG ───────────────────────────────────────────────────
HF_TOKEN  = "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"   # ← your Hugging Face token
IMG_SIZE  = 224
BOX_SIZE  = 250
# ─────────────────────────────────────────────────────────────

print("Loading model...")
model = tf.keras.models.load_model("asl_final.keras")

with open("class_labels.json") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

print(f"Model loaded! {len(label_map)} classes ready.")
print("─────────────────────────────")
print("SPACE = add letter shown on screen")
print("ENTER = generate image for word")
print("B     = delete last letter")
print("C     = clear entire word")
print("Q     = quit")
print("─────────────────────────────\n")

# ── Image generation (background thread — no lag) ─────────────
def generate_image(word):
    def _generate():
        print(f"\nGenerating image for: '{word}'...")
        print("Image will open automatically when ready!")
        try:
            response = requests.post(
                "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={
                    "inputs": f"Simple colorful illustration of a {word}, plain white background, children's book style, clear bright colors"
                },
                timeout=60
            )

            if response.status_code in [503, 410]:
                print("Trying backup model...")
                response = requests.post(
                    "https://router.huggingface.co/hf-inference/models/runwayml/stable-diffusion-v1-5",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                    json={
                        "inputs": f"Simple illustration of a {word}, white background"
                    },
                    timeout=60
                )

            if response.status_code == 200:
                out_path = f"generated_{word}.png"
                with open(out_path, "wb") as f:
                    f.write(response.content)
                print(f"Done! Image saved as: generated_{word}.png")
                os.startfile(out_path)   # Opens in default image viewer
            elif response.status_code == 401:
                print("Invalid Hugging Face token!")
            elif response.status_code == 503:
                print("Model loading — wait 30 seconds and press ENTER again.")
            else:
                print(f"Error {response.status_code}: {response.text[:200]}")

        except requests.exceptions.Timeout:
            print("Timed out — try again.")
        except Exception as e:
            print(f"Error: {e}")

    # Run in background — camera never freezes
    t = threading.Thread(target=_generate)
    t.daemon = True
    t.start()

# ── Predict sign ──────────────────────────────────────────────
def predict_sign(frame, x1, y1, x2, y2):
    cropped = frame[y1:y2, x1:x2]
    img     = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    img     = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    arr     = np.expand_dims(img.astype("float32"), axis=0)
    preds   = model.predict(arr, verbose=0)[0]
    top3    = np.argsort(preds)[::-1][:3]
    return [(label_map[i], float(preds[i]) * 100) for i in top3]

# ── Hand detection ────────────────────────────────────────────
def hand_in_box(frame, x1, y1, x2, y2):
    roi   = frame[y1:y2, x1:x2]
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0,   20, 70]),
                             np.array([20,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 20, 70]),
                             np.array([180, 255, 255]))
    mask  = cv2.bitwise_or(mask1, mask2)
    return np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) > 0.08

# ── Camera ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found! Try VideoCapture(1)")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)

# ── State ─────────────────────────────────────────────────────
current_word  = ""
last_preds    = []
flash_counter = 0
frame_count   = 0
live_label    = "---"
live_conf     = 0.0
PREDICT_EVERY = 30      # Predict every 30 frames — smooth camera

print("Camera open!")
print("TIP: Use plain white wall as background for best accuracy!\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Box coordinates
    x1 = max(0, w // 2 - BOX_SIZE // 2)
    y1 = max(0, h // 2 - BOX_SIZE // 2)
    x2 = min(w, x1 + BOX_SIZE)
    y2 = min(h, y1 + BOX_SIZE)

    frame_count += 1

    # Update live prediction every 30 frames
    if frame_count % PREDICT_EVERY == 0:
        try:
            preds      = predict_sign(frame, x1, y1, x2, y2)
            live_label = preds[0][0]
            live_conf  = preds[0][1]
        except:
            pass

    detected = hand_in_box(frame, x1, y1, x2, y2)

    # ── Box color ─────────────────────────────────────────────
    if flash_counter > 0:
        box_color = (0, 0, 255)      # Red — just captured
    elif detected:
        box_color = (0, 255, 0)      # Green — hand detected
    else:
        box_color = (0, 165, 255)    # Orange — no hand

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Corner markers
    c = 20
    for cx, cy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
        dx = c if cx == x1 else -c
        dy = c if cy == y1 else -c
        cv2.line(frame, (cx, cy), (cx+dx, cy), box_color, 3)
        cv2.line(frame, (cx, cy), (cx, cy+dy), box_color, 3)

    # ── Status above box ──────────────────────────────────────
    if detected:
        cv2.putText(frame,
                    "Hand detected — SPACE to add letter",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    else:
        cv2.putText(frame,
                    "Show hand inside the box",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    # ── Live prediction below box — big and clear ─────────────
    conf_color = (0, 255, 0)   if live_conf > 65 else \
                 (0, 255, 255) if live_conf > 40 else \
                 (0, 165, 255)

    # Big letter shown clearly
    cv2.putText(frame,
                f"{live_label}",
                (x1, y2 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, conf_color, 3)

    cv2.putText(frame,
                f"{live_conf:.0f}%",
                (x1 + 60, y2 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, conf_color, 2)

    cv2.putText(frame,
                "Press SPACE to add this letter",
                (x1, y2 + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)

    # ── Word display — top of screen ──────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 42), (0, 0, 0), -1)
    word_show = current_word if current_word else "_ _ _"
    cv2.putText(frame,
                f"Word: {word_show}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # ENTER hint when word has letters
    if len(current_word) >= 2:
        cv2.putText(frame,
                    "ENTER = generate image",
                    (w - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ── Flash — shows added letter ────────────────────────────
    if flash_counter > 0:
        cv2.putText(frame,
                    f"Added: {live_label}",
                    (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        flash_counter -= 1

    # ── Bottom instructions bar ───────────────────────────────
    cv2.rectangle(frame, (0, h - 25), (w, h), (0, 0, 0), -1)
    cv2.putText(frame,
                "SPACE=Add letter   ENTER=Generate image   B=Delete   C=Clear   Q=Quit",
                (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    cv2.imshow("ASL Sign Language Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    # Q = quit
    if key == ord('q') or key == ord('Q'):
        print("Closing...")
        break

    # SPACE = add EXACTLY the letter showing on screen
    elif key == ord(' '):
        if not detected:
            print("No hand in box! Show your hand first.")
        elif live_label == "---" or live_conf == 0.0:
            print("Wait for a letter to appear on screen first!")
        else:
            # ── Uses live_label directly — same as what screen shows ──
            letter        = live_label
            flash_counter = 25

            if letter.upper() not in ['SPACE', 'DELETE', 'NOTHING', 'DEL']:
                current_word += letter.upper()
                print(f"Added  : {letter.upper()}")
                print(f"Word   : {current_word}")
            elif letter.upper() == 'SPACE':
                current_word += ' '
                print("Space added")
            elif letter.upper() in ['DELETE', 'DEL']:
                current_word = current_word[:-1]
                print(f"Deleted. Word: {current_word}")

    # ENTER = generate image for full word
    elif key == 13:
        if current_word.strip():
            generate_image(current_word.strip().lower())
        else:
            print("Spell a word first using hand signs!")

    # B = delete last letter
    elif key == ord('b') or key == ord('B'):
        if current_word:
            removed      = current_word[-1]
            current_word = current_word[:-1]
            print(f"Deleted '{removed}'. Word: '{current_word}'")
        else:
            print("Nothing to delete!")

    # C = clear word
    elif key == ord('c') or key == ord('C'):
        print(f"Cleared: '{current_word}'")
        current_word  = ""
        last_preds    = []
        live_label    = "---"
        live_conf     = 0.0

cap.release()
cv2.destroyAllWindows()
print("App closed!")