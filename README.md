# 🎮 Rock-Paper-Scissors Gesture Recognition

A fun and interactive real-time Rock-Paper-Scissors game that uses your **hand gestures** to play against the computer! This project leverages **OpenCV** and **MediaPipe** to recognize your hand pose via your webcam and determine your move.

---

## 📌 What is this Game?

Just like the classic game Rock-Paper-Scissors:

- ✊ **Rock** beats ✌️ Scissors
- ✋ **Paper** beats ✊ Rock
- ✌️ **Scissors** beats ✋ Paper

But here, **you play using your hand gestures**, and the computer randomly selects its move.

---

## 🧠 How the Game Works

1. The webcam captures your hand in real-time.
2. **MediaPipe** identifies key hand landmarks (fingertips and joints).
3. Based on finger positions, the system classifies your gesture as Rock, Paper, or Scissors.
4. The computer randomly picks its own move.
5. The winner is determined, the result is shown, and the score is updated live.

---

## 🎯 Features

- 🖐️ Gesture recognition using hand tracking.
- 🧠 Real-time classification of Rock, Paper, and Scissors.
- 🎮 Live gameplay with score updates.
- ⚡ Fast and responsive performance with webcam input.


##Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe**
- **NumPy**
