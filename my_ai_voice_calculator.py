import json
import queue
import re
import threading
import time
import os
import sys
import numpy as np

import sounddevice as sd              # For capturing audio from microphone in real time
from vosk import Model, KaldiRecognizer  # Offline speech recognition engine (Vosk)
import pyttsx3                        # Offline text-to-speech engine
import tkinter as tk                  # Standard Python GUI toolkit
from tkinter import ttk, messagebox   # Themed widgets + message dialog helpers
from word2number import w2n           # Convert spoken number words (“twenty five”) to integers (25)

# ============================================================
# ====================== CONFIG SECTION =======================
# ============================================================

# Audio sampling configuration
SAMPLE_RATE = 16000          # Samples per second for the microphone input (16kHz is common for speech)
BLOCK_SIZE = 8000            # Number of samples processed per callback (roughly 0.5s at 16kHz)

# Wake words required to trigger the calculation
WAKE_WORDS = ["calculator", "calculate"]

# Recording/recognition behavior
MAX_LISTEN_SECONDS = 30      # Maximum time to keep listening for a phrase before timeout
SILENCE_THRESHOLD = 2.5      # If this many seconds of silence occur after speech, we stop listening
MAX_RESULT_ABS = 1e15        # Safety limit: reject results with absolute value larger than this

# Audio filtering / noise reduction configuration
NOISE_REDUCTION_ENABLED = True
HIGH_PASS_FREQ = 80          # Placeholder for future improvement (not used directly in this code)
LOW_PASS_FREQ = 8000         # Placeholder for future improvement (not used directly in this code)
VOLUME_THRESHOLD = 500       # Minimum RMS level to treat audio as speech (helps ignore background noise)

# Calculator-specific grammar for Vosk
# This limits recognition vocabulary to math-related words so the recognizer behaves more predictably
CALC_GRAMMAR = [
    "calculator", "calculate",

    # operators / keywords
    "plus", "add", "minus", "subtract",
    "times", "multiply", "multiplied by",
    "divided by", "divide by", "over",
    "mod", "modulo", "remainder",
    "to the power of", "power of", "power",

    # number words (for word2number)
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen",
    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
    "seventy", "eighty", "ninety", "hundred", "thousand",
]


def resource_path(relative_path: str) -> str:
    """
    Return the absolute path to a resource (e.g. Vosk model folder).

    This helper is compatible with PyInstaller:
    - When bundled, files are extracted to a temporary directory (sys._MEIPASS).
    - When running as a normal script, we use the current working directory.
    """
    # When compiled with PyInstaller, _MEIPASS attribute exists and points to temp folder
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        # When running normally, just use the current directory
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Folder name for the Vosk speech model
VOSK_MODEL_DIR = "india-english"
# Resolve full model path (handles PyInstaller vs normal execution)
VOSK_MODEL_PATH = resource_path(VOSK_MODEL_DIR)

# ============================================================
# ====================== GLOBAL STATE ========================
# ============================================================

# Global reference to the Vosk model (loaded in main())
model = None

# Global text-to-speech engine instance
engine = pyttsx3.init()

# Lock to ensure only one thread uses the TTS engine at a time
engine_lock = threading.Lock()

# Thread-safe queue used to send events from background threads to the GUI thread
gui_queue: "queue.Queue[dict]" = queue.Queue()

# Event to signal whether the assistant is currently running
running_event = threading.Event()

# Global GUI widget references (initialized in main())
root: tk.Tk | None = None
status_label: ttk.Label | None = None
you_said_label: ttk.Label | None = None
live_text_label: ttk.Label | None = None
parsed_expr_label: ttk.Label | None = None
canvas: tk.Canvas | None = None
avatar_circle: int | None = None
start_button: ttk.Button | None = None
history_text: tk.Text | None = None

# ============================================================
# =================== AUDIO FILTERING LAYER ==================
# ============================================================

def apply_audio_filter(audio_data: np.ndarray) -> np.ndarray:
    """
    Apply a simple high-pass-like filter and normalization to microphone audio.

    Steps:
    1. Convert int16 audio samples to float for processing.
    2. Apply a simple recursive operation that behaves like a crude high-pass filter
       (it removes very low frequency components and DC offset).
    3. Normalize the signal so its peak is around 80% of the int16 maximum value.
    4. Convert back to int16 for further processing by Vosk.

    If filtering fails for some reason, the original audio is returned.
    """
    if not NOISE_REDUCTION_ENABLED:
        # Filtering disabled by configuration
        return audio_data

    try:
        # Convert input audio array from int16 to float32
        audio_float = audio_data.astype(np.float32)

        # Simple high-pass-like effect to remove DC / very low frequencies
        alpha_high = 0.95
        filtered = np.zeros_like(audio_float)
        filtered[0] = audio_float[0]

        # Recursive filter: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        for i in range(1, len(audio_float)):
            filtered[i] = alpha_high * (filtered[i - 1] + audio_float[i] - audio_float[i - 1])

        # Normalize to 80% of max int16 range
        max_val = np.max(np.abs(filtered))
        if max_val > 0:
            filtered = filtered / max_val * 32767 * 0.8

        return filtered.astype(np.int16)

    except Exception as e:
        # If anything goes wrong, print error and return unmodified audio
        print(f"[FILTER ERROR]: {e}")
        return audio_data


def is_speech_detected(audio_data: np.ndarray) -> bool:
    """
    Very rough speech detection based on RMS (root mean square) volume.

    - Compute RMS of the current audio block.
    - Compare it to a configured VOLUME_THRESHOLD.
    - If RMS exceeds the threshold, treat it as speech.

    Returns:
        True if speech (loud enough), False for silence/low noise.
    """
    try:
        # Convert to float and compute RMS
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        return rms > VOLUME_THRESHOLD
    except Exception:
        # If detection fails, assume we have speech to avoid missing user input
        return True

# ============================================================
# ====================== CORE FUNCTIONS ======================
# ============================================================

def speak(text: str):
    """
    Speak a text string using the pyttsx3 offline TTS engine.

    Also:
    - Updates GUI state to show "speaking" status and current message.
    - Uses a lock so that only one thread uses the engine at once.
    """
    # Tell GUI we are in "speaking" state (change circle color, etc.)
    gui_queue.put({"type": "state", "state": "speaking"})
    gui_queue.put({"type": "status", "text": f"Speaking: {text}"})

    # Log to console for debugging
    print(f"[ASSISTANT]: {text}")

    # Use lock to avoid concurrent access from different threads
    with engine_lock:
        engine.say(text)
        engine.runAndWait()

    # After finishing speech, set state back to "idle"
    gui_queue.put({"type": "state", "state": "idle"})


def is_number_word(word: str) -> bool:
    """
    Return True if a given word is part of the recognized English number vocabulary.

    This is used to detect sequences like "twenty one" so we can group them
    and convert to numeric values via word2number.
    """
    number_words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
        "seventy", "eighty", "ninety", "hundred", "thousand", "and"
    }
    return word in number_words


def words_to_math(text: str) -> str:
    """
    Convert a spoken English sentence into a mathematical expression string.

    Example:
        "calculate two plus three times five"
    becomes:
        "2 + 3 * 5"

    Steps:
        1. Lowercase and remove filler phrases (e.g. "please", "can you").
        2. Replace multi-word operators with markers (e.g. "divided by" -> §DIV§).
        3. Use word2number to convert runs of number words into digits.
        4. Replace operator words (plus, minus, times, etc.) with symbols (+, -, *, /, %).
        5. Restore markers to corresponding operators (*, /, **).
        6. Strip out any characters that are not digits, operators, parentheses or space.
    """
    # Normalize to lowercase
    text = text.lower()

    # Words/phrases we consider "garbage" / filler for calculator phrases
    fillers = [
        "what is", "calculate", "calculator", "please", "can you",
        "hey", "assistant", "the", "a", "an", "is", "equals",
        "time", "gary", "create", "relate", "late",
        "dollars", "fire",
    ]
    # Remove each filler phrase if present
    for garbage in fillers:
        text = text.replace(garbage, " ")

    # Multi-word operator phrases replaced with special markers so they
    # don't get split incorrectly during processing
    op_markers = {
        "multiplied by": "§MUL§",
        "multiply by": "§MUL§",
        "divided by": "§DIV§",
        "divide by": "§DIV§",
        "to the power of": "§POW§",
        "power of": "§POW§",
    }
    for phrase, marker in op_markers.items():
        text = text.replace(phrase, marker)

    # Split into tokens so we can group consecutive number words
    words = text.split()
    result_words: list[str] = []
    i = 0

    # Process token by token
    while i < len(words):
        # If this word is part of a number phrase (e.g. "twenty", "three")
        if is_number_word(words[i]):
            phrase_words = []

            # Collect as many consecutive number words as possible
            while i < len(words) and is_number_word(words[i]):
                phrase_words.append(words[i])
                i += 1

            # Try converting the collected phrase into a numeric value
            phrase_str = " ".join(phrase_words)
            try:
                num = w2n.word_to_num(phrase_str)
                result_words.append(str(num))  # Add converted number as string
            except ValueError:
                # If conversion fails, drop the phrase but continue
                print(f"[WARNING] Could not convert number phrase: {phrase_str}")
        else:
            # Non-number word: keep as is and move forward
            result_words.append(words[i])
            i += 1

    # Rebuild the text after number-word conversion
    text = " ".join(result_words)

    # Simple 1-word operators mapping to math symbols
    simple_ops = {
        "plus": "+",
        "add": "+",
        "minus": "-",
        "subtract": "-",
        "times": "*",
        "multiply": "*",
        "multiplied": "*",
        "x": "*",
        "into": "*",
        "over": "/",
        "divide": "/",
        "divided": "/",
        "mod": "%",
        "modulo": "%",
        "remainder": "%",
    }
    # Replace each operator keyword with its math symbol using regex word boundaries
    for word, symbol in simple_ops.items():
        text = re.sub(rf"\b{word}\b", symbol, text)

    # Restore previously marked multi-word operators
    text = text.replace("§MUL§", "*")
    text = text.replace("§DIV§", "/")
    text = text.replace("§POW§", "**")

    # Keep only valid math characters: digits, operators, parentheses, dot for decimal, and spaces
    cleaned = ""
    for char in text:
        if char in "0123456789+-*/%.() ":
            cleaned += char

    # Normalize whitespace: collapse multiple spaces into one and strip ends
    text = re.sub(r"\s+", " ", cleaned).strip()
    return text


def is_safe_expression(expr: str) -> bool:
    """
    Ensure that the expression string contains only allowed characters.

    This is a safety measure before using eval().
    """
    allowed_chars = "0123456789+-*/%.() "
    return all(c in allowed_chars for c in expr)


def evaluate_expression(expr: str):
    """
    Safely evaluate a mathematical expression provided as a string.

    Safety checks:
    - Expression must not be empty.
    - Expression cannot be just an operator.
    - Parentheses must be balanced.
    - Only allowed characters (digits, basic operators, dot, parentheses).
    - Execution uses a restricted eval environment with no builtins.

    Raises:
        ValueError on invalid expressions or too large results.
        ZeroDivisionError on division by zero.

    Returns:
        Computed numerical result.
    """
    expr = expr.strip()

    # Basic validations
    if not expr:
        raise ValueError("Empty expression")

    if expr in ["+", "-", "*", "/", "%", "**"]:
        raise ValueError("Expression is only an operator")

    if expr.count("(") != expr.count(")"):
        raise ValueError("Unbalanced parentheses")

    if not is_safe_expression(expr):
        raise ValueError("Expression contains invalid characters")

    # Evaluate expression in a very restricted environment:
    # - No builtins
    # - No variables or functions
    try:
        result = eval(expr, {"__builtins__": None}, {})
    except ZeroDivisionError:
        # Re-raise to handle separately in caller
        raise ZeroDivisionError("Division by zero")
    except Exception as e:
        # Catch any other evaluation error
        raise ValueError(f"Could not evaluate expression: {e}")

    # Safety: check magnitude of result
    try:
        if abs(float(result)) > MAX_RESULT_ABS:
            raise ValueError("Result too large")
    except Exception:
        # If result can't be cast to float, we ignore this check
        pass

    return result


def listen_phrase_with_silence(
    max_seconds: int = MAX_LISTEN_SECONDS,
    silence_threshold: float = SILENCE_THRESHOLD,
) -> str:
    """
    Listen for a spoken phrase from the microphone until either:
    - A period of silence longer than 'silence_threshold' occurs after speech, or
    - 'max_seconds' have elapsed in total.

    Uses:
        - Vosk KaldiRecognizer for streaming speech recognition.
        - A grammar restricted to calculator vocabulary (CALC_GRAMMAR).
        - Audio callbacks via sounddevice.InputStream.

    Returns:
        Final recognized text (lowercase) or an empty string if nothing understood.
    """
    # Create a recognizer that uses our custom grammar to limit vocabulary
    recognizer = KaldiRecognizer(model, SAMPLE_RATE, json.dumps(CALC_GRAMMAR))

    # Queue to collect audio blocks from the callback
    audio_q: "queue.Queue[bytes]" = queue.Queue()

    # Track time of last detected speech and whether speech has ever occurred
    last_speech_time = time.time()
    has_spoken = False

    def audio_callback(indata, frames, t, status):
        """
        Callback function called by sounddevice for each audio block.

        Arguments:
            indata: 2D numpy array of shape (frames, channels) with int16 samples
            frames: Number of frames in this block
            t:      Time information (unused)
            status: Contains warnings or errors from audio backend
        """
        if status:
            print(f"[AUDIO STATUS]: {status}")

        # Take first channel and copy to avoid modifying original buffer
        audio_int16 = indata[:, 0].copy().astype(np.int16)

        # Apply our noise reduction / filtering
        filtered_audio = apply_audio_filter(audio_int16)

        # Decide if this block contains speech or not
        if is_speech_detected(filtered_audio):
            # Speech: push actual audio data
            audio_q.put(filtered_audio.tobytes())
        else:
            # Silence: push zeros (keeps recognizer timing consistent)
            silence = np.zeros_like(filtered_audio, dtype=np.int16)
            audio_q.put(silence.tobytes())

    # Open microphone input stream and keep it open until we stop listening
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        start_time = time.time()
        print(f"[LISTENING] Max {max_seconds}s, silence stop after {silence_threshold}s")

        # Main listening loop
        while running_event.is_set():
            elapsed = time.time() - start_time

            # Hard timeout: stop if we exceed max_seconds
            if elapsed > max_seconds:
                print(f"[TIMEOUT] Max recording time ({max_seconds}s) reached")
                break

            # If we've already detected speech and then silence for a while, stop
            if has_spoken and (time.time() - last_speech_time) > silence_threshold:
                print(f"[SILENCE] {silence_threshold}s of silence, finalizing")
                break

            try:
                # Try to fetch the next audio chunk from the queue
                data = audio_q.get(timeout=0.2)
            except queue.Empty:
                # No audio within timeout, loop again
                continue

            # Feed data into recognizer; AcceptWaveform returns True
            # when a final (complete) utterance is recognized
            if recognizer.AcceptWaveform(data):
                result_json = recognizer.Result()
                result = json.loads(result_json)

                # Clear any partial text in the GUI
                gui_queue.put({"type": "partial_text", "text": ""})

                final_text = result.get("text", "")
                print(f"[FINAL STT]: {final_text}")
                return final_text
            else:
                # Partial result: update GUI for live feedback
                partial_json = recognizer.PartialResult()
                partial = json.loads(partial_json).get("partial", "")
                if partial:
                    print(f"[PARTIAL STT]: {partial}")
                    gui_queue.put({"type": "partial_text", "text": partial})
                    last_speech_time = time.time()
                    has_spoken = True

        # If we exit loop due to timeout or stop, ask recognizer for final result
        final_json = recognizer.FinalResult()
        final = json.loads(final_json)

        # Clear partial text from GUI
        gui_queue.put({"type": "partial_text", "text": ""})

        final_text = final.get("text", "")
        print(f"[FINAL STT timeout]: {final_text}")
        return final_text

# ============================================================
# ================= ASSISTANT MAIN LOOP (THREAD) =============
# ============================================================

def assistant_loop():
    """
    Background thread function that runs the main assistant logic:

    1. Greets the user ("Voice calculator ready...").
    2. Repeatedly:
        - Listens for spoken input.
        - Checks for wake word ("calculator" / "calculate").
        - Converts recognized speech to a math expression.
        - Evaluates the expression safely.
        - Speaks back the result or an error message.
    3. Updates the GUI via gui_queue for status, history, and visual feedback.
    """
    # Initial spoken greeting
    speak("Voice calculator ready. Say your calculation starting with calculator.")

    # Main loop: continue while the global running_event is set
    while running_event.is_set():
        # Update GUI: show listening state and instructions
        gui_queue.put({"type": "state", "state": "listening"})
        gui_queue.put({"type": "status", "text": "Listening... Speak your calculation."})

        # Blocking call: waits for user speech with silence/time-based stop
        text = listen_phrase_with_silence()

        # If assistant was stopped while listening, exit the loop
        if not running_event.is_set():
            break

        # If nothing was recognized, ask user to try again
        if not text:
            gui_queue.put({"type": "status", "text": "I didn't catch that. Try again."})
            continue

        # Update GUI "You said: ..."
        gui_queue.put({"type": "user_text", "text": text})
        print(f"[YOU]: {text}")

        lower = text.lower()

        # Wake-word check: if required and not present, ignore this phrase
        if WAKE_WORDS and not any(w in lower for w in WAKE_WORDS):
            gui_queue.put(
                {"type": "status", "text": "No wake word (calculator) detected, ignoring."}
            )
            continue

        # Remove wake words from the recognized text to simplify math parsing
        for w in WAKE_WORDS:
            lower = lower.replace(w, " ")

        # Convert the spoken phrase to a math expression string
        expr = words_to_math(lower)

        # Show parsed expression in GUI
        gui_queue.put({"type": "parsed_expr", "text": f"Expression: {expr or '(empty)'}"})
        gui_queue.put({"type": "status", "text": f"Evaluating: {expr}"})
        print(f"[PARSED]: {expr}")

        # Inform GUI that assistant is "thinking"
        gui_queue.put({"type": "state", "state": "thinking"})

        try:
            # Try to compute the result of the expression
            result = evaluate_expression(expr)
            answer = f"The result is {result}"

            # Update history and status in GUI
            gui_queue.put({"type": "assistant_text", "text": answer})
            gui_queue.put({"type": "history", "expr": expr, "result": result})

            # Speak the result out loud
            speak(answer)

        except ZeroDivisionError:
            # Handle division by zero separately for a clearer user message
            error_msg = "Error: division by zero"
            gui_queue.put({"type": "assistant_text", "text": error_msg})
            speak(error_msg)

        except ValueError as e:
            # Any other evaluation / parsing issue
            print(f"[EVAL ERROR]: {e}")
            error_msg = "Sorry, I could not understand that calculation."
            gui_queue.put({"type": "assistant_text", "text": error_msg})
            speak(error_msg)

    # When loop ends, set GUI state back to idle
    gui_queue.put({"type": "state", "state": "idle"})
    gui_queue.put({"type": "status", "text": "Assistant stopped."})

# ============================================================
# ======================= GUI HELPERS ========================
# ============================================================

def set_avatar_state(state: str):
    """
    Update the color of the avatar circle depending on assistant state.

    States:
        - "listening" : Blue
        - "thinking"  : Yellow
        - "speaking"  : Green
        - anything else: Gray (idle)
    """
    if canvas is None or avatar_circle is None:
        # GUI not initialized yet
        return

    if state == "listening":
        color = "#3b82f6"   # Blue
    elif state == "thinking":
        color = "#facc15"   # Yellow
    elif state == "speaking":
        color = "#22c55e"   # Green
    else:
        color = "#9ca3af"   # Gray (idle)

    # Change the fill color of the circle
    canvas.itemconfig(avatar_circle, fill=color)


def clear_history():
    """
    Clear the history of previous calculations from the text widget.
    """
    global history_text
    if history_text is None:
        return

    history_text.config(state="normal")
    history_text.delete("1.0", "end")
    history_text.config(state="disabled")


def process_gui_queue():
    """
    Process all pending events from gui_queue and update the GUI.

    This function is called periodically using root.after(), so all
    GUI updates happen in the main thread (Tkinter is not thread-safe).
    """
    global status_label, you_said_label, live_text_label, parsed_expr_label, history_text

    try:
        # Handle all events currently in the queue
        while True:
            event = gui_queue.get_nowait()
            etype = event.get("type")

            if etype == "state":
                # Assistant state change: update avatar color
                state = event.get("state", "idle")
                set_avatar_state(state)

            elif etype == "status":
                # Status line at the center of the screen
                text = event.get("text", "")
                if status_label is not None:
                    status_label.config(text=text)

            elif etype == "user_text":
                # "You said: ..." label gets updated
                text = event.get("text", "")
                if you_said_label is not None:
                    you_said_label.config(text=f"You said: {text}")

            elif etype == "assistant_text":
                # Assistant's final message replaces status text
                text = event.get("text", "")
                if status_label is not None:
                    status_label.config(text=text)

            elif etype == "partial_text":
                # Live partial transcription while user is speaking
                text = event.get("text", "")
                if live_text_label is not None:
                    if text:
                        live_text_label.config(text=f"Listening: {text}")
                    else:
                        live_text_label.config(text="Waiting for speech...")

            elif etype == "parsed_expr":
                # Display parsed math expression
                text = event.get("text", "")
                if parsed_expr_label is not None:
                    parsed_expr_label.config(text=text)

            elif etype == "history":
                # Append new line to history text box
                expr = event.get("expr", "")
                result = event.get("result", "")
                if history_text is not None:
                    history_text.config(state="normal")
                    history_text.insert("end", f"{expr} = {result}\n")
                    history_text.see("end")  # Scroll to bottom
                    history_text.config(state="disabled")

    except queue.Empty:
        # No more events at the moment
        pass

    # Schedule the next check
    if root is not None:
        root.after(150, process_gui_queue)


def toggle_assistant():
    """
    Start or stop the assistant depending on current running state.

    - If not running: set event, start background thread, update button text.
    - If running: clear event to ask thread to stop, update button text.
    """
    global start_button

    # If assistant is currently not running, start it
    if not running_event.is_set():
        running_event.set()

        if start_button is not None:
            start_button.config(text="Stop AI")

        if status_label is not None:
            status_label.config(text="Starting assistant...")

        # Launch assistant_loop in a separate daemon thread
        thread = threading.Thread(target=assistant_loop, daemon=True)
        thread.start()

    else:
        # Assistant is running: send stop signal
        running_event.clear()

        if start_button is not None:
            start_button.config(text="Start AI")

        if status_label is not None:
            status_label.config(text="Stopping assistant...")


def cleanup():
    """
    Clean shutdown:
    - Stop assistant loop.
    - Stop TTS engine.
    - Destroy the Tk root window.
    """
    print("[CLEANUP] Stopping assistant and closing.")
    running_event.clear()

    # Stop TTS engine if possible
    with engine_lock:
        try:
            engine.stop()
        except Exception:
            pass

    # Close GUI window
    if root is not None:
        root.destroy()

# ============================================================
# ================== MAIN ENTRY / GUI SETUP ==================
# ============================================================

def main():
    """
    Application entry point.

    1. Load Vosk speech model.
    2. Create main Tkinter window.
    3. Build all GUI widgets (avatar, labels, buttons, history box).
    4. Start periodic processing of GUI event queue.
    5. Enter Tkinter mainloop.
    """
    global model, root, status_label, you_said_label, live_text_label
    global parsed_expr_label, canvas, avatar_circle, start_button, history_text

    # Load the Vosk model (may take a few seconds depending on model size)
    print("Loading Vosk model from:", VOSK_MODEL_PATH)
    try:
        model = Model(VOSK_MODEL_PATH)
    except Exception as e:
        # If loading fails, show error dialog and exit
        messagebox.showerror("Error", f"Failed to load Vosk model:\n{e}")
        print("Failed to load Vosk model:", e)
        sys.exit(1)

    print("Model loaded.")
    print(f"Audio filtering: {'ENABLED' if NOISE_REDUCTION_ENABLED else 'DISABLED'}")
    print(f"Max recording time: {MAX_LISTEN_SECONDS} seconds")
    print(f"Silence threshold: {SILENCE_THRESHOLD} seconds")

    # Create main application window
    root = tk.Tk()
    root.title("My First AI Voice Assistant")
    root.geometry("460x700")          # Fixed window size
    root.resizable(False, False)      # Disable manual resizing

    # Main content frame with padding
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill="both", expand=True)

    # Title label at top
    title_label = ttk.Label(
        main_frame,
        text="My AI Voice Calculator",
        font=("Segoe UI", 16, "bold"),
    )
    title_label.pack(pady=(0, 20))

    # Avatar canvas with circle representing assistant state
    canvas = tk.Canvas(main_frame, width=200, height=200, highlightthickness=0)
    canvas.pack()
    avatar_circle = canvas.create_oval(20, 20, 180, 180, fill="#9ca3af", outline="")

    # Live partial transcription label
    live_text_label = ttk.Label(
        main_frame,
        text="Waiting for speech...",
        font=("Segoe UI", 10, "bold"),
        wraplength=420,
        justify="center",
        foreground="#111827",
    )
    live_text_label.pack(pady=(10, 5))

    # Parsed expression label (shows final math expression derived from voice)
    parsed_expr_label = ttk.Label(
        main_frame,
        text="Expression: (none yet)",
        font=("Courier New", 10),
        wraplength=420,
        justify="center",
        foreground="#6b7280",
    )
    parsed_expr_label.pack(pady=(0, 10))

    # Status label (instructions / errors / final result message)
    status_label = ttk.Label(
        main_frame,
        text="Press 'Start AI' and speak your calculation.",
        font=("Segoe UI", 10),
        wraplength=420,
        justify="center",
    )
    status_label.pack(pady=(10, 10))

    # "You said" label to display final recognized phrase
    you_said_label = ttk.Label(
        main_frame,
        text="You said: ",
        font=("Segoe UI", 10, "italic"),
        wraplength=420,
        justify="center",
    )
    you_said_label.pack(pady=(0, 15))

    # Buttons row for "Start AI" and "Clear History"
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=(5, 10))

    # Start/Stop assistant button
    start_button = ttk.Button(button_frame, text="Start AI", command=toggle_assistant)
    start_button.pack(side="left", padx=5)

    # Clear calculation history button
    clear_button = ttk.Button(button_frame, text="Clear History", command=clear_history)
    clear_button.pack(side="left", padx=5)

    # Horizontal separator line
    ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=(5, 5))

    # History section label
    history_label = ttk.Label(
        main_frame,
        text="History",
        font=("Segoe UI", 11, "bold"),
    )
    history_label.pack()

    # History frame holds text widget and scrollbar
    history_frame = ttk.Frame(main_frame)
    history_frame.pack(fill="both", expand=True, pady=(5, 0))

    # Text widget for listing past calculations
    history_text = tk.Text(
        history_frame,
        height=6,
        width=50,
        state="disabled",
        font=("Courier New", 9),
        bg="#f9fafb",
        relief="flat",
        borderwidth=1,
    )
    history_text.pack(side="left", fill="both", expand=True)

    # Vertical scrollbar for the history text widget
    scrollbar = ttk.Scrollbar(history_frame, command=history_text.yview)
    scrollbar.pack(side="right", fill="y")
    history_text.config(yscrollcommand=scrollbar.set)

    # Start periodic GUI queue processing
    root.after(150, process_gui_queue)

    # When window close button is pressed, run cleanup() first
    root.protocol("WM_DELETE_WINDOW", cleanup)

    # Enter Tkinter main event loop
    root.mainloop()


# Standard Python entry point guard
if __name__ == "__main__":
    main()
