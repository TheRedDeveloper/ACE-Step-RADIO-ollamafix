#!/usr/bin/env python3
"""
AI Radio Station - Command Line Version
Continuous music generation with keyboard controls
"""

import os
import sys
import time
import queue
import random
import argparse
import threading
import gc
import re
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import json
import readline
import torch
import pygame
import select
import sys
import tty
import termios

from acestep.pipeline_ace_step import ACEStepPipeline

# Try to import ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama package not available. Please install it: pip install ollama")


# ============================================================================
# Constants and Configuration
# ============================================================================

DEFAULT_DURATION = 130  # seconds

SUPPORTED_LANGUAGES = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese",
    "Japanese", "Chinese", "Russian", "Arabic", "Hindi", "Korean", "Finnish"
]

ALL_GENRES = [
    "pop", "rock", "electronic", "lofi", "jazz", "classical",
    "ambient", "country", "metal", "death metal", "doom metal", "reggae",
    "blues", "delta blues", "funk", "disco", "punk",
    "ballad", "retro", "folk", "chiptune"
]

ALL_MOODS = [
    "upbeat", "melancholic", "energetic", "calm", "reflective"
]

GENRE_TEMPOS = {
    "pop": 120, "rock": 140, "electronic": 128,
    "lofi": 85, "jazz": 110, "classical": 80, "ambient": 70,
    "country": 100, "metal": 160, "death metal": 180, "doom metal": 60,
    "reggae": 90, "blues": 100, "delta blues": 80,
    "funk": 115, "disco": 120, "punk": 180,
    "ballad": 75, "retro": 110, "folk": 95, "chiptune": 140,
    "default": 120
}

THEME_SUGGESTIONS = {
    "pop": [
        "summer love", "heartbreak", "dancing", "celebrity crush", "night out",
        "young and free", "last goodbye", "party all night", "secret admirer",
        "teenage dreams", "radio hit", "glitter and gold", "midnight kiss"
    ],
    "rock": [
        "rebellion", "road trip", "broken dreams", "rock and roll lifestyle",
        "guitar hero", "against the machine", "burning bridges", "small town blues",
        "arena anthem", "leather and spikes", "last train home", "bar fight"
    ],
    "electronic": [
        "neon lights", "cosmic journey", "digital dreams", "midnight drive",
        "rave till dawn", "synthetic love", "cyberpunk city", "bass drop",
        "laser fantasy", "underground warehouse", "alien transmission"
    ],
    "lofi": [
        "rainy day", "coffee shop", "study session", "chill vibes",
        "late night thoughts", "window seat", "old cassette tapes",
        "sunday morning", "vinyl crackle", "city skyline at dusk"
    ],
    "jazz": [
        "smooth nights", "blue mood", "speakeasy", "saxophone dreams",
        "moonlight serenade", "smoky bar", "improvisation", "uptown swing",
        "jazz age", "walking bassline", "after hours jam", "velvet voice"
    ],
    "classical": [
        "moonlight", "spring morning", "winter tale", "royal ball",
        "symphony no. 9", "opera drama", "baroque garden", "piano concerto",
        "sunrise sonata", "gothic cathedral", "waltz of the flowers"
    ],
    "ambient": [
        "ocean waves", "forest walk", "mountain sunrise", "deep space",
        "arctic winds", "desert mirage", "underwater caves", "celestial bodies",
        "morning mist", "ancient ruins", "silent meditation"
    ],
    "country": [
        "small town", "truck driving", "lost love", "whiskey nights", "backroads",
        "dusty boots", "honky tonk angel", "rodeo clown", "front porch swing",
        "blue jeans", "pickup truck", "county fair", "dirt road diary"
    ],
    "metal": [
        "battle cries", "apocalypse now", "dragon slayer", "demonic possession",
        "forge of souls", "viking conquest", "doomsday prophecy", "blackened sky",
        "mosh pit madness", "necrotic ritual", "shred till death"
    ],
    "reggae": [
        "island breeze", "one love", "ganja farmer", "sunshine state of mind",
        "beach bonfire", "rasta revolution", "dub plate special",
        "tropical storm", "kingston nights", "irie vibrations"
    ],
    "blues": [
        "crossroads deal", "mississippi delta", "empty whiskey bottle",
        "train whistle blues", "juke joint", "broken guitar string",
        "stormy monday", "dust my broom", "hard luck woman"
    ],
    "ballad": [
        "lost in time", "unspoken words", "farewell letter", "moonlit promise", "eternal embrace",
        "broken wings", "silent tears", "last dance", "fading memories", "star-crossed lovers"
    ],
    "retro": [
        "disco fever", "neon nights", "roller rink", "vintage dreams", "old school groove",
        "arcade adventure", "boombox beats", "retro romance", "cassette rewind", "synthwave sunset"
    ],
    "folk": [
        "mountain trail", "campfire stories", "wandering soul", "river crossing", "old country road",
        "harvest moon", "family roots", "traveler's tale", "wildflower fields", "timberland song"
    ],
    "chiptune": [
        "pixel quest", "level up", "game over", "bitwise love", "arcade hero",
        "boss battle", "high score", "retro runner", "glitch in time", "power-up anthem"
    ],
    "default": [
        "love", "dreams", "adventure", "nostalgia", "second chances",
        "hidden truth", "forbidden fruit", "eternal youth", "fading memories"
    ]
}


# ============================================================================
# Data Classes
# ============================================================================

class RadioState(Enum):
    STOPPED = auto()
    BUFFERING = auto()
    PLAYING = auto()
    PAUSED = auto()


@dataclass
class Song:
    title: str
    artist: str
    genre: str
    theme: str
    duration: float
    lyrics: str
    language: str
    prompt: str
    audio_path: str
    generation_time: float
    timestamp: float
    tempo: int
    intensity: str
    mood: str
    metadata: dict


# ============================================================================
# AI Radio Station
# ============================================================================

class AIRadioStation:
    def __init__(self, checkpoint_dir: str, ollama_lyrics_model: str = "gemma3:12b-it-q4_K_M", ollama_vision_model: str = "minicpm-v"):
        """Initialize the AI Radio Station"""
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama is required but not available. Install with: pip install ollama")

        self.checkpoint_dir = checkpoint_dir
        self.ollama_lyrics_model = ollama_lyrics_model
        self.ollama_vision_model = ollama_vision_model
        self.pipeline_args = {
            'checkpoint_dir': checkpoint_dir,
            'dtype': "bfloat16",
            'torch_compile': False,
            'silent': True
        }
        self.current_pipeline = None
        self.current_song: Optional[Song] = None
        self.song_queue = queue.Queue(maxsize=1)
        self.state = RadioState.STOPPED
        self.stop_event = threading.Event()
        self.skip_event = threading.Event()
        self.restart_event = threading.Event()
        self.pause_event = threading.Event()
        self.generation_thread = None
        self.playback_thread = None
        self.playback_history = []
        self.last_genres = []
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.user_message = ""
        self.user_message_lock = threading.Lock()

        # Clean up output folder on startup
        self._cleanup_output_folder()

        # Initialize pygame mixer for audio playback
        pygame.mixer.init(frequency=48_000, size=-16, channels=2, buffer=512)

        # Playback state
        self.playback_lock = threading.Lock()
        self.is_paused = False

    def _cleanup_output_folder(self):
        """Clean up all files in the output folder on startup"""
        try:
            if self.output_dir.exists():
                for file in self.output_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                print(f"🧹 Cleaned up output folder")
        except Exception as e:
            print(f"⚠️  Error cleaning up output folder: {e}")

    def get_pipeline(self):
        """Get or create the pipeline"""
        if self.current_pipeline is None:
            self.current_pipeline = ACEStepPipeline(**self.pipeline_args)
        return self.current_pipeline

    def release_pipeline(self):
        """Clean up pipeline resources thoroughly"""
        if self.current_pipeline is not None:
            # First delete ACE-specific resources
            if hasattr(self.current_pipeline, 'ace_model'):
                if hasattr(self.current_pipeline.ace_model, 'cpu'):
                    try:
                        self.current_pipeline.ace_model.cpu()
                    except:
                        pass
                del self.current_pipeline.ace_model

            # Now delete the full pipeline
            del self.current_pipeline
            self.current_pipeline = None

            # Force cleanup
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

    def clean_all_memory(self):
        """More aggressive memory cleanup"""
        # Then release pipeline resources
        self.release_pipeline()

        # Force CUDA cleanup with synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all kernels to finish
            torch.cuda.empty_cache()  # Clear cache
            torch.cuda.reset_peak_memory_stats()  # Reset tracking

        gc.collect()

    def generate_random_parameters(self) -> dict:
        """Generate random song parameters"""
        genre = random.choice(ALL_GENRES)
        themes = THEME_SUGGESTIONS.get(genre, THEME_SUGGESTIONS["default"])
        theme = random.choice(themes)
        duration = DEFAULT_DURATION
        tempo = GENRE_TEMPOS.get(genre, GENRE_TEMPOS["default"])
        language = "English"  # random.choice(SUPPORTED_LANGUAGES)

        # Add some variation to tempo
        tempo = tempo + random.randint(-10, 10)
        tempo = max(40, min(200, tempo))  # Clamp between 40-200 BPM

        intensity = random.choice(["low", "medium", "high"])
        mood = random.choice(["upbeat", "melancholic", "energetic", "calm", "reflective"])

        return {
            "genre": genre,
            "theme": theme,
            "duration": duration,
            "tempo": tempo,
            "language": language,
            "intensity": intensity,
            "mood": mood
        }

    def generate_parameters_from_screen_with_ollama(self) -> dict:
        """Generate song parameters from what's on the screen using Ollama, supporting Wayland."""
        import shutil
        import subprocess

        if self.state == RadioState.BUFFERING:
            print("🖼️ Looking at screen...")

        screenshot_path = os.path.join(self.output_dir, "screenshot.jpg")
        screenshot_taken = False

        # Try grim (Wayland, Sway/wlroots)
        if shutil.which("grim"):
            try:
                subprocess.run(["grim", "-s", "0.5", screenshot_path], check=True)
                screenshot_taken = True
            except Exception as e:
                print(f"⚠️  grim screenshot failed: {e}")
        # Try gnome-screenshot (GNOME Wayland)
        elif shutil.which("gnome-screenshot"):
            try:
                subprocess.run(["gnome-screenshot", f"--file={screenshot_path}"], check=True)
                screenshot_taken = True
            except Exception as e:
                print(f"⚠️  gnome-screenshot failed: {e}")
        # Fallback to mss (X11)
        else:
            try:
                import mss
                with mss.mss() as sct:
                    sct.shot(output=screenshot_path)
                screenshot_taken = True
            except Exception as e:
                print(f"⚠️  mss screenshot failed: {e}")

        if not screenshot_taken or not os.path.exists(screenshot_path):
            print(f"⚠️  Screenshot not found at {screenshot_path}")
            return self.generate_random_parameters()

        playback_history_text = (
            f"This is the playback history: ... {', '.join(self.playback_history)}, [ YOUR SONG ]\n"
            "Do not base yourself on the playback history. Come up with something different, based on the screenshot.\n"
            "Too much of one genre becomes straining. Do not repeat a genre 3 times, there are so many creative possibilities.\n"
            "This does not hint at the music taste, the songs were not chosen by the user.\n"
        ) if self.playback_history else ""

        playback_genre_warning = f"The genre {self.last_genres[-1]} IS NOT ALLOWED, it would repeat 3 times in a row. Choose a different genre.\n" if len(self.last_genres) >= 2 and self.last_genres[-1] == self.last_genres[-2] else ""

        with self.user_message_lock:
            user_msg = self.user_message

        user_message_text = f"\nUser Message: \"{user_msg}\"\nConsider this message by the user when choosing parameters.\n" if user_msg else ""

        prompt = (
            "Generate a set of song json song parameters, fitting the provided screenshot.\n"
            "Think of what the user might want to listen to right now, based on the screenshot.\n"
            "Return only the response json, nothing else.\n"
            f"{playback_history_text}"
            f"{playback_genre_warning}"
            f"{user_message_text}"
            "\n"
            "You are limited to the exact genres provided, do not deviate.\n"
            f"This is the array of genres, DO NOT deviate: {', '.join(ALL_GENRES)}.\n"
            "Choose the genre that the user would most likely enjoy, based on the screenshot.\n"
            "You may freely choose a theme as a topic of the music and lyrics based on the screenshot.\n"
            "Your theme must be short and concise of 1 to 5 words, like a title for the song.\n"
            "Your theme may latch onto objects, colors, emotions, places, people, actions, or anything else you see.\n"
            "Do not always choose the most obvious theme, be creative.\n"
            "You should think if the user would like something different.\n"
            "You are limited to these intensities: low, medium, high - DO NOT DEVIATE.\n"
            "This will be the intensity of the song, think about what fits best.\n"
            f"You are limited to these moods, DO NOT DEVIATE: {', '.join(ALL_MOODS)}\n"
            "Choose the mood that best fits what the user might want to hear.\n"
            "You need to choose a tempo in BPM, unlike the other parameters, this needs to be an integer and it should be reasonable.\n"
            "\n"
            "The output must be in this exact format:\n"
            "```json\n"
            f"{{\"action\": \"<description of the users screen and what the user is doing in the screenshot>\", \"genre\": \"<one of these genres: {', '.join(ALL_GENRES)} - DO NOT DEVIATE>\", \"theme\": \"<a theme of the song used for lyric and music generation - 1 to 5 words>\", \"tempo\": <tempo in BPM, integer>, \"intensity\": \"<one of these: low, medium, high - DO NOT DEVIATE>\", \"mood\": \"<one of these: {', '.join(ALL_MOODS)} - DO NOT DEVIATE>\"}}\n"
            "```\n"
            "\n"
            "DO NOT DEVIATE FROM THE FORMAT\n"
            "DO NOT ADD ANYTHING ELSE\n"
            "DO NOT EXPLAIN ANYTHING\n"
            "SERVE JSON ONLY\n"
            "START LIKE THIS: ```json\n"
        )

        if self.state == RadioState.BUFFERING:
            print("💭 Inventing song...")
        for _ in range(10):
            try:
                output = ollama.generate(
                    self.ollama_vision_model,
                    prompt,
                    images=[screenshot_path],
                    keep_alive=0.5
                )
                # Find the JSON block in Ollama's output
                match = re.search(r'```json\s*(\{[^\}]*\})\s*```', output['response'], re.DOTALL)
                if not match:
                    continue

                json_str = match.group(1)
                try:
                    params = json.loads(json_str)
                    # Validate required keys
                    required_keys = {"action", "genre", "theme", "tempo", "intensity", "mood"}
                    # Check for missing keys
                    if not required_keys.issubset(params.keys()):
                        continue
                    # Check for extra keys
                    if set(params.keys()) != required_keys:
                        continue
                    # Check types
                    if not (
                        isinstance(params["action"], str) and
                        isinstance(params["genre"], str) and
                        isinstance(params["theme"], str) and
                        isinstance(params["intensity"], str) and
                        isinstance(params["mood"], str) and
                        isinstance(params["tempo"], int)
                    ):
                        continue
                    # Check the values are of the correct form
                    if params["genre"] not in ALL_GENRES:
                        continue
                    if params["intensity"] not in ["low", "medium", "high"]:
                        continue
                    if params["mood"] not in ALL_MOODS:
                        continue
                    if len(params["theme"].strip().split()) > 5 or len(params["theme"].strip()) == 0:
                        continue
                    if not (40 <= params["tempo"] <= 200):
                        continue

                    # Check for genre repetition (no 3 in a row)
                    last_genres = self.last_genres[-2:] if hasattr(self, 'last_genres') else []
                    if len(last_genres) == 2 and all(g == params["genre"] for g in last_genres):
                        continue

                    # Add language and duration, remove action
                    del params['action']
                    params["language"] = "English"
                    params["duration"] = DEFAULT_DURATION
                    # Delete the screenshot after parsing
                    try:
                        if os.path.exists(screenshot_path):
                            os.remove(screenshot_path)
                    except Exception as e:
                        print(f"⚠️  Error deleting screenshot: {e}")
                    return params
                except Exception as e:
                    print(f"⚠️  Error parsing Ollama JSON: {e}")
                    continue
            except Exception as e:
                print(f"⚠️  Ollama generation error: {e}")
        # Fallback to random parameters if all attempts fail
        try:
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)
        except Exception as e:
            print(f"⚠️  Error deleting screenshot: {e}")
        return self.generate_random_parameters()

    def generate_lyrics_with_ollama(self, genre: str, theme: str, duration: float,
                                    language: str, tempo: int, intensity: str,
                                    mood: str) -> Tuple[str, str]:
        """Generate lyrics using Ollama"""

        long_structures = {
            "country": (
                "[Steel Guitar Intro]\n\n"
                "[Verse 1] (storytelling)\n{lyrics}\n\n"
                "[Chorus] (big melody)\n{lyrics}\n\n"
                "[Verse 2] (develop story)\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Fiddle Solo] (8 bars)\n\n"
                "[Bridge] (emotional peak)\n{lyrics}\n\n"
                "[Double Chorus] (with harmonies)"
            ),
            "pop": (
                "[Verse 1]\n{lyrics}\n\n"
                "[Pre-Chorus]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Pre-Chorus]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge]\n{lyrics}\n\n"
                "[Final Chorus] (with ad-libs)"
            ),
            "rock": (
                "[Guitar Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Guitar Solo] (8-16 bars)\n\n"
                "[Bridge]\n{lyrics}\n\n"
                "[Double Chorus] (big finish)"
            ),
            "electronic": (
                "[Atmospheric Intro] (16 bars)\n\n"
                "[Build-Up]\n{lyrics}\n\n"
                "[Drop]\n{lyrics}\n\n"
                "[Breakdown Verse]\n{lyrics}\n\n"
                "[Build-Up]\n{lyrics}\n\n"
                "[Drop]\n{lyrics}\n\n"
                "[Outro] (beat fade)"
            ),
            "lofi": (
                "[Ambient Intro] (with vinyl noise)\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chill Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chill Chorus]\n{lyrics}\n\n"
                "[Instrumental Break] (8 bars)\n\n"
                "[Outro] (fade with rain sounds)"
            ),
            "jazz": (
                "[Piano Intro] (improvised)\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Swing Chorus]\n{lyrics}\n\n"
                "[Instrumental Break] (sax solo)\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Swing Chorus]\n{lyrics}\n\n"
                "[Outro] (group improv)"
            ),
            "classical": (
                "[Orchestral Introduction]\n\n"
                "[Theme A]\n{lyrics}\n\n"
                "[Theme B] (variation)\n\n"
                "[Development Section]\n\n"
                "[Recapitulation]\n{lyrics}\n\n"
                "[Coda] (grand finale)"
            ),
            "ambient": (
                "[Textural Intro] (2-4 minutes)\n\n"
                "[Drone Section]\n{lyrics}\n\n"
                "[Modulation]\n\n"
                "[Resolution Section]\n{lyrics}\n\n"
                "[Fade Out] (gradual)"
            ),
            "metal": (
                "[Shredding Intro] (fast picking)\n\n"
                "[Verse 1] (growled vocals)\n{lyrics}\n\n"
                "[Chorus] (clean vocals)\n{lyrics}\n\n"
                "[Guitar Solo] (tapping)\n\n"
                "[Breakdown] (chugging riffs)\n\n"
                "[Final Blast] (double bass)"
            ),
            "reggae": (
                "[Skank Guitar Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus] (call-and-response)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Dub Section] (instrumental)\n\n"
                "[Final Chorus] (with harmonies)"
            ),
            "blues": (
                "[Guitar Lick Intro] (12-bar)\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Response] (guitar answers vocal)\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Harmonica Solo] (12-bar)\n\n"
                "[Outro] (repeat and fade)"
            ),
            "ballad": (
                "[Piano Intro] (soft, emotional)\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus] (heartfelt)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge] (emotional peak)\n{lyrics}\n\n"
                "[Final Chorus] (with harmonies)"
            ),
            "retro": (
                "[Synthwave Intro] (vintage sounds)\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus] (catchy)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge] (nostalgic)\n{lyrics}\n\n"
                "[Final Chorus] (big finish)"
            ),
            "folk": (
                "[Acoustic Guitar Intro]\n\n"
                "[Verse 1] (storytelling)\n{lyrics}\n\n"
                "[Chorus] (singalong)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge] (reflective)\n{lyrics}\n\n"
                "[Final Chorus] (group vocals)"
            ),
            "chiptune": (
                "[8-bit Intro] (game sounds)\n\n"
                "[Level 1]\n{lyrics}\n\n"
                "[Boss Battle]\n{lyrics}\n\n"
                "[Level 2]\n{lyrics}\n\n"
                "[Power-Up]\n{lyrics}\n\n"
                "[Final Level]\n{lyrics}\n\n"
                "[Victory Theme] (high score)"
            ),
            "default": (
                "[Intro]\n{lyrics}\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge/Middle 8]\n{lyrics}\n\n"
                "[Chorus] (variation)\n\n"
                "[Outro] (optional vamp)"
            )
        }

        medium_structures = {
            "country": (
                "[Steel Guitar Intro]\n\n"
                "[Verse 1] (storytelling)\n{lyrics}\n\n"
                "[Chorus] (big melody)\n{lyrics}\n\n"
                "[Verse 2] (develop story)\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge] (emotional peak)\n{lyrics}\n\n"
                "[Final Chorus]"
            ),
            "pop": (
                "[Verse 1]\n{lyrics}\n\n"
                "[Pre-Chorus]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge]\n{lyrics}\n\n"
                "[Chorus] (final)"
            ),
            "rock": (
                "[Guitar Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Guitar Solo] (4-8 bars)\n\n"
                "[Final Chorus]"
            ),
            "hip hop": (
                "[Intro Hook]\n{lyrics}\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Outro]"
            ),
            "electronic": (
                "[Build-Up] (8 bars)\n{lyrics}\n\n"
                "[Drop]\n{lyrics}\n\n"
                "[Breakdown]\n{lyrics}\n\n"
                "[Build-Up]\n{lyrics}\n\n"
                "[Drop] (final)"
            ),
            "lofi": (
                "[Ambient Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chill Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chill Chorus]\n{lyrics}\n\n"
                "[Outro] (fade)"
            ),
            "jazz": (
                "[Piano Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Swing Chorus]\n{lyrics}\n\n"
                "[Sax Solo] (8 bars)\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Swing Chorus]"
            ),
            "classical": (
                "[Introduction]\n\n"
                "[Theme A]\n{lyrics}\n\n"
                "[Theme B]\n{lyrics}\n\n"
                "[Development]\n{lyrics}\n\n"
                "[Recapitulation]\n{lyrics}"
            ),
            "ambient": (
                "[Textural Intro]\n\n"
                "[Drone Section]\n{lyrics}\n\n"
                "[Modulation]\n{lyrics}\n\n"
                "[Resolution]\n{lyrics}\n\n"
                "[Fade Out]"
            ),
            "metal": (
                "[Heavy Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Guitar Solo] (8 bars)\n\n"
                "[Final Chorus]"
            ),
            "death metal": (
                "[Blast Intro]\n\n"
                "[Verse 1] (growled)\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Breakdown]\n\n"
                "[Final Blast]"
            ),
            "doom metal": (
                "[Slow Heavy Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus] (crushing)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Outro] (fade)"
            ),
            "reggae": (
                "[Skank Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus] (call-response)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Dub Break] (8 bars)\n\n"
                "[Final Chorus]"
            ),
            "blues": (
                "[Guitar Lick Intro]\n\n"
                "[Verse 1] (12-bar)\n{lyrics}\n\n"
                "[Response] (guitar)\n\n"
                "[Verse 2] (12-bar)\n{lyrics}\n\n"
                "[Harmonica Solo] (8 bars)\n\n"
                "[Verse 3]\n{lyrics}\n\n"
                "[Outro]"
            ),
            "delta blues": (
                "[Acoustic Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Slide Guitar Response]\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Instrumental Break]\n\n"
                "[Verse 3]\n{lyrics}\n\n"
                "[Outro]"
            ),
            "funk": (
                "[Groove Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bass Break] (4 bars)\n\n"
                "[Chorus] (vamp out)"
            ),
            "disco": (
                "[Four-on-Floor Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[String Break] (4 bars)\n\n"
                "[Final Chorus]"
            ),
            "punk": (
                "[Fast Intro] (4 bars)\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus] (shout it)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge]\n{lyrics}\n\n"
                "[Final Chorus]"
            ),
            "ballad": (
                "[Piano Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge]\n{lyrics}\n\n"
                "[Final Chorus]"
            ),
            "retro": (
                "[Synthwave Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge]\n{lyrics}\n\n"
                "[Final Chorus]"
            ),
            "folk": (
                "[Acoustic Guitar Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge]\n{lyrics}\n\n"
                "[Final Chorus]"
            ),
            "chiptune": (
                "[8-bit Intro]\n\n"
                "[Level 1]\n{lyrics}\n\n"
                "[Boss Battle]\n{lyrics}\n\n"
                "[Level 2]\n{lyrics}\n\n"
                "[Power-Up]\n{lyrics}\n\n"
                "[Final Level]"
            ),
            "default": (
                "[Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Bridge]\n{lyrics}\n\n"
                "[Final Chorus]"
            )
        }

        short_structures = {
            "country": (
                "[Steel Guitar Intro]\n\n"
                "[Verse 1] (storytelling)\n{lyrics}\n\n"
                "[Chorus] (big melody)\n{lyrics}\n\n"
                "[Verse 2] (develop story)\n{lyrics}\n\n"
                "[Chorus] (big finish)"
            ),
            "pop": (
                "[Verse 1]\n{lyrics}\n\n"
                "[Pre-Chorus]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus] (with ad-libs)"
            ),
            "rock": (
                "[Guitar Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Guitar Solo] (4 bars)\n\n"
                "[Chorus] (big finish)"
            ),
            "hip hop": (
                "[Intro Hook]\n{lyrics}\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Hook]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Outro Hook]"
            ),
            "electronic": (
                "[Build-Up] (8 bars)\n{lyrics}\n\n"
                "[Drop]\n{lyrics}\n\n"
                "[Breakdown]\n{lyrics}\n\n"
                "[Drop] (final)"
            ),
            "lofi": (
                "[Ambient Intro] (with vinyl noise)\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chill Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Outro] (fade)"
            ),
            "jazz": (
                "[Piano Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Swing Chorus]\n{lyrics}\n\n"
                "[Brief Solo] (4 bars)\n\n"
                "[Outro]"
            ),
            "classical": (
                "[Introduction]\n\n"
                "[Theme A]\n{lyrics}\n\n"
                "[Theme B]\n{lyrics}\n\n"
                "[Recapitulation]\n{lyrics}"
            ),
            "ambient": (
                "[Textural Intro]\n\n"
                "[Main Section]\n{lyrics}\n\n"
                "[Fade Out]"
            ),
            "metal": (
                "[Heavy Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Breakdown] (4 bars)\n\n"
                "[Final Chorus]"
            ),
            "death metal": (
                "[Blast Intro]\n\n"
                "[Verse 1] (growled)\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Breakdown]\n\n"
                "[Final Blast]"
            ),
            "doom metal": (
                "[Slow Heavy Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus] (crushing)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Outro] (fade)"
            ),
            "reggae": (
                "[Skank Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus] (call-response)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus] (with harmonies)"
            ),
            "blues": (
                "[Guitar Lick Intro]\n\n"
                "[Verse 1] (12-bar)\n{lyrics}\n\n"
                "[Response] (guitar)\n\n"
                "[Verse 2] (12-bar)\n{lyrics}\n\n"
                "[Outro]"
            ),
            "delta blues": (
                "[Acoustic Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Slide Guitar Response]\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Outro] (fade)"
            ),
            "funk": (
                "[Groove Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus] (tight groove)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus] (vamp out)"
            ),
            "disco": (
                "[Four-on-Floor Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus] (danceable)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus] (with strings)"
            ),
            "punk": (
                "[Fast Intro] (4 bars)\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus] (shout it)\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus] (fast finish)"
            ),
            "ballad": (
                "[Piano Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus] (soft ending)"
            ),
            "retro": (
                "[Synthwave Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus] (fade out)"
            ),
            "folk": (
                "[Acoustic Guitar Intro]\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus] (campfire ending)"
            ),
            "chiptune": (
                "[8-bit Intro]\n\n"
                "[Level 1]\n{lyrics}\n\n"
                "[Boss Battle]\n{lyrics}\n\n"
                "[Power-Up]\n{lyrics}\n\n"
                "[Victory Theme]"
            ),
            "default": (
                "[Intro]\n{lyrics}\n\n"
                "[Verse 1]\n{lyrics}\n\n"
                "[Chorus]\n{lyrics}\n\n"
                "[Verse 2]\n{lyrics}\n\n"
                "[Chorus] (variation)\n\n"
                "[Outro]"
            )
        }

        if duration >= 180:
            structures = long_structures
        elif duration >= 120:
            structures = medium_structures
        else:
            structures = short_structures
        structure = structures.get(genre.lower(), structures["default"])

        prompt_addons = {
            "pop": "radio-ready, catchy hooks, polished production",
            "rock": "electric guitars, driving drums, raw energy",
            "electronic": "synthesizers, pulsing bass, euphoric drops",
            "lofi": "chill beats, vinyl crackle, relaxed vibe",
            "jazz": "smooth saxophone, walking bass, improvisational solos",
            "classical": "orchestral arrangements, dynamic phrasing, emotional depth",
            "ambient": "atmospheric pads, subtle textures, immersive soundscapes",
            "country": "steel guitar, fiddle, storytelling, twangy vocals",
            "metal": "distorted guitars, double bass drums, aggressive vocals",
            "reggae": "offbeat rhythms, organ skanks, dub effects",
            "blues": "bent notes, 12-bar structure, soulful vocals",
            "ballad": "emotional vocals, piano, heartfelt lyrics, slow tempo",
            "retro": "vintage synths, nostalgic melodies, 80s/90s vibe, groovy bass",
            "folk": "acoustic guitar, storytelling, earthy harmonies, natural sound",
            "chiptune": "8-bit sounds, video game melodies, retro synths, energetic beats",
            "default": "melodic, emotionally expressive, professional mix"
        }

        intensity_modifiers = {
            "soft": "gentle, subtle, delicate, mellow, calm",
            "medium": "balanced, moderate, steady",
            "high": "energetic, powerful, intense, loud, aggressive"
        }

        mood_modifiers = {
            "happy": "uplifting, joyful, positive, cheerful",
            "sad": "melancholic, sorrowful, emotional, touching",
            "reflective": "thoughtful, introspective, contemplative",
            "angry": "passionate, intense, raw, powerful",
            "upbeat": "lively, optimistic, vibrant, spirited",
            "chill": "relaxed, laid-back, peaceful, smooth"
        }

        with self.user_message_lock:
            user_msg = self.user_message

        user_message_text = f"User Message: \"{user_msg}\"\nConsider this message by the user, when writing the lyrics and theme.\n\n" if user_msg else ""

        prompt = (
            f"Write a {genre} song in {language} about '{theme}' using this exact structure:\n"
            f"{structure}\n\n"
            f"{user_message_text}"
            "STRICT REQUIREMENTS:\n"
            "1. Write ONLY the song lyrics - no translations, no explanations, no notes\n"
            "2. Use ONLY the specified language: {language}\n"
            "3. Follow the structure EXACTLY as shown\n"
            "4. Format each section header exactly as shown (e.g. [Verse 1])\n"
            f"5. The music must be no longer than {int(duration)}s so {int(duration * 1.2)} to {int(duration * 1.3)} words.\n"
            f"6. DO NOT surpass {int(duration * 1.3)} words. DO NOT fail to meet {int(duration * 1.2)} words. Keep to about {int(duration * 1.25 / structure.count('{lyrics}'))} words per section.\n"
            "7. Never include any text outside the {lyrics} structure. Do not include the \"{lyrics}\" itself.\n\n"
            "STYLE GUIDELINES:\n"
            f"- {prompt_addons.get(genre.lower(), prompt_addons['default'])}\n"
            f"- {intensity_modifiers.get(intensity, intensity_modifiers['medium'])} feel\n"
            f"- {mood_modifiers.get(mood, mood_modifiers['upbeat'])} mood\n"
            "- Use vivid imagery and emotional resonance\n"
            f"- Match the rhythm and phrasing to {genre} conventions\n"
            f"{'- Incorporate country idioms and themes' if genre.lower() == 'country' else ''}"
            f"{'- Use rap flow patterns and urban vocabulary' if genre.lower() == 'hip hop' else ''}\n\n"
        )

        try:
            response = ollama.generate(model=self.ollama_lyrics_model, prompt=prompt, keep_alive=0)
            lyrics = response['response'].strip()

            # Clean up the lyrics
            lyrics = self._clean_lyrics(lyrics)

            # Generate audio generation prompt
            audio_prompt = f"{genre} music ({prompt_addons.get(genre.lower(), prompt_addons['default'])}), {mood} mood ({mood_modifiers.get(mood, mood_modifiers['upbeat'])}), {intensity} intensity ({intensity_modifiers.get(intensity, intensity_modifiers['medium'])}), {tempo} BPM, Theme: {theme}"

            return lyrics, audio_prompt

        except Exception as e:
            print(f"⚠️  Error generating lyrics with Ollama: {e}")
            lyrics = self._fallback_lyrics(genre, theme)
            return lyrics, f"{genre} music about {theme}"

    def _clean_lyrics(self, lyrics_text: str) -> str:
        """Clean up generated lyrics"""
        # Remove <think>...</think> blocks
        lyrics_text = re.sub(r'<think>.*?</think>', '', lyrics_text, flags=re.DOTALL | re.IGNORECASE)
        # Remove common unwanted prefixes
        unwanted_prefixes = ["here are", "here's", "lyrics:", "song:"]
        lines = lyrics_text.split('\n')
        cleaned_lines = []

        for line in lines:
            line_lower = line.lower().strip()
            # Skip lines that are just explanations
            if any(line_lower.startswith(prefix) for prefix in unwanted_prefixes):
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()

    def _fallback_lyrics(self, genre: str, theme: str) -> str:
        """Generate simple fallback lyrics"""
        return f"""[Verse 1]
{theme}, {theme}
In the style of {genre}
Feeling the rhythm tonight

[Chorus]
This is our moment
This is our time
{theme} in the air
Everything feels right

[Verse 2]
Moving to the beat
Lost in the sound
{genre} taking over
Feel it all around

[Chorus]
This is our moment
This is our time
{theme} in the air
Everything feels right"""

    def generate_song(self, genre: str, theme: str, duration: float,
                      tempo: int, language: str, intensity: str, mood: str) -> Optional[Song]:
        """Generate a complete song"""

        start_time = time.time()
        timestamp = time.strftime("%Y%m%d%H%M%S")

        try:
            if self.state == RadioState.BUFFERING:
                print("📜 Generating lyrics...")
            # Generate lyrics
            lyrics, audio_prompt = self.generate_lyrics_with_ollama(
                genre, theme, duration, language, tempo, intensity, mood
            )

            # Generate title
            title = f"{theme.title()} ({genre.title()})"
            artist = "AI Radio"

            # Save lyrics
            lyrics_path = self.output_dir / f"song_{timestamp}_lyrics.txt"
            with open(lyrics_path, 'w', encoding='utf-8') as f:
                f.write(lyrics)

            if self.state == RadioState.BUFFERING:
                print("🎵 Generating audio...")
            # Generate audio
            pipeline = self.get_pipeline()

            results = None
            for attempt in range(3):  # Retry up to 3 times
                try:
                    # Add torch.inference_mode() to disable gradient tracking
                    with torch.inference_mode():
                        results = pipeline(
                            audio_duration=duration,
                            prompt=audio_prompt,
                            lyrics=lyrics,
                            infer_step=30,
                            guidance_scale=15.0,
                            scheduler_type="euler",
                            cfg_type="apg",
                            omega_scale=10.0,
                            batch_size=1
                        )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()

                    break

                except Exception as e:
                    print(f"⚠️  Audio generation attempt {attempt + 1} failed: {e}")
                    continue

            if not results or len(results) < 2:
                raise RuntimeError("Audio generation failed after multiple attempts.")

            audio_path = results[0]
            metadata = results[-1]

            # Get actual duration of generated audio file
            try:
                import librosa
                actual_duration = librosa.get_duration(path=audio_path)

                # If duration is significantly different, adjust metadata
                if abs(actual_duration - duration) > 5.0:
                    print(f"Note: Actual duration differs from target by {abs(actual_duration - duration):.2f}s")
                    metadata['actual_duration'] = actual_duration
                    metadata['target_duration'] = duration
            except Exception as e:
                print(f"Couldn't measure audio duration: {e}")
                actual_duration = duration  # Fallback to target duration

            generation_time = time.time() - start_time

            song = Song(
                title=title,
                artist=artist,
                genre=genre,
                theme=theme,
                duration=actual_duration,
                lyrics=lyrics,
                language=language,
                prompt=audio_prompt,
                audio_path=audio_path,
                generation_time=generation_time,
                timestamp=time.time(),
                tempo=tempo,
                intensity=intensity,
                mood=mood,
                metadata={"lyrics_path": str(lyrics_path), **metadata}
            )

            return song

        except Exception as e:
            print(f"❌ Error generating song: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Always clean up after generation attempt
            self.clean_all_memory()

    def _generation_worker(self):
        """Background worker for continuous song generation"""
        print("🔧 Song generation worker started")

        while not self.stop_event.is_set():
            try:
                # Clean memory at start of loop
                self.clean_all_memory()

                # Check if queue needs more songs
                if self.song_queue.full():
                    time.sleep(1)
                    continue

                # Generate song parameters
                params = self.generate_parameters_from_screen_with_ollama()

                self.playback_history.append("[ Theme: " + params['theme'] + " | Genre: " + params['genre'] + "]")
                if len(self.playback_history) > 3:
                    self.playback_history = self.playback_history[-3:]

                self.last_genres.append(params["genre"])
                if len(self.last_genres) > 3:
                    self.last_genres = self.last_genres[-3:]

                # Generate song
                song = self.generate_song(**params)

                if song and not self.stop_event.is_set():
                    self.song_queue.put(song)

            except Exception as e:
                print(f"❌ Generation worker error: {e}")
                import traceback
                traceback.print_exc()
                # Ensure cleanup on error
                self.clean_all_memory()
                time.sleep(5)
            finally:
                # Always clean after generation attempt
                self.clean_all_memory()

        print("🔧 Song generation worker stopped")

    def _playback_worker(self):
        """Background worker for continuous playback"""
        print("🔊 Playback worker started")

        while not self.stop_event.is_set():
            try:
                self.state = RadioState.BUFFERING
                song = self.song_queue.get(block=True)

                if self.stop_event.is_set():
                    break

                # Play the song
                self.state = RadioState.PLAYING
                self._play_song(song)

            except Exception as e:
                print(f"❌ Playback worker error: {e}")
                time.sleep(1)

        print("🔊 Playback worker stopped")

    def _play_song(self, song: Song):
        """Play a single song with skip/restart support and clean up previous files after playback"""
        self.current_song = song
        self.state = RadioState.PLAYING

        print(f"{'='*10}")
        print(f"{song.title}")
        print(f"{song.mood.title()}, {song.tempo} BPM, {song.intensity.title()} Intensity")
        print(f"{'='*10}")
        print(f"{song.lyrics}")

        # Track previous song for cleanup
        prev_song = getattr(self, '_last_played_song', None)

        # Load and play audio with pygame
        try:
            with self.playback_lock:
                pygame.mixer.music.load(song.audio_path)
                pygame.mixer.music.play()
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return

        # Play until song ends or skip/restart is triggered
        while not self.stop_event.is_set():
            # Check for pause/unpause
            if self.pause_event.is_set():
                self.pause_event.clear()
                with self.playback_lock:
                    if self.is_paused:
                        pygame.mixer.music.unpause()
                        self.is_paused = False
                        print("▶️  Resumed")
                    else:
                        pygame.mixer.music.pause()
                        self.is_paused = True
                        print("⏸️  Paused")

            # Check for restart
            if self.restart_event.is_set():
                self.restart_event.clear()
                with self.playback_lock:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.play()
                    self.is_paused = False
                print("🔄 Restarting song...")

            # Check for skip
            if self.skip_event.is_set():
                self.skip_event.clear()
                with self.playback_lock:
                    pygame.mixer.music.stop()
                    self.is_paused = False
                print("⏭️  Skipping song...")
                break

            # Check if song is still playing (only when not paused)
            with self.playback_lock:
                if not self.is_paused and not pygame.mixer.music.get_busy():
                    # Song finished naturally
                    break

            # Small sleep to avoid busy waiting
            time.sleep(0.1)

        # Clean up
        with self.playback_lock:
            pygame.mixer.music.stop()
            self.is_paused = False

        # Mark this song as last played for next cleanup
        self._last_played_song = song

        # Clean up previous song's files (not the current one)
        if prev_song is not None:
            try:
                # Remove .wav file
                if prev_song.audio_path and os.path.exists(prev_song.audio_path):
                    os.remove(prev_song.audio_path)
                # Remove .json file (input params)
                base = os.path.splitext(os.path.basename(prev_song.audio_path))[0]
                json_path = os.path.join(str(self.output_dir), base + '_input_params.json')
                if os.path.exists(json_path):
                    os.remove(json_path)
                # Remove lyrics .txt file
                lyrics_path = prev_song.metadata.get('lyrics_path')
                if lyrics_path and os.path.exists(lyrics_path):
                    os.remove(lyrics_path)
            except Exception as e:
                print(f"⚠️  Error cleaning up previous song files: {e}")

        self.current_song = None

    def start(self):
        """Start the radio station"""
        if self.state != RadioState.STOPPED:
            print("⚠️  Radio is already running")
            return

        self.stop_event.clear()
        self.skip_event.clear()
        self.restart_event.clear()
        self.pause_event.clear()
        self.is_paused = False
        self.state = RadioState.BUFFERING

        print("\n" + "=" * 60)
        print("🎵 AI RADIO STATION STARTING...")
        print("=" * 60)

        # Start worker threads
        self.generation_thread = threading.Thread(target=self._generation_worker, daemon=True)
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)

        self.generation_thread.start()
        self.playback_thread.start()

        print("✅ Radio started successfully!")
        print("⌨️  Press [N] to skip | [R] to restart | [P] to pause/unpause | [I] to edit message | [Q] to quit\n")

    def stop(self):
        """Stop the radio station"""
        print("\n🛑 Stopping radio station...")
        self.stop_event.set()
        self.state = RadioState.STOPPED

        # Stop audio playback
        with self.playback_lock:
            pygame.mixer.music.stop()

        # Wait for threads
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=5)
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=5)

        # Cleanup all memory
        self.clean_all_memory()
        pygame.mixer.quit()

        # Clean up output folder on termination
        self._cleanup_output_folder()

        print("✅ Radio stopped")

    def skip_song(self):
        """Skip the current song"""
        if self.state == RadioState.PLAYING:
            self.skip_event.set()

    def restart_song(self):
        """Restart the current song"""
        if self.state == RadioState.PLAYING:
            self.restart_event.set()

    def toggle_pause(self):
        """Toggle pause/unpause"""
        if self.state == RadioState.PLAYING:
            self.pause_event.set()

    def edit_user_message(self):
        """Edit the user message for personalized music generation"""

        with self.user_message_lock:
            current_msg = self.user_message

        try:
            # Save current terminal settings and restore to normal (cooked) mode
            fd = sys.stdin.fileno()
            new_settings = termios.tcgetattr(fd)

            # Restore to cooked mode (canonical mode with echo)
            new_settings[3] = new_settings[3] | termios.ICANON | termios.ECHO
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

            # Flush any pending input
            termios.tcflush(fd, termios.TCIFLUSH)

            # Pre-fill the input buffer with the current message
            def prefill_input():
                readline.insert_text(current_msg)
                readline.redisplay()

            readline.set_pre_input_hook(prefill_input)

            try:
                new_message = input("\nMessage: ").strip()
            finally:
                # Clear the pre-input hook
                readline.set_pre_input_hook()

            with self.user_message_lock:
                self.user_message = new_message

            # Set terminal back to raw mode (cbreak)
            tty.setcbreak(fd)

        except Exception as e:
            print(f"❌ Error editing user message: {e}")
            # Try to restore terminal mode even on error
            try:
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                pass


# ============================================================================
# Keyboard Input Handler
# ============================================================================

class KeyboardHandler:
    def __init__(self, radio: AIRadioStation):
        self.radio = radio
        self.running = False
        self.input_thread = None
        self.old_settings = None

    def _input_worker(self):
        """Worker thread to read keyboard input"""
        while self.running:
            try:
                # Check if input is available (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1).lower()

                    if char == 'n':
                        self.radio.skip_song()
                    elif char == 'r':
                        self.radio.restart_song()
                    elif char == 'p':
                        self.radio.toggle_pause()
                    elif char == 'i':
                        self.radio.edit_user_message()
                    elif char == 'q':
                        print("\n👋 Quitting...")
                        self.running = False
                        self.radio.stop()
                        break
            except Exception:
                # Silently ignore input errors
                pass

    def start(self):
        """Start listening for keyboard input"""
        self.running = True

        # Set terminal to raw mode for single character input
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            print("⚠️  Warning: Could not set terminal to raw mode. Input may not work properly.")

        # Start input thread
        self.input_thread = threading.Thread(target=self._input_worker, daemon=True)
        self.input_thread.start()

    def stop(self):
        """Stop listening for keyboard input"""
        self.running = False

        # Restore terminal settings
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

        # Wait for input thread
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1)
# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="AI Radio Station - Continuous music generation with random parameters"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Path to ACE-Step model checkpoints"
    )
    parser.add_argument(
        "--ollama_vision_model",
        type=str,
        default="minicpm-v",
        help="Ollama model to use for screen analysis and music parameter generation"
    )
    parser.add_argument(
        "--ollama_lyrics_model",
        type=str,
        default="gemma3:12b-it-q4_K_M",
        help="Ollama model to use for lyric generation"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="GPU device ID to use"
    )

    args = parser.parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # Initialize radio
    print("🎵 Initializing AI Radio Station...")
    radio = AIRadioStation(
        checkpoint_dir=args.checkpoint_dir,
        ollama_lyrics_model=args.ollama_lyrics_model,
        ollama_vision_model=args.ollama_vision_model
    )

    # Initialize keyboard handler
    keyboard_handler = KeyboardHandler(radio)

    try:
        # Start radio
        radio.start()
        keyboard_handler.start()

        # Keep main thread alive
        while keyboard_handler.running:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")

    finally:
        # Cleanup
        keyboard_handler.stop()
        radio.stop()
        print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    main()
