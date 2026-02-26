import os
from pathlib import Path
from typing import Optional

import time
import threading

import numpy as np
import pygame
from dataclasses import dataclass

@dataclass(frozen=True)
class NoteEvent:
    midi: int
    duration_s: float = 0.25
    volume: float = 0.6  # 0..1

class MusicModule:
    """
    1 module par robot (1 channel dédié).
    Génère un son simple (sine)
    ou joue un fichier audio prédéfini
    pour une note MIDI.
    """

    _is_init = False
    _sample_rate = 44100 # FPS du son
    _cache: dict[tuple[int, float], pygame.mixer.Sound] = {}
    _cache_sample : dict[tuple[str,int], pygame.mixer.Sound] = {}

    _instrument_dir: Optional[Path] = None
    _instrument_key: str = "default"
    _extensions = (".wav", ".ogg")

    @staticmethod
    def init_global(sample_rate: int = 44100, channels: int = 2, buffer: int = 512, num_mixer_channels: int = 64):
        """
        À appeler UNE fois au lancement (avant de jouer des sons).
        """
        if MusicModule._is_init:
            return

        MusicModule._sample_rate = int(sample_rate)
        pygame.mixer.pre_init(frequency=MusicModule._sample_rate, size=-16, channels=channels, buffer=buffer)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(int(num_mixer_channels))
        MusicModule._is_init = True

    def __init__(self, channel_id: int):
        if not MusicModule._is_init:
            raise RuntimeError("MusicModule.init_global() must be called before creating MusicModule instances.")
        self.channel_id = int(channel_id)
        self.channel = pygame.mixer.Channel(self.channel_id)
        self._play_token = 0

    @staticmethod
    def set_instrument_samples(instrument_dir: str | Path, instrument_key: str = "default", preload: bool = False):
        """
        Définit un dossier d'instrument contenant des samples nommés par midi:
          60.wav, 61.wav, ...
        Si preload=True, charge tout de suite tous les fichiers trouvés (plus rapide ensuite).
        """
        p = Path(instrument_dir)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Instrument directory not found: {p}")

        MusicModule._instrument_dir = p
        MusicModule._instrument_key = str(instrument_key)

        if preload:
            for f in p.iterdir():
                if f.is_file() and f.suffix.lower() in MusicModule._extensions:
                    # nom = "60.wav" -> midi=60
                    try:
                        midi = int(f.stem)
                    except ValueError:
                        continue
                    MusicModule._load_sample(midi)

    @staticmethod
    def _find_sample_path(midi: int) -> Optional[Path]:
        """
        Cherche un fichier 'midi.ext' dans le dossier instrument.
        """
        if MusicModule._instrument_dir is None:
            return None
        base = MusicModule._instrument_dir / str(int(midi))
        for ext in MusicModule._extensions:
            candidate = base.with_suffix(ext)
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _load_sample(midi: int) -> Optional[pygame.mixer.Sound]:
        """
        Charge un sample depuis disque et le met en cache.
        Retourne None si pas trouvé.
        """
        if MusicModule._instrument_dir is None:
            return None

        key = (MusicModule._instrument_key, int(midi))
        if key in MusicModule._cache_sample:
            return MusicModule._cache_sample[key]

        path = MusicModule._find_sample_path(midi)
        if path is None:
            return None

        snd = pygame.mixer.Sound(str(path))
        MusicModule._cache_sample[key] = snd
        return snd


    @staticmethod
    def midi_to_freq(midi: int) -> float:
        return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))

    @staticmethod
    def _sine_sound(midi: int, duration_s: float) -> pygame.mixer.Sound:
        """
        Génère un Sound stéréo 16-bit, avec petite enveloppe pour éviter les clics.
        Cache pour éviter de regénérer tout le temps.
        """
        duration_s = float(duration_s)
        key = (int(midi), round(duration_s, 4))
        if key in MusicModule._cache:
            return MusicModule._cache[key]

        sr = MusicModule._sample_rate
        n = max(1, int(sr * duration_s))
        t = np.linspace(0.0, duration_s, n, endpoint=False)
        freq = MusicModule.midi_to_freq(midi)

        wave = np.sin(2.0 * np.pi * freq * t)

        # Enveloppe simple (attack/release) pour éviter "click"
        a = int(0.01 * sr)  # 10ms
        r = int(0.02 * sr)  # 20ms
        env = np.ones_like(wave)
        if a > 1:
            env[:a] = np.linspace(0.0, 1.0, a)
        if r > 1:
            env[-r:] = np.linspace(1.0, 0.0, r)
        wave = wave * env

        stereo = np.column_stack([wave, wave])

        # 16-bit signed
        audio = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)

        sound = pygame.sndarray.make_sound(audio)
        MusicModule._cache[key] = sound
        return sound

    def play_note(self, midi: int, duration_s: float = 0.25, volume: float = 0.6,
                   stop_previous: bool = True, prefer_sample: bool = True, fadeout_ms: float = 30.0, one_note_at_a_time = False):
        if one_note_at_a_time and self.channel.get_busy():
            return
        if stop_previous:
            self.channel.stop()
        self.channel.set_volume(float(volume))

        duration_s = max(0.0, float(duration_s))
        fadeout_ms = max(0.0, float(fadeout_ms))

        snd: Optional[pygame.mixer.Sound] = None
        if prefer_sample:
            snd = MusicModule._load_sample(int(midi))
        if snd is None:
            snd = MusicModule._sine_sound(int(midi), float(duration_s))
            self.channel.play(snd)
            return
        
        if duration_s <=0.0:
            return
        
        # Utiliser un timer pour arrêter le son après la durée spécifiée
        self._play_token += 1
        token = self._play_token
        self.channel.play(snd)

        if fadeout_ms > 0.0:
            note_off_at = max(0.0, duration_s - fadeout_ms / 1000.0)  #start the fadout before the end to end on time
        else:
            note_off_at = duration_s

        def _note_off_latter(expected_token: int, delay_s: float):
            time.sleep(delay_s)
            if self._play_token != expected_token:
                return # new note has been played since, do not stop
            if fadeout_ms > 0.0:
                self.channel.fadeout(int(fadeout_ms))
            else:
                self.channel.stop()
        
        threading.Thread(
            target=_note_off_latter,
            args=(token, note_off_at), 
            daemon=True
        ).start()



    def stop(self):
        self.channel.stop()
