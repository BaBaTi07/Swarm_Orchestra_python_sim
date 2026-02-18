# AUDIO/music_module.py
from __future__ import annotations

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
    Génère un son simple (sine) pour une note MIDI.
    """

    _is_init = False
    _sample_rate = 44100
    _cache: dict[tuple[int, float], pygame.mixer.Sound] = {}

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

        # stéréo
        stereo = np.column_stack([wave, wave])

        # 16-bit signed
        audio = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)

        sound = pygame.sndarray.make_sound(audio)
        MusicModule._cache[key] = sound
        return sound

    def play_note(self, midi: int, duration_s: float = 0.25, volume: float = 0.6, stop_previous: bool = True):
        if stop_previous:
            self.channel.stop()
        self.channel.set_volume(float(volume))
        sound = MusicModule._sine_sound(int(midi), float(duration_s))
        self.channel.play(sound)

    def stop(self):
        self.channel.stop()
