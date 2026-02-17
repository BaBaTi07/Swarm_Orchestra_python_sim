class MusicModule:
    """
    Placeholder: génération/planification de notes, tempo, etc.
    """
    def __init__(self):
        self.current_note = None
        self.is_playing = False

    def play_note(self, note, duration_s: float, velocity: float = 1.0):
        self.current_note = (note, duration_s, velocity)
        self.is_playing = True

    def stop(self):
        self.current_note = None
        self.is_playing = False
