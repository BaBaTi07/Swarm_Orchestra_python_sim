
def note_to_color(note: int) -> tuple[float, float, float]:
    """
    Map pitch-class (note % 12) to RGB color.
    """
    pc = int(note) % 12

    # 12 couleurs (RGB) : ajuste si tu veux un autre arc-en-ciel
    palette = {
        0:  (1.0, 0.0, 0.0),   # C  (Do)        -> rouge
        1:  (1.0, 0.5, 0.0),   # C# / Db (Re♭)  -> orange
        2:  (1.0, 1.0, 0.0),   # D  (Re)        -> jaune
        3:  (0.6, 1.0, 0.0),   # D# / Eb        -> jaune-vert
        4:  (0.0, 1.0, 0.0),   # E  (Mi)        -> vert
        5:  (0.0, 1.0, 1.0),   # F              -> cyan
        6:  (0.0, 0.6, 1.0),   # F# / Gb        -> bleu clair
        7:  (0.0, 0.0, 1.0),   # G              -> bleu
        8:  (0.5, 0.0, 1.0),   # G# / Ab        -> violet
        9:  (1.0, 0.0, 1.0),   # A              -> magenta
        10: (1.0, 0.2, 0.6),   # A# / Bb        -> rose
        11: (1.0, 1.0, 1.0),   # B (Si)         -> blanc
    }
    return palette.get(pc, (1.0, 1.0, 1.0))