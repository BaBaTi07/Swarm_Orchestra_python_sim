import numpy as np

def angle_to_sector(rel_angle_rad: float, fov_rad: float, n_sectors: int) -> int:
    """
    rel_angle_rad: angle relatif entre receveur et émetteur
    fov_rad: champ de vision du receveur en radians
    n_sectors: nombre de secteurs à discrétiser dans le champ de vision
    Retourne l'index du secteur (0..n-1)
    """
    n = int(max(1, n_sectors))
    half = 0.5 * float(fov_rad)

    # clamp 
    a = float(np.clip(rel_angle_rad, -half, half))

    # normaliser sur [0,1]
    x = (a + half) / (2.0 * half) if half > 0 else 0.5

    # index [0..n-1]
    idx = int(np.floor(x * n))
    if idx >= n:
        idx = n - 1
    return idx