#List object of the differents musical scales

class Scale:
    def __init__(self, name, notes):
        self.name = name
        self.notes = notes

# c = 0, c# = 1, d = 2, d# = 3, e = 4, e#=f = 5, f# = 6, g = 7, g# = 8, a = 9, a# = 10, b = 11, c = 12, ...
MajorScales = [
    Scale("C Major", [0, 2, 4, 5, 7, 9, 11]),
    Scale("G Major", [7, 9, 11, 0, 2, 4, 6]),
    Scale("D Major", [2, 4, 6, 7, 9, 11, 1]),
    Scale("A Major", [9, 11, 1, 2, 4, 6, 8]),
    Scale("E Major", [4, 6, 8, 9, 11, 1, 3]),
    Scale("B Major", [11, 1, 3, 4, 6, 8, 10]),
    Scale("F# Major", [6, 8, 10, 11, 1, 3, 5]),
    Scale("C# Major", [1, 3, 5, 6, 8, 10, 0]),
    Scale("F Major", [5, 7, 9, 10, 0, 2, 4]),
    Scale("Bb Major", [10, 0, 2, 3, 5, 7, 9]),
    Scale("Eb Major", [3, 5, 7, 8, 10, 0, 2]),
    Scale("Ab Major", [8, 10, 0, 1, 3, 5, 7]),
]
MinorScales = []

Scales = MajorScales + MinorScales