import numpy as np


class RandomFunctionGenerator:
    def __init__(self) -> None:
        self.pF = np.zeros(8)
        self.sF = np.zeros(8)
        self.cF = np.zeros(8)

    def generate(self) -> None:
        for i in range(8):
            self.pF[i], self.sF[i], self.cF[i] = np.floor(np.random.random() * 3), np.floor(
                np.random.random() * 3), np.floor(np.random.random() * 3)

    def __call__(self, x, y) -> float:
        if not np.all(self.pF) and not np.all(self.cF) and not np.all(self.sF):
            return self.pF[0] * (self.pF[1] * x - self.pF[2]) ** self.pF[3] + self.pF[4] * (
                    self.pF[5] * y - self.pF[6]) ** self.pF[7] + self.sF[0] * np.sin(
                self.sF[1] * x - self.sF[2]) ** \
                self.sF[3] + self.sF[4] * np.sin(self.sF[5] * y - self.sF[6]) ** self.sF[7] + self.cF[0] * np.cos(
                    self.cF[1] * x - self.cF[2]) ** self.cF[3] + self.cF[4] * np.cos(
                    self.cF[5] * y - self.cF[6]) ** \
                self.cF[7]
