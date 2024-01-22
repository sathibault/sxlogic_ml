import numpy as np

class Hist:
    def __init__(self):
        self.cur_min = 1e9
        self.cur_max = -1e9

    def add(self, x):
        mn = np.min(x)
        mx = np.max(x)
        if mn < self.cur_min:
            self.cur_min = mn
        if mx > self.cur_max:
            self.cur_max = mx

    def range(self):
        return [self.cur_min, self.cur_max]

    def absmax(self):
        [a,b] = self.range()
        return max(abs(a),abs(b))
