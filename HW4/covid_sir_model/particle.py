"""
To model 2D elastic collisions, I refered the codes in a github repository "pyCollisions2D".

@Citation

Copyright (c) 2014 Pablo Caro. All Rights Reserved.

Pablo Caro <me@pcaro.es> - http://pcaro.es/

url: https://github.com/pcaro90/pyCollisions2D.git
"""

import numpy as np
from config import (
    AGENT_MASS,
    AGENT_RADIUS,
    AGENT_SPEED,
    NUM_AGENTS,
    SPACE_SIZE,
)


class Particle:
    def __init__(self, isolated, id=0):
        self.id = id
        self.s = [np.random.uniform(0, SPACE_SIZE[i] - 1) for i in range(2)]
        self.r = AGENT_RADIUS
        self.isolated = id < NUM_AGENTS * isolated

        if self.isolated:
            self.m = 100
            self.v = [0, 0]
        else:
            self.m = AGENT_MASS
            angle = np.random.uniform(0, 2 * np.pi)
            self.v = [AGENT_SPEED * np.cos(angle), AGENT_SPEED * np.sin(angle)]

        self.a = [0.0] * 2

    def move(self, ms):
        for i in range(len(self.s)):
            self.v[i] += self.a[i] * (ms / 1000.0)
            self.s[i] += self.v[i] * (ms / 1000.0)

    def distance(self, p):
        d = 0.0
        for x1, x2 in zip(self.s, p.s, strict=False):
            d += abs(x1 - x2) ** 2.0
        return d**0.5


def random_particles(isolated, n):
    particles = []
    for i in range(n):
        inserted = False
        while not inserted:
            np = Particle(isolated, i)
            for p in particles:
                if p.distance(np) < p.r + np.r:
                    break
            else:
                inserted = True
        particles.append(np)

    return particles


def dot_product(v1, v2):
    r = 0.0
    for a, b in zip(v1, v2, strict=False):
        r += a * b
    return r


def scalar_product(v, n):
    return [i * n for i in v]


def normalize(v):
    m = 0.0
    for spam in v:
        m += spam**2.0
    m = m**0.5

    return [spam / m for spam in v]
