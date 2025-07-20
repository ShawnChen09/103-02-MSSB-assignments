STEP_PER_DAY = 24

# Engine parameters
SPACE_SIZE = (500, 500)
MAX_FPS = 120

# Colors
COLORS = {
    0: (0, 0, 255),  # Susceptible - Blue
    1: (255, 0, 0),  # Infected - Red
    2: (0, 255, 0),  # Recovered - Green
}

# Particle parameters
NUM_AGENTS = 400
AGENT_RADIUS = 5
AGENT_SPEED = 150
AGENT_MASS = 0.1

# Disease parameters
INFECTED_PROP = 0.01
INITIAL_INFECTED = int(NUM_AGENTS * INFECTED_PROP)
INFECT_DIST = AGENT_RADIUS * 2 + 1
REINFECT_PROB = 0.00
INFECT_PERIOD = 30 * STEP_PER_DAY
DEATH_PROB = 0.0044
