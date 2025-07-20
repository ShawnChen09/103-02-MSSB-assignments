import numpy as np
from config import (
    DEATH_PROB,
    INFECT_DIST,
    INFECT_PERIOD,
    REINFECT_PROB,
)


def init_health_states(num_agents, initial_infected):
    # Health states: 0 = susceptible, 1 = infected, 2 = recovered, -1 = dead
    health_states = np.zeros(num_agents, dtype=int)
    infection_timers = np.zeros(num_agents, dtype=int)

    for idx in np.random.choice(num_agents, initial_infected, replace=False):
        health_states[idx] = 1

    return health_states, infection_timers


def update_health_states(
    particles, health_states, infect_prob, infection_timers
):
    newly_infected = []

    # Process infection spreading
    for i in range(len(particles)):
        if health_states[i] == 1:
            infection_timers[i] += 1

            for j in range(len(particles)):
                if i == j:
                    continue

                dist = particles[i].distance(particles[j])

                if health_states[j] == 0 and dist < INFECT_DIST:
                    if np.random.random() < infect_prob:
                        health_states[j] = 1
                        newly_infected.append(j)

                elif health_states[j] == 2 and dist < INFECT_DIST:
                    if np.random.random() < REINFECT_PROB:
                        health_states[j] = 1
                        infection_timers[j] = 0
                        newly_infected.append(j)

    # Process recoveries and deaths
    for i in range(len(particles)):
        if health_states[i] == 1 and infection_timers[i] >= INFECT_PERIOD:
            if np.random.random() < DEATH_PROB:
                health_states[i] = -1  # Died
            else:
                health_states[i] = 2  # Recovered

    return health_states, infection_timers


def count_health_states(health_states):
    s_count = np.sum(health_states == 0)
    i_count = np.sum(health_states == 1)
    r_count = np.sum(health_states == 2)
    return s_count, i_count, r_count
