import os
import time

import numpy as np
import pygame
from config import (
    COLORS,
    INITIAL_INFECTED,
    MAX_FPS,
    NUM_AGENTS,
    SPACE_SIZE,
    STEP_PER_DAY,
)
from health import (
    count_health_states,
    init_health_states,
    update_health_states,
)
from particle import dot_product, normalize, random_particles, scalar_product


def run_simulation(
    infect_prob, isolated, save_frames=False, output_dir="frames"
):
    pygame.init()
    windowSurface = pygame.display.set_mode(SPACE_SIZE, 0, 32)

    # Create output directory for frames if saving is enabled
    if save_frames:
        os.makedirs(output_dir, exist_ok=True)

    # Font for displaying statistics
    font = pygame.font.SysFont("Arial", 24)

    mainClock = pygame.time.Clock()

    # Initialize particles
    particles = random_particles(isolated, NUM_AGENTS)

    # Initialize health states
    health_states, infection_timers = init_health_states(
        NUM_AGENTS, INITIAL_INFECTED
    )

    frame_count = 0
    history = {"s": [], "i": [], "r": [], "d": []}

    end = False
    while not end:
        t = mainClock.tick(MAX_FPS)
        frame_count += 1

        # Process events to prevent UI freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                end = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    end = True

        # Particles collision
        for i, p1 in enumerate(particles):
            for p2 in particles[i + 1 :]:
                d = p1.distance(p2)
                if d <= p1.r + p2.r:
                    N = normalize([p1.s[0] - p2.s[0], p1.s[1] - p2.s[1]])

                    d1 = 1.1 * ((p1.r + p2.r - d) * p2.m) / (p1.m + p2.m)
                    d2 = 1.1 * ((p1.r + p2.r - d) * p1.m) / (p1.m + p2.m)

                    p1.s[0] += N[0] * d1
                    p1.s[1] += N[1] * d1

                    p2.s[0] -= N[0] * d2
                    p2.s[1] -= N[1] * d2

                    T = [-N[1], N[0]]

                    v1n = dot_product(N, p1.v)
                    v1t = dot_product(T, p1.v)

                    v2n = dot_product(N, p2.v)
                    v2t = dot_product(T, p2.v)

                    u1n = v1n
                    v1n = (v1n * (p1.m - p2.m) + 2.0 * p2.m * v2n) / (
                        p1.m + p2.m
                    )
                    v2n = (v2n * (p2.m - p1.m) + 2.0 * p1.m * u1n) / (
                        p2.m + p1.m
                    )

                    vn = scalar_product(N, v1n)
                    vt = scalar_product(T, v1t)
                    p1.v = [a + b for a, b in zip(vn, vt, strict=False)]

                    vn = scalar_product(N, v2n)
                    vt = scalar_product(T, v2t)
                    p2.v = [a + b for a, b in zip(vn, vt, strict=False)]

        # Bounce on edges
        for p in particles:
            for i in range(2):
                if p.s[i] < p.r and p.v[i] < 0:
                    p.v[i] = -p.v[i]
                elif p.s[i] + p.r > SPACE_SIZE[i] and p.v[i] > 0:
                    p.v[i] = -p.v[i]

        # Move particles
        for p in particles:
            if not p.isolated:
                p.move(t)

        # Update health states
        health_states, infection_timers = update_health_states(
            particles, health_states, infect_prob, infection_timers
        )

        # Update count statistics
        s_count, i_count, r_count = count_health_states(health_states)
        d_count = NUM_AGENTS - (s_count + i_count + r_count)

        # Store history for plotting
        history["s"].append(s_count)
        history["i"].append(i_count)
        history["r"].append(r_count)
        history["d"].append(d_count)

        windowSurface.fill((255, 255, 255))

        # Update particles and draw with colors based on health state
        tmp_particles = []
        tmp_health_states = []
        tmp_infection_timers = []
        for i, p in enumerate(particles):
            if health_states[i] != -1:
                color = COLORS[health_states[i]]
                pygame.draw.circle(
                    windowSurface, color, [int(p.s[0]), int(p.s[1])], int(p.r)
                )
                tmp_particles.append(p)
                tmp_health_states.append(health_states[i])
                tmp_infection_timers.append(infection_timers[i])

        particles = np.array(tmp_particles)
        health_states = np.array(tmp_health_states)
        infection_timers = np.array(tmp_infection_timers)

        # Display statistics
        days = frame_count // STEP_PER_DAY
        hours = frame_count % STEP_PER_DAY
        stats_text = f"{days} Day {hours} Hours | S: {s_count} | I: {i_count} | R: {r_count} | D: {d_count}"
        stats_surf = font.render(stats_text, True, (0, 0, 0))
        windowSurface.blit(stats_surf, (10, 10))

        if i_count == 0:
            summary_text = f"Simulation complete. Final: S:{s_count} I: {i_count} R:{r_count} D:{d_count}"
            print(summary_text)
            summary_surf = font.render(summary_text, True, (255, 0, 0))
            windowSurface.blit(summary_surf, (10, 40))
            end = True

        pygame.display.update()

        if save_frames:
            pygame.image.save(
                windowSurface,
                os.path.join(output_dir, f"frame_{frame_count:05d}.png"),
            )

    time.sleep(0.5)
    pygame.quit()

    return history
