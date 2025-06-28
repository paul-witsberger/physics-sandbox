from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np
import random

# This script creates an animated bouncing, rolling ball with friction, gravity, and aerodynamic drag.

class Ball:
    def __init__(self, radius=0.1, position=(0, 1), velocity=(1, 0), gravity=9.81,
                 restitution=0.9, rolling_friction=0.9, drag_coeff=0.05, min_x=-5, max_x=5, min_y=0, max_y=5):
        self.radius = radius
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.gravity = gravity
        self.restitution = restitution  # Coefficient of restitution (energy loss on bounce)
        self.rolling_friction = rolling_friction  # Coefficient of rolling friction
        self.drag_coeff = drag_coeff  # Aerodynamic drag coefficient
        self.min_x = min_x + radius
        self.max_x = max_x - radius
        self.min_y = min_y + radius
        self.max_y = max_y - radius

    def update(self, dt):
        # Apply aerodynamic drag: F_drag = -c * v^2 * sign(v)
        drag_force = -self.drag_coeff * self.velocity * np.abs(self.velocity)
        self.velocity += drag_force * dt
        # Gravity
        self.velocity[1] -= self.gravity * dt
        self.position += self.velocity * dt

        # Ground (y min)
        if self.position[1] <= self.min_y:
            self.position[1] = self.min_y
            self.velocity[1] *= -self.restitution
            if abs(self.velocity[1]) < 0.01:
                self.velocity[1] = 0
        # Ceiling (y max)
        if self.position[1] >= self.max_y:
            self.position[1] = self.max_y
            self.velocity[1] *= -self.restitution
        # Left wall (x min)
        if self.position[0] <= self.min_x:
            self.position[0] = self.min_x
            self.velocity[0] *= -self.restitution
        # Right wall (x max)
        if self.position[0] >= self.max_x:
            self.position[0] = self.max_x
            self.velocity[0] *= -self.restitution
        # Rolling friction
        if self.position[1] == self.min_y:  # Only apply friction when on the ground
            self.velocity[0] *= self.rolling_friction
        # Stop rolling if velocity is very low
        if abs(self.velocity[0]) < 0.01:
            self.velocity[0] = 0
            self.velocity[1] = 0  # Stop vertical motion as well if rolling stops

def animate_ball(ball, dt=0.05, frames=200):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#181825')  # Set figure background
    ax.set_facecolor("#474952")         # Set axes background
    ax.set_xlim(ball.min_x - ball.radius, ball.max_x + ball.radius)
    ax.set_ylim(ball.min_y - ball.radius, ball.max_y + ball.radius)
    ax.set_aspect('equal')
    text_color = 'white'
    ax.set_xlabel('X Position', color=text_color)
    ax.set_ylabel('Y Position', color=text_color)
    ax.set_title('Bouncing, Rolling Ball Animation', color=text_color)
    ax.tick_params(colors=text_color)
    ball_patch = plt.Circle(ball.position.copy(), ball.radius, color='green')
    ax.add_patch(ball_patch)
    # Trail data
    trail_x = [ball.position[0]]
    trail_y = [ball.position[1]]
    speeds = [np.linalg.norm(ball.velocity)]
    segments = []
    cmap = 'coolwarm'
    trail_lc = LineCollection(segments, cmap=cmap, linewidth=2)
    ax.add_collection(trail_lc)

    def update(frame):
        ball.update(dt)
        ball_patch.center = ball.position.copy()
        trail_x.append(ball.position[0])
        trail_y.append(ball.position[1])
        speeds.append(np.linalg.norm(ball.velocity))
        # Create segments for the trail
        if len(trail_x) > 1:
            points = np.array([trail_x, trail_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            speed_arr = np.array(speeds)
            # Use actual speed values for color mapping
            trail_lc.set_segments(segments)
            trail_lc.set_array(speed_arr[:-1])
            trail_lc.set_cmap(cmap=cmap)
            trail_lc.set_norm(plt.Normalize(0., speed_arr.max() + 1e-8))
        # Ball color logic
        if np.allclose(ball.velocity, [0, 0], atol=0.01):
            ball_patch.set_color('red')
        else:
            ball_patch.set_color('green')
        return ball_patch, trail_lc

    ani = FuncAnimation(fig, update, frames=frames, interval=dt*1000, blit=True)
    plt.show()


def main():
    # Start ball with random initial velocity
    init_velocity = (random.uniform(-20, 20), random.uniform(-10, 10))
    ball = Ball(radius=0.2, position=(0, 3), velocity=init_velocity,
                restitution=0.75, rolling_friction=0.9, drag_coeff=0.1,
                min_x=-5, max_x=5, min_y=0, max_y=5)
    # Start animation
    animate_ball(ball, dt=0.02)

if __name__ == "__main__":
    main()
