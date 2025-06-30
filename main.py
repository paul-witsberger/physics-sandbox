from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from PIL import Image
import numpy as np
import random
import io

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
            if abs(self.velocity[1]) < 0.01:
                self.velocity[1] = 0
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
        # Clamp position to keep the ball fully inside the box
        self.position[0] = np.clip(self.position[0], self.min_x, self.max_x)
        self.position[1] = np.clip(self.position[1], self.min_y, self.max_y)

def balls_collide(ball1, ball2):
    # Returns True if balls overlap
    dist = np.linalg.norm(ball1.position - ball2.position)
    return dist <= (ball1.radius + ball2.radius)

def resolve_collision(ball1, ball2):
    # Elastic collision with restitution
    delta = ball1.position - ball2.position
    dist = np.linalg.norm(delta)
    if dist == 0:
        # Prevent division by zero
        delta = np.random.rand(2) - 0.5
        dist = np.linalg.norm(delta)
    n = delta / dist
    v_rel = np.dot(ball1.velocity - ball2.velocity, n)
    if v_rel > 0:
        return  # Balls are moving apart
    restitution = min(ball1.restitution, ball2.restitution)
    impulse = -(1 + restitution) * v_rel / 2  # Assume equal mass
    ball1.velocity += impulse * n
    ball2.velocity -= impulse * n
    # Separate balls so they don't stick
    overlap = (ball1.radius + ball2.radius) - dist
    correction = n * (overlap / 2)
    ball1.position += correction
    ball2.position -= correction


def animate_balls(balls, dt=0.05, frames=2000, show_trails=True, save_gif=True):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#181825')
    ax.set_facecolor("#474952")
    margin = balls[0].radius * 1.1  # Add a small margin to the axes
    min_x = min(ball.min_x for ball in balls) - margin
    max_x = max(ball.max_x for ball in balls) + margin
    min_y = min(ball.min_y for ball in balls) - margin
    max_y = max(ball.max_y for ball in balls) + margin
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    text_color = 'white'
    ax.set_xlabel('X Position', color=text_color)
    ax.set_ylabel('Y Position', color=text_color)
    ax.set_title('Bouncing, Rolling Balls Animation', color=text_color)
    ax.tick_params(colors=text_color)
    colors = ['green', 'orange', 'cyan', 'magenta', 'yellow', 'lime', 'aqua', 'violet', 'gold', 'pink']
    cmaps = ['coolwarm', 'plasma', 'viridis', 'magma', 'cividis', 'twilight', 'hsv', 'inferno', 'spring', 'summer']
    ball_patches = []
    trail_xs, trail_ys, speeds, trail_lcs = [], [], [], []
    for i, ball in enumerate(balls):
        color = colors[i % len(colors)]
        cmap = cmaps[i % len(cmaps)]
        patch = plt.Circle(ball.position.copy(), ball.radius, color=color)
        ax.add_patch(patch)
        ball_patches.append(patch)
        trail_xs.append([ball.position[0]])
        trail_ys.append([ball.position[1]])
        speeds.append([np.linalg.norm(ball.velocity)])
        if show_trails:
            trail_lc = LineCollection([], cmap=cmap, linewidth=2)
            ax.add_collection(trail_lc)
        else:
            trail_lc = None
        trail_lcs.append(trail_lc)

    if save_gif:
        frames_list = []
        stopped = False
        for frame in range(frames):
            for ball in balls:
                ball.update(dt)
            n = len(balls)
            for i in range(n):
                for j in range(i+1, n):
                    if balls_collide(balls[i], balls[j]):
                        resolve_collision(balls[i], balls[j])
            for i, ball in enumerate(balls):
                ball_patches[i].center = ball.position.copy()
                last_x, last_y = trail_xs[i][-1], trail_ys[i][-1]
                if np.linalg.norm([ball.position[0] - last_x, ball.position[1] - last_y]) > 1e-4:
                    trail_xs[i].append(ball.position[0])
                    trail_ys[i].append(ball.position[1])
                    speeds[i].append(np.linalg.norm(ball.velocity))
                if show_trails and len(trail_xs[i]) > 1:
                    points = np.array([trail_xs[i], trail_ys[i]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    speed_arr = np.array(speeds[i])
                    trail_lcs[i].set_segments(segments)
                    trail_lcs[i].set_array(speed_arr[:-1])
                    trail_lcs[i].set_norm(plt.Normalize(0., speed_arr.max() + 1e-8))
                # Ball color logic
                if np.allclose(ball.velocity, [0, 0], atol=0.01):
                    ball_patches[i].set_color('red')
                else:
                    ball_patches[i].set_color(colors[i % len(colors)])
            plt.draw()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            frames_list.append(img.convert('P', palette=Image.ADAPTIVE))
            buf.close()
            if all(np.allclose(ball.velocity, [0, 0], atol=0.01) for ball in balls):
                stopped = True
                break
        # Save GIF with a pause at the end
        if frames_list:
            pause_frames = int(0.5 / dt)
            for _ in range(pause_frames):
                frames_list.append(frames_list[-1])
            frames_list[0].save('bouncing_balls.gif', save_all=True, append_images=frames_list[1:], duration=int(dt*1000), loop=0)
        plt.show()
    else:
        def update(frame):
            for ball in balls:
                ball.update(dt)
            n = len(balls)
            for i in range(n):
                for j in range(i+1, n):
                    if balls_collide(balls[i], balls[j]):
                        resolve_collision(balls[i], balls[j])
            for i, ball in enumerate(balls):
                ball_patches[i].center = ball.position.copy()
                last_x, last_y = trail_xs[i][-1], trail_ys[i][-1]
                if np.linalg.norm([ball.position[0] - last_x, ball.position[1] - last_y]) > 1e-4:
                    trail_xs[i].append(ball.position[0])
                    trail_ys[i].append(ball.position[1])
                    speeds[i].append(np.linalg.norm(ball.velocity))
                if show_trails and len(trail_xs[i]) > 1:
                    points = np.array([trail_xs[i], trail_ys[i]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    speed_arr = np.array(speeds[i])
                    trail_lcs[i].set_segments(segments)
                    trail_lcs[i].set_array(speed_arr[:-1])
                    trail_lcs[i].set_norm(plt.Normalize(0., speed_arr.max() + 1e-8))
                # Ball color logic
                if np.allclose(ball.velocity, [0, 0], atol=0.01):
                    ball_patches[i].set_color('red')
                else:
                    ball_patches[i].set_color(colors[i % len(colors)])
            return ball_patches + [lc for lc in trail_lcs if lc is not None and show_trails]
        from matplotlib.animation import FuncAnimation
        ani = FuncAnimation(fig, update, interval=dt*1000, blit=False)
        plt.show()


def main():
    num_balls = 20  # User can change this
    show_trails = False  # Set to False to hide trails
    save_gif = False  # Set to False to only show animation, not save GIF
    balls = []
    for i in range(num_balls):
        radius = 0.1
        # Random position, avoid spawning balls overlapping
        while True:
            pos = (random.uniform(-4, 4), random.uniform(1, 4.5))
            if all(np.linalg.norm(np.array(pos) - b.position) > 2*radius for b in balls):
                break
        velocity = (random.uniform(-10, 10), random.uniform(-10, 10))
        ball = Ball(radius=radius, position=pos, velocity=velocity,
                    restitution=0.8, rolling_friction=0.85, drag_coeff=0.1,
                    min_x=-5, max_x=5, min_y=0, max_y=5)
        balls.append(ball)
    animate_balls(balls, dt=0.02, show_trails=show_trails, save_gif=save_gif)

if __name__ == "__main__":
    main()
