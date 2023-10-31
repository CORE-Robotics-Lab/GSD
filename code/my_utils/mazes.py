# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import matplotlib.pyplot as plt
import numpy as np


class CircleMaze:
    # Deprecated (as of now, especially regarding render())
    def __init__(self):
        raise NotImplementedError()

        self.ring_r = 0.15
        self.stop_t = 0.05
        self.s_angle = 30

        self.mean_s0 = (
            float(np.cos(np.pi * self.s_angle / 180)),
            float(np.sin(np.pi * self.s_angle / 180))
        )
        self.mean_g = (
            float(np.cos(np.pi * (360 - self.s_angle) / 180)),
            float(np.sin(np.pi * (360 - self.s_angle) / 180))
        )

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 4))
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 4))
        rads = np.linspace(self.stop_t * 2 * np.pi, (1 - self.stop_t) * 2 * np.pi)
        xs_i = (1 - self.ring_r) * np.cos(rads)
        ys_i = (1 - self.ring_r) * np.sin(rads)
        xs_o = (1 + self.ring_r) * np.cos(rads)
        ys_o = (1 + self.ring_r) * np.sin(rads)
        ax.plot(xs_i, ys_i, 'k', linewidth=3)
        ax.plot(xs_o, ys_o, 'k', linewidth=3)
        ax.plot([xs_i[0], xs_o[0]], [ys_i[0], ys_o[0]], 'k', linewidth=3)
        ax.plot([xs_i[-1], xs_o[-1]], [ys_i[-1], ys_o[-1]], 'k', linewidth=3)
        lim = 1.1 + self.ring_r
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])

    def sample_start(self):
        STD = 0.1
        return self.move(self.mean_s0, (STD * np.random.randn(), STD * np.random.randn()))

    def sample_goal(self):
        STD = 0.1
        return self.move(self.mean_g, (STD * np.random.randn(), STD * np.random.randn()))

    @staticmethod
    def xy_to_rt(xy):
        x = xy[0]
        y = xy[1]
        r = np.sqrt(x ** 2 + y ** 2)
        t = np.arctan2(y, x) % (2 * np.pi)
        return r, t

    def move(self, coords, action):
        xp, yp = coords
        rp, tp = self.xy_to_rt(coords)

        xy = (coords[0] + action[0], coords[1] + action[1])

        r, t = self.xy_to_rt(xy)
        t = np.clip(t % (2 * np.pi), (0.001 + self.stop_t) * (2 * np.pi), (1 - (0.001 + self.stop_t)) * (2 * np.pi))
        x = np.cos(t) * r
        y = np.sin(t) * r

        if coords is not None:

            if xp > 0:
                if (y < 0) and (yp > 0):
                    t = self.stop_t * 2 * np.pi
                elif (y > 0) and (yp < 0):
                    t = (1 - self.stop_t) * 2 * np.pi
            x = np.cos(t) * r
            y = np.sin(t) * r

        n = 8
        xyi = np.array([xp, yp]).astype(np.float32)
        dxy = (np.array([x, y]).astype(np.float32) - xyi) / n
        new_r = float(rp)
        new_t = float(tp)

        count = 0

        def r_ok(r_):
            return (1 - self.ring_r) <= r_ <= (1 + self.ring_r)

        def t_ok(t_):
            return (self.stop_t * (2 * np.pi)) <= (t_ % (2 * np.pi)) <= ((1 - self.stop_t) * (2 * np.pi))

        while r_ok(new_r) and t_ok(new_t) and count < n:
            xyi += dxy
            new_r, new_t = self.xy_to_rt(xyi)
            count += 1

        r = np.clip(new_r, 1 - self.ring_r + 0.01, 1 + self.ring_r - 0.01)
        t = np.clip(new_t % (2 * np.pi), (0.001 + self.stop_t) * (2 * np.pi), (1 - (0.001 + self.stop_t)) * (2 * np.pi))
        x = np.cos(t) * r
        y = np.sin(t) * r

        return float(x), float(y)


class Maze:
    def __init__(self, *segment_dicts, goal_squares=None, start_squares=None,
                 min_wall_coord=None, walls_to_add=(), walls_to_remove=(), start_random_range=0.5):
        self._segments = {'origin': {'loc': (0.0, 0.0), 'connect': set()}}
        self._locs = set()
        self._locs.add(self._segments['origin']['loc'])
        self._walls = set()
        for direction in ['up', 'down', 'left', 'right']:
            self._walls.add(self._wall_line(self._segments['origin']['loc'], direction))
        self._last_segment = 'origin'
        self.goal_squares = None

        # These allow to implement more complex mazes
        self.min_wall_coord = min_wall_coord
        self.walls_to_add = walls_to_add
        self.walls_to_remove = walls_to_remove

        self.start_random_range = start_random_range

        if goal_squares is None:
            self._goal_squares = None
        elif isinstance(goal_squares, str):
            self._goal_squares = [goal_squares.lower()]
        elif isinstance(goal_squares, (tuple, list)):
            self._goal_squares = [gs.lower() for gs in goal_squares]
        else:
            raise TypeError

        if start_squares is None:
            self.start_squares = ['origin']
        elif isinstance(start_squares, str):
            self.start_squares = [start_squares.lower()]
        elif isinstance(start_squares, (tuple, list)):
            self.start_squares = [ss.lower() for ss in start_squares]
        else:
            raise TypeError

        for segment_dict in segment_dicts:
            self._add_segment(**segment_dict)
        self._finalize()

        self.fig, self.ax = None, None

        wall_xs = [wall[0] for walls in self._walls for wall in walls]
        wall_ys = [wall[1] for walls in self._walls for wall in walls]
        self.min_x = min(wall_xs)
        self.max_x = max(wall_xs)
        self.min_y = min(wall_ys)
        self.max_y = max(wall_ys)

    @staticmethod
    def _wall_line(coord, direction):
        x, y = coord
        if direction == 'up':
            w = [(x - 0.5, x + 0.5), (y + 0.5, y + 0.5)]
        elif direction == 'right':
            w = [(x + 0.5, x + 0.5), (y + 0.5, y - 0.5)]
        elif direction == 'down':
            w = [(x - 0.5, x + 0.5), (y - 0.5, y - 0.5)]
        elif direction == 'left':
            w = [(x - 0.5, x - 0.5), (y - 0.5, y + 0.5)]
        else:
            raise ValueError
        w = tuple([tuple(sorted(line)) for line in w])
        return w

    def _add_segment(self, name, anchor, direction, connect=None, times=1):
        name = str(name).lower()
        original_name = str(name).lower()
        if times > 1:
            assert connect is None
            last_name = str(anchor).lower()
            for time in range(times):
                this_name = original_name + str(time)
                self._add_segment(name=this_name.lower(), anchor=last_name, direction=direction)
                last_name = str(this_name)
            return

        anchor = str(anchor).lower()
        assert anchor in self._segments

        direction = str(direction).lower()

        final_connect = set()

        if connect is not None:
            if isinstance(connect, str):
                connect = str(connect).lower()
                assert connect in ['up', 'down', 'left', 'right']
                final_connect.add(connect)
            elif isinstance(connect, (tuple, list)):
                for connect_direction in connect:
                    connect_direction = str(connect_direction).lower()
                    assert connect_direction in ['up', 'down', 'left', 'right']
                    final_connect.add(connect_direction)

        sx, sy = self._segments[anchor]['loc']
        dx, dy = 0.0, 0.0
        if direction == 'left':
            dx -= 1
            final_connect.add('right')
        elif direction == 'right':
            dx += 1
            final_connect.add('left')
        elif direction == 'up':
            dy += 1
            final_connect.add('down')
        elif direction == 'down':
            dy -= 1
            final_connect.add('up')
        else:
            raise ValueError

        new_loc = (sx + dx, sy + dy)
        assert new_loc not in self._locs

        self._segments[name] = {'loc': new_loc, 'connect': final_connect}
        for direction in ['up', 'down', 'left', 'right']:
            self._walls.add(self._wall_line(new_loc, direction))
        self._locs.add(new_loc)

        self._last_segment = name

    def _finalize(self):
        bottom_wall_coord = min([min(w[0]) for w in self._walls]) + 0.5
        left_wall_coord = min([min(w[1]) for w in self._walls]) + 0.5

        def _rm_wall(wall):
            coords = wall[0] + wall[1]
            # Check if this is the bottom wall
            if wall[0][0] < bottom_wall_coord and wall[0][1] < bottom_wall_coord:
                return False
            # Check if this is the left wall
            if wall[1][0] < left_wall_coord and wall[1][1] < left_wall_coord:
                return False
            # Remove walls in the bottom-left corner
            return all(c < self.min_wall_coord for c in coords)

        if self.min_wall_coord is not None:
            self._walls = set([w for w in self._walls if not _rm_wall(w)])

        for wall in self.walls_to_remove:
            if wall in self._walls:
                self._walls.remove(wall)

        for segment in self._segments.values():
            for c_dir in list(segment['connect']):
                wall = self._wall_line(segment['loc'], c_dir)
                if wall in self._walls:
                    self._walls.remove(wall)

        for wall in self.walls_to_add:
            self._walls.add(wall)

        if self._goal_squares is None:
            self.goal_squares = [self._last_segment]
        else:
            self.goal_squares = []
            for gs in self._goal_squares:
                assert gs in self._segments
                self.goal_squares.append(gs)

    def plot_maze(self, ax):
        for x, y in self._walls:
            ax.plot(x, y, 'k-')

    def plot(self, trajectory):
        """Plot trajectory onto the screen."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.fig.canvas.draw()
        self.ax.cla()
        # self.plot_maze(self.ax)
        trajectory = np.array(trajectory)
        self.ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=0.7)
        self.ax.axis('scaled')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.fig.draw(self.fig.canvas.renderer)
        plt.pause(0.0001)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        """Plot trajectories onto given ax."""
        # self.plot_maze(ax)
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)
        if plot_axis is not None:
            ax.axis(plot_axis)
        else:
            ax.axis('scaled')

    def sample(self):
        segment_keys = list(self._segments.keys())
        square_id = segment_keys[np.random.randint(low=0, high=len(segment_keys))]
        square_loc = self._segments[square_id]['loc']
        shift = np.random.uniform(low=-0.5, high=0.5, size=(2,))
        loc = square_loc + shift
        return loc[0], loc[1]

    def sample_start(self):
        min_wall_dist = 0.05

        s_square = self.start_squares[np.random.randint(low=0, high=len(self.start_squares))]
        s_square_loc = self._segments[s_square]['loc']

        while True:
            shift = np.random.uniform(low=-self.start_random_range, high=self.start_random_range, size=(2,))
            loc = s_square_loc + shift
            dist_checker = np.array([min_wall_dist, min_wall_dist]) * np.sign(shift)
            stopped_loc = self.move(loc, dist_checker)
            if float(np.sum(np.abs((loc + dist_checker) - stopped_loc))) == 0.0:
                break

        return loc[0], loc[1]

    def sample_goal(self, min_wall_dist=None):
        if min_wall_dist is None:
            min_wall_dist = 0.1
        else:
            min_wall_dist = min(0.4, max(0.01, min_wall_dist))

        g_square = self.goal_squares[np.random.randint(low=0, high=len(self.goal_squares))]
        g_square_loc = self._segments[g_square]['loc']
        while True:
            shift = np.random.uniform(low=-0.5, high=0.5, size=(2,))
            loc = g_square_loc + shift
            dist_checker = np.array([min_wall_dist, min_wall_dist]) * np.sign(shift)
            stopped_loc = self.move(loc, dist_checker)
            if float(np.sum(np.abs((loc + dist_checker) - stopped_loc))) == 0.0:
                break
        return loc[0], loc[1]

    def move(self, coord_start, coord_delta, depth=None):
        if depth is None:
            depth = 0
        cx, cy = coord_start
        loc_x0 = np.round(cx)
        loc_y0 = np.round(cy)
        # assert (float(loc_x0), float(loc_y0)) in self._locs
        dx, dy = coord_delta
        loc_x1 = np.round(cx + dx)
        loc_y1 = np.round(cy + dy)
        d_loc_x = int(np.abs(loc_x1 - loc_x0))
        d_loc_y = int(np.abs(loc_y1 - loc_y0))
        xs_crossed = [loc_x0 + (np.sign(dx) * (i + 0.5)) for i in range(d_loc_x)]
        ys_crossed = [loc_y0 + (np.sign(dy) * (i + 0.5)) for i in range(d_loc_y)]

        rds = []

        for x in xs_crossed:
            r = (x - cx) / dx
            loc_x = np.round(cx + (0.999 * r * dx))
            loc_y = np.round(cy + (0.999 * r * dy))
            direction = 'right' if dx > 0 else 'left'
            crossed_line = self._wall_line((loc_x, loc_y), direction)
            if crossed_line in self._walls:
                rds.append([r, direction])

        for y in ys_crossed:
            r = (y - cy) / dy
            loc_x = np.round(cx + (0.999 * r * dx))
            loc_y = np.round(cy + (0.999 * r * dy))
            direction = 'up' if dy > 0 else 'down'
            crossed_line = self._wall_line((loc_x, loc_y), direction)
            if crossed_line in self._walls:
                rds.append([r, direction])

        # The wall will only stop the agent in the direction perpendicular to the wall
        if rds:
            rds = sorted(rds)
            r, direction = rds[0]
            if depth < 3:
                new_dx = r * dx
                new_dy = r * dy
                repulsion = float(np.abs(np.random.rand() * 0.01))
                if direction in ['right', 'left']:
                    new_dx -= np.sign(dx) * repulsion
                    partial_coords = cx + new_dx, cy + new_dy
                    remaining_delta = (0.0, (1 - r) * dy)
                else:
                    new_dy -= np.sign(dy) * repulsion
                    partial_coords = cx + new_dx, cy + new_dy
                    remaining_delta = ((1 - r) * dx, 0.0)
                return self.move(partial_coords, remaining_delta, depth + 1)
        else:
            r = 1.0

        dx *= r
        dy *= r
        return cx + dx, cy + dy


def make_crazy_maze(size, seed=None):
    np.random.seed(seed)

    deltas = [
        [(-1, 0), 'right'],
        [(1, 0), 'left'],
        [(0, -1), 'up'],
        [(0, 1), 'down'],
    ]

    empty_locs = []
    for x in range(size):
        for y in range(size):
            empty_locs.append((x, y))

    locs = [empty_locs.pop(0)]
    dirs = [None]
    anchors = [None]

    while len(empty_locs) > 0:
        still_empty = []
        np.random.shuffle(empty_locs)
        for empty_x, empty_y in empty_locs:
            found_anchor = False
            np.random.shuffle(deltas)
            for (dx, dy), direction in deltas:
                c = (empty_x + dx, empty_y + dy)
                if c in locs:
                    found_anchor = True
                    locs.append((empty_x, empty_y))
                    dirs.append(direction)
                    anchors.append(c)
                    break
            if not found_anchor:
                still_empty.append((empty_x, empty_y))
        empty_locs = still_empty[:]

    locs = [str(x) + ',' + str(y) for x, y in locs[1:]]
    dirs = dirs[1:]
    anchors = [str(x) + ',' + str(y) for x, y in anchors[1:]]
    anchors = ['origin' if a == '0,0' else a for a in anchors]

    segments = []
    for loc, d, anchor in zip(locs, dirs, anchors):
        segments.append(dict(name=loc, anchor=anchor, direction=d))

    np.random.seed()
    return Maze(*segments, goal_squares='{s},{s}'.format(s=size - 1))


def make_experiment_maze(h, half_w, sz0):
    if h < 2:
        h = 2
    if half_w < 3:
        half_w = 3
    w = 1 + (2 * half_w)
    # Create the starting row
    segments = [{'anchor': 'origin', 'direction': 'right', 'name': '0,1'}]
    for w_ in range(1, w - 1):
        segments.append({'anchor': '0,{}'.format(w_), 'direction': 'right', 'name': '0,{}'.format(w_ + 1)})

    # Add each row to create H
    for h_ in range(1, h):
        segments.append({'anchor': '{},{}'.format(h_ - 1, w - 1), 'direction': 'up', 'name': '{},{}'.format(h_, w - 1)})

        c = None if h_ == sz0 else 'down'
        for w_ in range(w - 2, -1, -1):
            segments.append(
                {'anchor': '{},{}'.format(h_, w_ + 1), 'direction': 'left', 'connect': c,
                 'name': '{},{}'.format(h_, w_)}
            )

    return Maze(*segments, goal_squares=['{},{}'.format(h - 1, half_w + d) for d in [0]])


def make_hallway_maze(corridor_length):
    corridor_length = int(corridor_length)
    assert corridor_length >= 1

    segments = []
    last = 'origin'
    for x in range(1, corridor_length + 1):
        next_name = '0,{}'.format(x)
        segments.append({'anchor': last, 'direction': 'right', 'name': next_name})
        last = str(next_name)

    return Maze(*segments, goal_squares=last)


def make_u_maze(corridor_length):
    corridor_length = int(corridor_length)
    assert corridor_length >= 1

    segments = []
    last = 'origin'
    for x in range(1, corridor_length + 1):
        next_name = '0,{}'.format(x)
        segments.append({'anchor': last, 'direction': 'right', 'name': next_name})
        last = str(next_name)

    assert last == '0,{}'.format(corridor_length)

    up_size = 2

    for x in range(1, up_size + 1):
        next_name = '{},{}'.format(x, corridor_length)
        segments.append({'anchor': last, 'direction': 'up', 'name': next_name})
        last = str(next_name)

    assert last == '{},{}'.format(up_size, corridor_length)

    for x in range(1, corridor_length + 1):
        next_name = '{},{}'.format(up_size, corridor_length - x)
        segments.append({'anchor': last, 'direction': 'left', 'name': next_name})
        last = str(next_name)

    assert last == '{},0'.format(up_size)

    return Maze(*segments, goal_squares=last)


mazes_dict = dict()

segments_a = [
    dict(name='A', anchor='origin', direction='down', times=4),
    dict(name='B', anchor='A3', direction='right', times=4),
    dict(name='C', anchor='B3', direction='up', times=4),
    dict(name='D', anchor='A1', direction='right', times=2),
    dict(name='E', anchor='D1', direction='up', times=2),
]
mazes_dict['square_a'] = {'maze': Maze(*segments_a, goal_squares=['c2', 'c3']), 'action_range': 0.95}

segments_b = [
    dict(name='A', anchor='origin', direction='down', times=4),
    dict(name='B', anchor='A3', direction='right', times=4),
    dict(name='C', anchor='B3', direction='up', times=4),
    dict(name='D', anchor='B1', direction='up', times=4),
]
mazes_dict['square_b'] = {'maze': Maze(*segments_b, goal_squares=['c2', 'c3']), 'action_range': 0.95}

segments_c = [
    dict(name='A', anchor='origin', direction='down', times=4),
    dict(name='B', anchor='A3', direction='right', times=2),
    dict(name='C', anchor='B1', direction='up', times=4),
    dict(name='D', anchor='C3', direction='right', times=2),
    dict(name='E', anchor='D1', direction='down', times=4)
]
mazes_dict['square_c'] = {'maze': Maze(*segments_c, goal_squares=['e2', 'e3']), 'action_range': 0.95}

segments_d = [
    dict(name='TL', anchor='origin', direction='left', times=3),
    dict(name='TLD', anchor='TL2', direction='down', times=3),
    dict(name='TLR', anchor='TLD2', direction='right', times=2),
    dict(name='TLU', anchor='TLR1', direction='up'),
    dict(name='TR', anchor='origin', direction='right', times=3),
    dict(name='TRD', anchor='TR2', direction='down', times=3),
    dict(name='TRL', anchor='TRD2', direction='left', times=2),
    dict(name='TRU', anchor='TRL1', direction='up'),
    dict(name='TD', anchor='origin', direction='down', times=3),
]
mazes_dict['square_d'] = {'maze': Maze(*segments_d, goal_squares=['tlu', 'tlr1', 'tru', 'trl1']), 'action_range': 0.95}

segments_crazy = [
    {'anchor': 'origin', 'direction': 'right', 'name': '1,0'},
    {'anchor': 'origin', 'direction': 'up', 'name': '0,1'},
    {'anchor': '1,0', 'direction': 'right', 'name': '2,0'},
    {'anchor': '0,1', 'direction': 'up', 'name': '0,2'},
    {'anchor': '0,2', 'direction': 'right', 'name': '1,2'},
    {'anchor': '2,0', 'direction': 'up', 'name': '2,1'},
    {'anchor': '1,2', 'direction': 'right', 'name': '2,2'},
    {'anchor': '0,2', 'direction': 'up', 'name': '0,3'},
    {'anchor': '2,1', 'direction': 'right', 'name': '3,1'},
    {'anchor': '1,2', 'direction': 'down', 'name': '1,1'},
    {'anchor': '3,1', 'direction': 'down', 'name': '3,0'},
    {'anchor': '1,2', 'direction': 'up', 'name': '1,3'},
    {'anchor': '3,1', 'direction': 'right', 'name': '4,1'},
    {'anchor': '1,3', 'direction': 'up', 'name': '1,4'},
    {'anchor': '4,1', 'direction': 'right', 'name': '5,1'},
    {'anchor': '4,1', 'direction': 'up', 'name': '4,2'},
    {'anchor': '5,1', 'direction': 'down', 'name': '5,0'},
    {'anchor': '3,0', 'direction': 'right', 'name': '4,0'},
    {'anchor': '1,4', 'direction': 'right', 'name': '2,4'},
    {'anchor': '4,2', 'direction': 'right', 'name': '5,2'},
    {'anchor': '2,4', 'direction': 'right', 'name': '3,4'},
    {'anchor': '3,4', 'direction': 'up', 'name': '3,5'},
    {'anchor': '1,4', 'direction': 'left', 'name': '0,4'},
    {'anchor': '1,4', 'direction': 'up', 'name': '1,5'},
    {'anchor': '2,2', 'direction': 'up', 'name': '2,3'},
    {'anchor': '3,1', 'direction': 'up', 'name': '3,2'},
    {'anchor': '5,0', 'direction': 'right', 'name': '6,0'},
    {'anchor': '3,2', 'direction': 'up', 'name': '3,3'},
    {'anchor': '4,2', 'direction': 'up', 'name': '4,3'},
    {'anchor': '6,0', 'direction': 'up', 'name': '6,1'},
    {'anchor': '6,0', 'direction': 'right', 'name': '7,0'},
    {'anchor': '6,1', 'direction': 'right', 'name': '7,1'},
    {'anchor': '3,4', 'direction': 'right', 'name': '4,4'},
    {'anchor': '1,5', 'direction': 'right', 'name': '2,5'},
    {'anchor': '7,1', 'direction': 'up', 'name': '7,2'},
    {'anchor': '1,5', 'direction': 'up', 'name': '1,6'},
    {'anchor': '4,4', 'direction': 'right', 'name': '5,4'},
    {'anchor': '5,4', 'direction': 'down', 'name': '5,3'},
    {'anchor': '0,4', 'direction': 'up', 'name': '0,5'},
    {'anchor': '7,2', 'direction': 'left', 'name': '6,2'},
    {'anchor': '1,6', 'direction': 'left', 'name': '0,6'},
    {'anchor': '7,0', 'direction': 'right', 'name': '8,0'},
    {'anchor': '7,2', 'direction': 'right', 'name': '8,2'},
    {'anchor': '2,5', 'direction': 'up', 'name': '2,6'},
    {'anchor': '8,0', 'direction': 'up', 'name': '8,1'},
    {'anchor': '3,5', 'direction': 'up', 'name': '3,6'},
    {'anchor': '6,2', 'direction': 'up', 'name': '6,3'},
    {'anchor': '6,3', 'direction': 'right', 'name': '7,3'},
    {'anchor': '3,5', 'direction': 'right', 'name': '4,5'},
    {'anchor': '7,3', 'direction': 'up', 'name': '7,4'},
    {'anchor': '6,3', 'direction': 'up', 'name': '6,4'},
    {'anchor': '6,4', 'direction': 'up', 'name': '6,5'},
    {'anchor': '8,1', 'direction': 'right', 'name': '9,1'},
    {'anchor': '8,2', 'direction': 'right', 'name': '9,2'},
    {'anchor': '2,6', 'direction': 'up', 'name': '2,7'},
    {'anchor': '8,2', 'direction': 'up', 'name': '8,3'},
    {'anchor': '6,5', 'direction': 'left', 'name': '5,5'},
    {'anchor': '5,5', 'direction': 'up', 'name': '5,6'},
    {'anchor': '7,4', 'direction': 'right', 'name': '8,4'},
    {'anchor': '8,4', 'direction': 'right', 'name': '9,4'},
    {'anchor': '0,6', 'direction': 'up', 'name': '0,7'},
    {'anchor': '2,7', 'direction': 'up', 'name': '2,8'},
    {'anchor': '7,4', 'direction': 'up', 'name': '7,5'},
    {'anchor': '9,4', 'direction': 'down', 'name': '9,3'},
    {'anchor': '9,4', 'direction': 'up', 'name': '9,5'},
    {'anchor': '2,7', 'direction': 'left', 'name': '1,7'},
    {'anchor': '4,5', 'direction': 'up', 'name': '4,6'},
    {'anchor': '9,1', 'direction': 'down', 'name': '9,0'},
    {'anchor': '6,5', 'direction': 'up', 'name': '6,6'},
    {'anchor': '3,6', 'direction': 'up', 'name': '3,7'},
    {'anchor': '1,7', 'direction': 'up', 'name': '1,8'},
    {'anchor': '3,7', 'direction': 'right', 'name': '4,7'},
    {'anchor': '2,8', 'direction': 'up', 'name': '2,9'},
    {'anchor': '2,9', 'direction': 'left', 'name': '1,9'},
    {'anchor': '7,5', 'direction': 'up', 'name': '7,6'},
    {'anchor': '1,8', 'direction': 'left', 'name': '0,8'},
    {'anchor': '6,6', 'direction': 'up', 'name': '6,7'},
    {'anchor': '0,8', 'direction': 'up', 'name': '0,9'},
    {'anchor': '7,5', 'direction': 'right', 'name': '8,5'},
    {'anchor': '6,7', 'direction': 'left', 'name': '5,7'},
    {'anchor': '2,9', 'direction': 'right', 'name': '3,9'},
    {'anchor': '3,9', 'direction': 'right', 'name': '4,9'},
    {'anchor': '7,6', 'direction': 'right', 'name': '8,6'},
    {'anchor': '3,7', 'direction': 'up', 'name': '3,8'},
    {'anchor': '9,5', 'direction': 'up', 'name': '9,6'},
    {'anchor': '7,6', 'direction': 'up', 'name': '7,7'},
    {'anchor': '5,7', 'direction': 'up', 'name': '5,8'},
    {'anchor': '3,8', 'direction': 'right', 'name': '4,8'},
    {'anchor': '8,6', 'direction': 'up', 'name': '8,7'},
    {'anchor': '5,8', 'direction': 'right', 'name': '6,8'},
    {'anchor': '7,7', 'direction': 'up', 'name': '7,8'},
    {'anchor': '4,9', 'direction': 'right', 'name': '5,9'},
    {'anchor': '8,7', 'direction': 'right', 'name': '9,7'},
    {'anchor': '7,8', 'direction': 'right', 'name': '8,8'},
    {'anchor': '8,8', 'direction': 'up', 'name': '8,9'},
    {'anchor': '5,9', 'direction': 'right', 'name': '6,9'},
    {'anchor': '6,9', 'direction': 'right', 'name': '7,9'},
    {'anchor': '8,9', 'direction': 'right', 'name': '9,9'},
    {'anchor': '9,9', 'direction': 'down', 'name': '9,8'}
]
mazes_dict['square_large'] = {'maze': Maze(*segments_crazy, goal_squares='9,9'), 'action_range': 0.95}

segments_tree = [
    dict(name='A', anchor='origin', direction='down', times=2),
    dict(name='BR', anchor='A1', direction='right', times=4),
    dict(name='BL', anchor='A1', direction='left', times=4),
    dict(name='CR', anchor='BR3', direction='down', times=2),
    dict(name='CL', anchor='BL3', direction='down', times=2),
    dict(name='DLL', anchor='CL1', direction='left', times=2),
    dict(name='DLR', anchor='CL1', direction='right', times=2),
    dict(name='DRL', anchor='CR1', direction='left', times=2),
    dict(name='DRR', anchor='CR1', direction='right', times=2),
    dict(name='ELL', anchor='DLL1', direction='down', times=2),
    dict(name='ELR', anchor='DLR1', direction='down', times=2),
    dict(name='ERL', anchor='DRL1', direction='down', times=2),
    dict(name='ERR', anchor='DRR1', direction='down', times=2),
]
mazes_dict['square_tree'] = {'maze': Maze(*segments_tree, goal_squares=['ELL1', 'ERR1']), 'action_range': 0.95}

segments_corridor = [
    dict(name='A', anchor='origin', direction='left', times=5),
    dict(name='B', anchor='origin', direction='right', times=5)
]
mazes_dict['square_corridor'] = {'maze': Maze(*segments_corridor, goal_squares=['b4']), 'action_range': 0.95}
mazes_dict['square_corridor2'] = {'maze': Maze(*segments_corridor, goal_squares=['b4'], start_squares=['a4']),
                                  'action_range': 0.95}

_walls_to_remove = [
    ((4.5, 4.5), (7.5, 8.5)),
    ((-0.5, 0.5), (5.5, 5.5)),
    ((2.5, 2.5), (4.5, 5.5)),
    ((3.5, 4.5), (3.5, 3.5)),
    ((4.5, 4.5), (2.5, 3.5)),
    ((4.5, 5.5), (2.5, 2.5)),
    ((3.5, 4.5), (0.5, 0.5)),
    ((4.5, 5.5), (4.5, 4.5)),
    ((5.5, 5.5), (0.5, 1.5)),
    ((8.5, 8.5), (-0.5, 0.5)),
    ((6.5, 7.5), (2.5, 2.5)),
    ((7.5, 7.5), (6.5, 7.5)),
    ((7.5, 8.5), (7.5, 7.5)),
    ((8.5, 8.5), (7.5, 8.5)),
    ((7.5, 7.5), (2.5, 3.5)),
    ((8.5, 9.5), (7.5, 7.5)),
    ((7.5, 8.5), (4.5, 4.5)),
    ((8.5, 8.5), (4.5, 5.5)),
    ((5.5, 6.5), (7.5, 7.5)),
    ((3.5, 4.5), (7.5, 7.5)),
    ((4.5, 4.5), (6.5, 7.5)),
    ((4.5, 4.5), (5.5, 6.5)),
    ((3.5, 3.5), (5.5, 6.5)),
    ((5.5, 5.5), (5.5, 6.5)),
    ((3.5, 4.5), (6.5, 6.5)),
    ((4.5, 5.5), (6.5, 6.5)),
    ((1.5, 1.5), (7.5, 8.5)),
    ((2.5, 2.5), (5.5, 6.5)),
    ((0.5, 0.5), (4.5, 5.5)),
    ((1.5, 1.5), (5.5, 6.5)),
    ((4.5, 4.5), (4.5, 5.5)),
    ((5.5, 5.5), (1.5, 2.5)),
    ((5.5, 5.5), (2.5, 3.5)),
    ((5.5, 5.5), (3.5, 4.5)),
    ((6.5, 7.5), (8.5, 8.5)),
    ((7.5, 7.5), (8.5, 9.5)),
    ((0.5, 0.5), (8.5, 9.5)),
    ((0.5, 1.5), (8.5, 8.5)),
    ((-0.5, 0.5), (7.5, 7.5)),
    ((0.5, 1.5), (6.5, 6.5)),
    ((0.5, 0.5), (6.5, 7.5)),
    ((2.5, 2.5), (6.5, 7.5)),
    ((2.5, 2.5), (7.5, 8.5)),
    ((2.5, 3.5), (8.5, 8.5)),
    ((3.5, 4.5), (8.5, 8.5)),
    ((4.5, 5.5), (8.5, 8.5)),
    ((5.5, 6.5), (8.5, 8.5)),
    ((7.5, 8.5), (5.5, 5.5)),
    ((8.5, 9.5), (6.5, 6.5)),
    ((8.5, 8.5), (5.5, 6.5)),
    ((7.5, 8.5), (3.5, 3.5)),
    ((8.5, 9.5), (2.5, 2.5)),
    ((8.5, 8.5), (2.5, 3.5)),
]
_walls_to_add = [
    ((-0.5, 0.5), (4.5, 4.5)),
    ((0.5, 1.5), (4.5, 4.5)),
    ((2.5, 3.5), (4.5, 4.5)),
    ((4.5, 4.5), (3.5, 4.5)),
    ((4.5, 4.5), (2.5, 3.5)),
    ((4.5, 4.5), (1.5, 2.5)),
    ((6.5, 6.5), (8.5, 9.5)),
]
mazes_dict['square_bottleneck'] = {'maze': Maze(*segments_crazy, goal_squares='9,9', min_wall_coord=4,
                                                walls_to_remove=_walls_to_remove, walls_to_add=_walls_to_add),
                                   'action_range': 0.95}
mazes_dict['square'] = {'maze': Maze(*segments_crazy, start_squares='4,4', goal_squares='9,9', min_wall_coord=9,
                                     walls_to_remove=_walls_to_remove + [((8.5, 9.5), (1.5, 1.5))]),
                        'action_range': 0.95}


# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT
from collections import defaultdict

import gym
import numpy as np

class DummyGoal:
    pass


class MazeEnv(gym.Env):
    def __init__(self, n, maze_type='square', use_antigoal=False, ddiff=True, ignore_reset_start=False,
                 done_on_success=True, action_range_override=None, start_random_range_override=None,
                 obs_include_delta=False, keep_direction=False,
                 action_noise_std=None):
        self.n = n
        self._max_episode_steps = n
        self.env = DummyGoal()

        self._mazes = mazes_dict
        self.maze_type = maze_type.lower()

        self._ignore_reset_start = bool(ignore_reset_start)
        self._done_on_success = bool(done_on_success)

        self._obs_include_delta = obs_include_delta
        self._keep_direction = keep_direction
        self._action_noise_std = action_noise_std

        self._cur_direction = None

        # Generate a crazy maze specified by its size and generation seed
        if self.maze_type.startswith('crazy'):
            _, size, seed = self.maze_type.split('_')
            size = int(size)
            seed = int(seed)
            self._mazes[self.maze_type] = {'maze': make_crazy_maze(size, seed), 'action_range': 0.95}

        # Generate an "experiment" maze specified by its height, half-width, and size of starting section
        if self.maze_type.startswith('experiment'):
            _, h, half_w, sz0 = self.maze_type.split('_')
            h = int(h)
            half_w = int(half_w)
            sz0 = int(sz0)
            self._mazes[self.maze_type] = {'maze': make_experiment_maze(h, half_w, sz0), 'action_range': 0.25}

        if self.maze_type.startswith('corridor'):
            corridor_length = int(self.maze_type.split('_')[1])
            self._mazes[self.maze_type] = {'maze': make_hallway_maze(corridor_length), 'action_range': 0.95}

        if self.maze_type.startswith('umaze'):
            corridor_length = int(self.maze_type.split('_')[1])
            self._mazes[self.maze_type] = {'maze': make_u_maze(corridor_length), 'action_range': 0.95}

        assert self.maze_type in self._mazes
        self.min_x = self.maze.min_x
        self.max_x = self.maze.max_x
        self.min_y = self.maze.min_y
        self.max_y = self.maze.max_y
        self.min_point = np.array([self.min_x, self.min_y], dtype=np.float32)
        self.max_point = np.array([self.max_x, self.max_y], dtype=np.float32)

        if action_range_override is not None:
            self._mazes[self.maze_type]['action_range'] = action_range_override
        if start_random_range_override is not None:
            self.maze.start_random_range = start_random_range_override

        self.use_antigoal = bool(use_antigoal)
        self.ddiff = bool(ddiff)

        self._state = dict(s0=None, prev_state=None, state=None, goal=None, n=None, done=None, d_goal_0=None,
                           d_antigoal_0=None)

        self.dist_threshold = 0.15

        self.trajectory = []

        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,)),
            'achieved_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            'desired_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        })

        self.action_space = gym.spaces.Box(low=-self.action_range, high=self.action_range, shape=(2,))

        self.reset()

    @staticmethod
    def dist(goal, outcome):
        # return np.sum(np.abs(goal - outcome))
        return np.sqrt(np.sum((goal - outcome) ** 2))

    @property
    def maze(self):
        return self._mazes[self.maze_type]['maze']

    @property
    def action_range(self):
        return self._mazes[self.maze_type]['action_range']

    @property
    def state(self):
        return self._state['state'].reshape(-1)

    @property
    def goal(self):
        return self._state['goal'].reshape(-1)

    @goal.setter
    def goal(self, _):
        pass

    @property
    def antigoal(self):
        return self._state['antigoal'].reshape(-1)

    @property
    def reward(self):
        # r_sparse = -np.ones(1) + float(self.is_success)
        # r_dense = -self.dist(self.goal, self.state)
        # if self.use_antigoal:
        #     r_dense += self.dist(self.antigoal, self.state)
        # if not self.ddiff:
        #     reward = r_sparse + np.clip(r_dense, -np.inf, 0.0)
        # else:
        #     r_dense_prev = -self.dist(self.goal, self._state['prev_state'])
        #     if self.use_antigoal:
        #         r_dense_prev += self.dist(self.antigoal, self._state['prev_state'])
        #     r_dense -= r_dense_prev
        #     reward = r_sparse + r_dense
        # reward = self.state[0] - self._state['prev_state'][0]
        # clipping
        # curr_xy = self.state[0:2].copy()
        # prev_xy = self._state['prev_state'][0:2].copy()
        # curr_xy[0] = np.clip(curr_xy[0], -np.inf, 1)
        # prev_xy[0] = np.clip(prev_xy[0], -np.inf, 1)
        # reward = np.mean(np.square(curr_xy - prev_xy))
        # clipping
        curr_xy = self.state[0:2].copy()
        goal_xy = np.array([1.0, -1.0])
        dist = np.linalg.norm(curr_xy - goal_xy)
        if curr_xy[0] > 1.0:
            dist *= 2
        reward = np.exp(-dist)
        return reward

    @property
    def achieved(self):
        return self.goal if self.is_success else self.state

    @property
    def is_done(self):
        return bool(self._state['done'])

    @property
    def is_success(self):
        d = self.dist(self.goal, self.state)
        return d <= self.dist_threshold

    @property
    def d_goal_0(self):
        return self._state['d_goal_0']

    @property
    def d_antigoal_0(self):
        return self._state['d_antigoal_0']

    @property
    def next_phase_reset(self):
        return {'state': self._state['s0'], 'goal': self.goal, 'antigoal': self.achieved}

    @property
    def sibling_reset(self):
        return {'state': self._state['s0'], 'goal': self.goal}

    def _get_mdp_state(self):
        observation = np.zeros(25)
        achieved_goal = np.zeros(3)
        desired_goal = np.zeros(3)
        observation[0:2] = self.state
        observation[3:5] = self.state
        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
        }

    def reset(self, state=None, goal=None, antigoal=None):
        # if state is None or self._ignore_reset_start:
        #     s_xy = self.maze.sample_start()
        # else:
        #     s_xy = state
        s_xy = np.zeros(2)
        s_xy = np.array(s_xy)
        if goal is None:
            if 'square' in self.maze_type:
                g_xy = self.maze.sample_goal(min_wall_dist=0.025 + self.dist_threshold)
            else:
                g_xy = self.maze.sample_goal()
        else:
            g_xy = goal
        g_xy = np.array(g_xy)

        if antigoal is None:
            ag_xy = g_xy
        else:
            ag_xy = antigoal

        if self._keep_direction:
            self._cur_direction = np.random.random() * 2 * np.pi

        self._state = {
            's0': s_xy,
            'prev_state': s_xy * np.ones_like(s_xy),
            'state': s_xy * np.ones_like(s_xy),
            'goal': g_xy,
            'antigoal': ag_xy,
            'n': 0,
            'done': False,
            'd_goal_0': self.dist(g_xy, s_xy),
            'd_antigoal_0': self.dist(g_xy, ag_xy),
        }
        self._current_action = np.zeros(2)

        self.trajectory = [self.state]

        return self._get_mdp_state()

    def step(self, action):
        obsbefore = self._get_mdp_state()
        self._current_action = np.array(action).copy()

        if self._action_noise_std is not None:
            action = action + np.random.normal(scale=self._action_noise_std, size=action.shape)

        # Clip action
        # for i in range(len(action)):
        #     action[i] = np.clip(action[i], -self.action_range, self.action_range) * 0.2

        # Scale action ST magnitude is in range
        action_mag = np.linalg.norm(action, axis=0, keepdims=True)
        action_utv = action / (action_mag + 1e-6)
        action_mag_clipped = np.clip(action_mag, 0, self.action_range)
        action = action_utv * self.action_range * action_mag_clipped * 0.2

        # Rotational actions
        # action = np.clip(action, -1, 1)
        # angle = np.pi*(action[0])
        # mag = ((action[1] + 1)/2)*self.action_range*0.2
        # action = mag * np.array([np.cos(angle), np.sin(angle)])

        try:
            next_state = self._state['state'] + action
            # next_state[0] = np.clip(next_state[0], -np.inf, 1)
            # if self._keep_direction:
            #     r = (action[0] + self.action_range) / 2
            #     theta = (action[1] + self.action_range) / (2 * self.action_range) * np.pi - np.pi / 2
            #     self._cur_direction += theta
            #     x = r * np.cos(self._cur_direction)
            #     y = r * np.sin(self._cur_direction)
            #     next_state = self.maze.move(
            #         self._state['state'],
            #         np.array([x, y]),
            #     )
            # else:
            #     next_state = self.maze.move(
            #         self._state['state'],
            #         action
            #     )
            next_state = np.array(next_state)
        except:
            print('state', self._state['state'])
            print('action', action)
            raise
        self._state['prev_state'] = self._state['state']
        self._state['state'] = next_state
        self._state['n'] += 1
        # done = self._state['n'] >= self.n
        # if self._done_on_success:
        #     done = done or self.is_success
        done = False
        self._state['done'] = done

        self.trajectory.append(self.state)

        # self.render()

        return self._get_mdp_state(), self.reward, self.is_done, {
            'coordinates': self._state['prev_state'],
            'next_coordinates': self._state['state'],
        }

    def sample(self):
        return self.maze.sample()

    def render(self, *args):
        self.maze.plot(trajectory=self.trajectory)

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        """Plot multiple trajectories onto ax"""
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.maze.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            if trajectory['env_infos']['coordinates'].ndim == 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'],
                    [trajectory['env_infos']['next_coordinates'][-1]]
                ]))
            elif trajectory['env_infos']['coordinates'].ndim > 2:
                # Nested array (due to the child policy)
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'].reshape(-1, 2),
                    trajectory['env_infos']['next_coordinates'].reshape(-1, 2)[-1:]
                ]))

        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        trajectory_eval_metrics = defaultdict(list)
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)

        for trajectory, coordinates_trajectory in zip(trajectories, coordinates_trajectories):
            # terminal distance
            trajectory_eval_metrics['TerminalDistance'].append(np.linalg.norm(
                coordinates_trajectory[0] - coordinates_trajectory[-1]
            ))

            # smoothed length
            smooth_window_size = 5
            num_smooth_samples = 6
            if len(coordinates_trajectory) >= smooth_window_size:
                smoothed_coordinates_trajectory = np.zeros((len(coordinates_trajectory) - smooth_window_size + 1, 2))
                for i in range(2):
                    smoothed_coordinates_trajectory[:, i] = np.convolve(
                        coordinates_trajectory[:, i], [1 / smooth_window_size] * smooth_window_size, mode='valid'
                    )
                idxs = np.round(np.linspace(0, len(smoothed_coordinates_trajectory) - 1, num_smooth_samples)).astype(int)
                smoothed_coordinates_trajectory = smoothed_coordinates_trajectory[idxs]
            else:
                smoothed_coordinates_trajectory = coordinates_trajectory
            sum_distances = 0
            for i in range(len(smoothed_coordinates_trajectory) - 1):
                sum_distances += np.linalg.norm(
                    smoothed_coordinates_trajectory[i] - smoothed_coordinates_trajectory[i + 1]
                )
            trajectory_eval_metrics['SmoothedLength'].append(sum_distances)

        # cell percentage
        num_grids = 10  # per one side
        grid_xs = np.linspace(self.min_x, self.max_x, num_grids + 1)
        grid_ys = np.linspace(self.min_y, self.max_y, num_grids + 1)
        is_exist = np.zeros((num_grids, num_grids))
        for coordinates_trajectory in coordinates_trajectories:
            for x, y in coordinates_trajectory:
                x_idx = np.searchsorted(grid_xs, x)  # binary search
                y_idx = np.searchsorted(grid_ys, y)
                x_idx = np.clip(x_idx, 1, num_grids) - 1
                y_idx = np.clip(y_idx, 1, num_grids) - 1
                is_exist[x_idx, y_idx] = 1
        is_exist = is_exist.flatten()
        cell_percentage = np.sum(is_exist) / len(is_exist)

        eval_metrics = {
            'MaxTerminalDistance': np.max(trajectory_eval_metrics['TerminalDistance']),
            'MeanTerminalDistance': np.mean(trajectory_eval_metrics['TerminalDistance']),
            'MaxSmoothedLength': np.max(trajectory_eval_metrics['SmoothedLength']),
            'MeanSmoothedLength': np.mean(trajectory_eval_metrics['SmoothedLength']),
            'CellPercentage': cell_percentage,
        }

        if is_option_trajectories:
            # option std
            option_terminals = defaultdict(list)
            for trajectory, coordinates_trajectory in zip(trajectories, coordinates_trajectories):
                option_terminals[
                    tuple(trajectory['agent_infos']['option'][0])
                ].append(coordinates_trajectory[-1])
            mean_option_terminals = [np.mean(terminals, axis=0) for terminals in option_terminals.values()]
            intra_option_std = np.mean([np.mean(np.std(terminals, axis=0)) for terminals in option_terminals.values()])
            inter_option_std = np.mean(np.std(mean_option_terminals, axis=0))

            eval_metrics['IntraOptionStd'] = intra_option_std
            eval_metrics['InterOptionStd'] = inter_option_std
            eval_metrics['InterIntraOptionStdDiff'] = inter_option_std - intra_option_std

        return eval_metrics


class MinimalMazeEnv(MazeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'achieved_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            'desired_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        })

    def _get_mdp_state(self):
        observation = np.zeros(2)
        achieved_goal = np.zeros(3)
        desired_goal = np.zeros(3)
        observation[0:2] = self.state
        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
        }
