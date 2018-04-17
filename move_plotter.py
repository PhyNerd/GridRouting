import numpy as np
from GridMover import GridMover
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

class MovePlotter(object):
    def __init__(self):
        self.calibration_offset = (0, 0)
        self.calibration_matrix = np.array([[.1, 0], [0, .1]])
        self.calibration_shape = (11, 11)

        self.ramp_speed = 1000  # in Nodes per s
        self.amplitude = 0  # in V for Intensity control
        self.mover = GridMover()

    def plot_moves(self, grid, ao_data, speed=50):
        fig = plt.figure()
        ax = plt.subplot(111)

        # create the parametric curve
        node_arr = (np.apply_along_axis(self.voltage_to_node, 1, ao_data[:, 0:2]) + 0.5) / 11.0
        y = 1 - node_arr[:, 0]
        x = node_arr[:, 1]
        z = ao_data[:, 2] / np.max(ao_data[:, 2])

        # create the first plot
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        image = ax.imshow(grid, extent=(0, 1, 0, 1), cmap='Greys', interpolation='nearest')
        ax.set_yticks(np.arange(0, 1, 1 / float(grid.shape[1])))
        ax.set_xticks(np.arange(0, 1, 1 / float(grid.shape[0])))
        ax.grid(color='gray', linestyle='-', linewidth=1)
        text = ax.text(1, 1, '0ms', fontsize=15)
        point, = ax.plot([x[0]], [y[0]], 'o', alpha=z[0])

        # second option - move the point position at every frame
        def update_point(n, x, y, z, point):
            n = n * speed
            if len(x) > n:
                point.set_data(np.array([x[n], y[n]]))
                point.set_alpha(alpha=z[n])
                for timeing, grid in self.grids:
                    if timeing - speed <= n:
                        image.set_data(grid)
                    else:
                        break
                text.set_text('{}ms'.format(n / samplerate * 1e3))
            return point

        ani = animation.FuncAnimation(fig, update_point, len(x) / speed+1, fargs=(x, y, z, point), interval=1)
        plt.show()
        #writer = FFMpegWriter(fps=10)
        #ani.save('im.mp4', writer=writer)

    def ramp(self, v0, v1, amplitude, ramp_samples, sin=True, int_ramps=0b00):
        ramps = []

        # raise trap
        if bool(int_ramps & 0b01):
            ramps.append(np.array([self.rise_ramps[0] * v0[0], self.rise_ramps[0] * v0[1], self.rise_ramps[1]]).T)

        # move Tweezer
        base_ramp = (np.sin(np.linspace(-0.5 * np.pi, 0.5 * np.pi, num=ramp_samples)) + 1) / 2.0 if sin else np.linspace(0, 1, num=ramp_samples)
        ramps.append(np.array([base_ramp * (v1[0] - v0[0]) + v0[0], base_ramp * (v1[1] - v0[1]) + v0[1], np.full(ramp_samples, amplitude)]).T)

        # lower trap
        if bool((int_ramps & 0b10) >> 1):
            ramps.append(np.array([self.fall_ramps[0] * v1[0], self.fall_ramps[0] * v1[1], self.fall_ramps[1]]).T)

        return ramps

    def create_voltage_ramps(self, grid):
        moves_list = self.mover.find_route(grid.copy(), self.res_grid)
        ramps = []
        for from_node, to_node, int_ramps in moves_list:
            # calculate and append new ramp for later concatenation
            ramp_samples = int(samplerate / float(self.ramp_speed) * (np.abs(to_node[0] - from_node[0]) + np.abs(to_node[1] - from_node[1])))
            ramps.extend(self.ramp(self.node_to_voltage(from_node), self.node_to_voltage(to_node), self.amplitude, ramp_samples, int_ramps=int_ramps))

            # DEBUG
            if not hasattr(self, 'grids') or self.grids is None:
                self.grids = [(0, grid.copy())]
            total_len = np.sum([len(rampe) for rampe in ramps])
            row_from, col_from = from_node
            row_to, col_to = to_node
            if isinstance(row_from, int) and isinstance(col_from, int):
                grid[row_from, col_from] = 0
            if isinstance(row_to, int) and isinstance(col_to, int):
                grid[row_to, col_to] = 1
            self.grids += [(total_len, grid.copy())]
            # DEBUG

        return np.concatenate(ramps)

    def node_to_voltage(self, node):
        """ Return the mapped grid node to Voltage space """
        # explicit multiplication is faster than np.dot()
        return (self.calibration_matrix[0, 0] * node[0] + self.calibration_matrix[0, 1] * node[1] + self.calibration_offset[0],
                self.calibration_matrix[1, 0] * node[0] + self.calibration_matrix[1, 1] * node[1] + self.calibration_offset[1])

    def voltage_to_node(self, node):
        """ Return the mapped grid node to Voltage space """
        return np.dot(np.linalg.inv(self.calibration_matrix), node - self.calibration_offset)

    def plot(self, grid, res_grid, ramp_speed, amplitude):
        self.ramp_speed = ramp_speed  # in V per s
        self.amplitude = amplitude  # in V for Intensity control
        self.res_grid = res_grid

        self.rise_time = 300e-6
        self.fall_time = 300e-6
        samples_rise = int(samplerate * self.rise_time)
        samples_fall = int(samplerate * self.fall_time)
        self.rise_ramps = [np.ones(samples_rise), np.linspace(0, self.amplitude, samples_rise)]
        self.fall_ramps = [np.ones(samples_fall), np.linspace(self.amplitude, 0, samples_fall)]

        if res_grid.shape != self.calibration_shape:
            raise Exception("""The result grid shape {} doesn't match the grid shape of the Tweezer calibration {}!
                               Please adjust the result grid or recalibrate.""".format(res_grid.shape, self.calibration_shape))

        ao_data = self.create_voltage_ramps(grid.copy())
        self.plot_moves(grid, ao_data)  # DEBUG


if __name__ == '__main__':
    res_grid = np.pad(np.ones((7, 7)), ((2, 2), (2, 2)), 'constant', constant_values=(0))
    controler = MovePlotter()
    ramp_speed = 1000 # nodes /s
    amplitude = 5
    samplerate = 1e5
    grid = np.random.randint(2, size=(11, 11))
    controler.plot(grid, res_grid, ramp_speed, amplitude)
