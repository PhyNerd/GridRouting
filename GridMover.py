import numpy as np
from scipy.ndimage import center_of_mass
import numbers
DEBUG = False


class GridMover(object):
    def __init__(self):
        pass

    def perform_moves(self, from_node, dest_nodes, grid):
        """
        Perform the changes made by movement from from_node to dest_nodes to grid in order.
        Then return said movement if it changed something.
        """
        moves = []
        for to_node in dest_nodes:
            if from_node != to_node:
                row_from, col_from = from_node
                row_to, col_to = to_node
                if isinstance(row_from, numbers.Integral) and isinstance(col_from, numbers.Integral):
                    grid[row_from, col_from] = 0
                if isinstance(row_to, numbers.Integral) and isinstance(col_to, numbers.Integral):
                    grid[row_to, col_to] = 1
                moves += [[from_node, to_node, 0b00]]
                from_node = to_node
        # add rising intensity ramp to first move
        moves[0][2] = moves[0][2] | 0b01
        # add falling intensity ramp to last move
        moves[-1][2] = moves[-1][2] | 0b10

        return moves, grid

    def calc_move(self, from_node, to_node, grid, trivial_movement=False):
        """
        Calculate the moves that need to be made to move a from_node to to_node.
        Trivial movement: move around obstacles
        Normal movement: If there are obstacles in the way move them instead and fill their place after.
        """
        row_from, col_from = from_node
        row_to, col_to = to_node

        # What directions are we moving in
        move_left = (col_to <= col_from)
        move_up = (row_to <= row_from)

        # check if there are obstacles along potential paths
        # then decide which path to move along
        # Path 1: horizontal movement first
        path1_horizontal = grid[row_from, col_to:col_from] if move_left else grid[row_from, col_from + 1:col_to + 1]
        path1_vertical = grid[row_to:row_from, col_to] if move_up else grid[row_from + 1:row_to + 1, col_to]
        path1_obstacles = sum(path1_horizontal) + sum(path1_vertical)  # horizontal movement first

        # Path 2: horizontal movement first
        path2_vertical = grid[row_to:row_from, col_from] if move_up else grid[row_from + 1:row_to + 1, col_from]
        path2_horizontal = grid[row_to, col_to:col_from] if move_left else grid[row_to, col_from + 1:col_to + 1]
        path2_obstacles = sum(path2_vertical) + sum(path2_horizontal)

        if trivial_movement:
            nodes = []
            if path1_obstacles == 0:
                nodes += [(row_from, col_to)]
            elif path2_obstacles == 0:
                nodes += [(row_to, col_from)]
            else:
                row_modifier = -0.5 if move_up else 0.5
                col_modifier = -0.5 if move_left else 0.5
                nodes += [(row_from + row_modifier, col_from + col_modifier)]
                nodes += [(row_to - row_modifier, col_from + col_modifier)]
                nodes += [(row_to - row_modifier, col_to - col_modifier)]
            nodes += [to_node]

            moves, grid = self.perform_moves(from_node, nodes, grid)
        else:
            moves = []
            vertical_first = path1_obstacles <= path2_obstacles  # move vertically first if this isn't inferior to horizontal fist
            if vertical_first:
                pause_node = (row_from, col_to)
                horizontal_path = path1_horizontal
                vertical_path = path1_vertical

                if any(horizontal_path):
                    # obstacle in the horizontal path
                    # move it to the destination insted then move to it's position
                    first_obstacle = np.argwhere(horizontal_path)[-1, 0] if move_left else np.argwhere(horizontal_path)[0, 0] + 1
                    obstacle_node = (row_from, first_obstacle + min(col_to, col_from))
                    move_list, grid = self.calc_move(obstacle_node, to_node, grid)
                    moves += move_list
                    move_list, grid = self.perform_moves(from_node, [obstacle_node], grid)
                    moves += move_list
                elif any(vertical_path):
                    # obstacle in the vertical path
                    # move it to the destination insted then move to it's position
                    first_obstacle = np.argwhere(vertical_path)[-1, 0] if move_up else np.argwhere(vertical_path)[0, 0] + 1
                    obstacle_node = (first_obstacle + min(row_to, row_from), col_to)
                    move_list, grid = self.calc_move(obstacle_node, to_node, grid)
                    moves += move_list
                    move_list, grid = self.perform_moves(from_node, [pause_node, obstacle_node], grid)
                    moves += move_list
                else:
                    # no obstacles
                    # move to the destination
                    move_list, grid = self.perform_moves(from_node, [pause_node, to_node], grid)
                    moves += move_list
            else:
                pause_node = (row_to, col_from)
                horizontal_path = path2_horizontal
                vertical_path = path2_vertical

                if any(vertical_path):
                    # obstacle in the vertical path
                    # move it to the destination insted then move to it's position
                    first_obstacle = np.argwhere(vertical_path)[-1, 0] if move_up else np.argwhere(vertical_path)[0, 0] + 1
                    obstacle_node = (first_obstacle + min(row_to, row_from), col_from)
                    move_list, grid = self.calc_move(obstacle_node, to_node, grid)
                    moves += move_list
                    move_list, grid = self.perform_moves(from_node, [obstacle_node], grid)
                    moves += move_list
                elif any(horizontal_path):
                    # obstacle in the horizontal path
                    # move it to the destination insted then move to it's position
                    first_obstacle = np.argwhere(horizontal_path)[-1, 0] if move_left else np.argwhere(horizontal_path)[0, 0] + 1
                    obstacle_node = (row_to, first_obstacle + min(col_to, col_from))
                    move_list, grid = self.calc_move(obstacle_node, to_node, grid)
                    moves += move_list
                    move_list, grid = self.perform_moves(from_node, [pause_node, obstacle_node], grid)
                    moves += move_list
                else:
                    # no obstacles
                    # move to the destination
                    move_list, grid = self.perform_moves(from_node, [pause_node, to_node], grid)
                    moves += move_list

        return moves, grid

    def get_distances(self, node, nodes):
        nodes = np.array(nodes)
        return (nodes[:, 0] - node[0])**2 + (nodes[:, 1] - node[1])**2

    def get_closest_node_index(self, node, nodes):
        """
        Calculate the clostes node to a given node from a list of nodes
        """
        return np.argmin(self.get_distances(node, nodes))

    def find_paths(self, start_grid, res_grid):
        """
        Calculate what nodes to move where in start_grid to fill all
        nodes that are filled in res_grid.
        """
        modified_res_grid = np.clip(res_grid, 0, None)
        diffgrid = start_grid - modified_res_grid
        to_empty = np.argwhere(diffgrid == 1).tolist()  # find filled spots in reservoir
        to_fill = np.argwhere(diffgrid == -1)  # find empty spots in main grid

        # create routes
        to_route = []
        center_mass = list(map(int, center_of_mass(modified_res_grid)))
        if len(to_fill) > 0:
            to_fill = to_fill[np.argsort(self.get_distances(center_mass, to_fill))]  # reorder around center of mass to reduce moves
            for node in to_fill:
                if len(to_empty) < 1:
                    break
                nearest_node_index = self.get_closest_node_index(node, to_empty)
                nearest_node = to_empty.pop(nearest_node_index)
                to_route.append((tuple(nearest_node), tuple(node)))

        # create the moves that remove excess atoms
        instant_moves = []
        for node in to_empty:
            if res_grid[node[0], node[1]] == -1:
                row_dir = -0.5 if center_mass[0] >= node[0] else 0.5
                col_dir = -0.5 if center_mass[1] >= node[1] else 0.5
                instant_moves.append((node, (node[0] + row_dir, node[1] + col_dir), 0b01))

        return to_route, instant_moves

    def find_route(self, start_grid, res_grid, trivial_movement=False):
        """
        Calculate the moves that need to be made in start_grid to fill
        all filled nodes from res_grid.
        """
        grid = np.asarray(start_grid) if not isinstance(start_grid, np.ndarray) else start_grid.copy()
        paths, instant_moves = self.find_paths(grid, res_grid)
        moves = []

        for from_node, to_node in paths:
            move_list, grid = self.calc_move(from_node, to_node, grid,  trivial_movement=trivial_movement)
            moves += move_list

        # remove unneeded intensity lowering/raising
        for i, (from_node, to_node, ramps) in enumerate(moves[:-1]):
            next_from_node = moves[i + 1][0]
            if to_node == next_from_node and bool(moves[i + 1][2] & 0b01):
                moves[i][2] = moves[i][2] & 0b01
                moves[i + 1][2] = moves[i + 1][2] & 0b10

        # add annihilate moves
        moves += instant_moves

        if DEBUG:
            if np.sum(grid) != np.sum(start_grid):
                raise Exception('Algorythm error atoms lost or gained!')

            if np.sum(start_grid) >= np.sum(res_grid):
                worked_grid = ((res_grid - grid) <= 0)
                if (not worked_grid.all()) or np.sum(grid) != np.sum(start_grid):
                    np.savetxt('before.out', start_grid)
                    np.savetxt('res.out', res_grid)
                    raise Exception('Algorythm did not work! saving grids.')

        return moves


if __name__ == '__main__':
    initial_grid = np.random.randint(2, size=(11, 11))
    res_grid = np.pad(np.ones((7, 7)), ((2, 2), (2, 2)), 'constant', constant_values=(0))
    moves = GridMover().find_route(initial_grid, res_grid)
    for move in moves:
        start, stop, intensity = move
        if intensity == 0b01:
            instenity_string = "raise intensity at the start"
        elif intensity == 0b10:
            instenity_string = "lover intensity at the end"
        elif intensity == 0b11:
            instenity_string = "raise intensity at start and lover it at the end"
        print("move from {0} to {1} {2}".format(start, stop, instenity_string))
