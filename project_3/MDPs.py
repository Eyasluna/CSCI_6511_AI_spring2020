import numpy as np
import time
import matplotlib.pyplot as plt
import random as rand
import linecache


class GridWorld:
    # Value Iteration Algorithm and Policy Iteration Algorithm to Solve GridWorld Game.
    # Args:
    #     prize_grid: Description of the environment
    #     trm_msk: Terminal of the environment
    #     wall_mask: Wall of the environment
    #     iters: Num of iteration times
    #     discount: Gamma time discount factor.
    #
    # Returns:
    #     policy_grids: the optimal policy
    #     util_grids: the optimal value function

    # north, east, south, west
    directions = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    ]
    _num_moves = len(directions)

    def __init__(self,
                 prize_grid,
                 trm_msk,
                 wall_mask,
                 move_probs,
                 no_move_prob):

        self._prize_grid = prize_grid
        self._trm_msk = trm_msk
        self._wall_mask = wall_mask
        self._T = self._make_trans_matrix(
            move_probs,
            no_move_prob,
            wall_mask
        )

    @property
    def shape(self):
        return self._prize_grid.shape

    @property
    def size(self):
        return self._prize_grid.size

    @property
    def prize_grid(self):
        return self._prize_grid

    # Value Iteration Algorithm.
    def run_val_iters(self, discount=.9,
                      iters=10):
        util_grids, policy_grids = self._init_util_policy_storage(iters)

        util_grid = np.zeros_like(self._prize_grid)
        for i in range(iters):
            util_grid = self._val_iter(util_grid=util_grid)
            # optimize the policy value
            policy_grids[:, :, i] = self.optimal_policy(util_grid)
            # change the policy
            util_grids[:, :, i] = util_grid
        return policy_grids, util_grids

    # Policy Iteration Algorithm.
    def run_policy_iters(self, discount=.9,iters=10):
        util_grids, policy_grids = self._init_util_policy_storage(iters)

        # start with a random policy
        policy_grid = np.random.randint(0, self._num_moves,
                                        self.shape)
        util_grid = self._prize_grid.copy()

        for i in range(iters):
            policy_grid, util_grid = self._policy_iter(
                policy_grid=policy_grid,
                util_grid=util_grid
            )
            policy_grids[:, :, i] = policy_grid
            util_grids[:, :, i] = util_grid
        return policy_grids, util_grids

    def gen_exp(self, current_state_idx, move_idx):
        A, B = self.grid_indices_to_coords(current_state_idx)
        next_state_probs = self._T[A, B, move_idx, :, :].flatten()
        next_state_idx = np.random.choice(np.arange(next_state_probs.size),
                                          p=next_state_probs)

        return (next_state_idx,
                self._prize_grid.flatten()[next_state_idx],
                self._trm_msk.flatten()[next_state_idx])

    def grid_indices_to_coords(self, indices=None):
        if indices is None:
            indices = np.arange(self.size)
        return np.unravel_index(indices, self.shape)

    def grid_coords_to_indices(self, coords=None):
        # only works for + indices
        if coords is None:
            return np.arange(self.size)
        return np.ravel_multi_index(coords, self.shape)

    def optimal_policy(self, util_grid):
        M, N = self.shape
        return np.argmax((util_grid.reshape((1, 1, 1, M, N)) * self._T)
                         .sum(axis=-1).sum(axis=-1), axis=2)

    def _init_util_policy_storage(self, depth):
        M, N = self.shape
        util_grids = np.zeros((M, N, depth))
        policy_grids = np.zeros_like(util_grids)
        return util_grids, policy_grids

    def _make_trans_matrix(self,
                           move_probs,
                           no_move_prob,
                           wall_mask):
        M, N = self.shape

        T = np.zeros((M, N, self._num_moves, M, N))

        A, B = self.grid_indices_to_coords()

        T[A, B, :, A, B] += no_move_prob

        for move in range(self._num_moves):
            for offset, P in move_probs:
                direction = (move + offset) % self._num_moves

                dr, dc = self.directions[direction]
                A1 = np.clip(A + dr, 0, M - 1)
                B1 = np.clip(B + dc, 0, N - 1)

                # Flatten puts the array into one dimension
                temp_mask = wall_mask[A1, B1].flatten()
                A1[temp_mask] = A[temp_mask]
                B1[temp_mask] = B[temp_mask]

                T[A, B, move, A1, B1] += P

        terminal_locs = np.where(self._trm_msk.flatten())[0]
        T[A[terminal_locs], B[terminal_locs], :, :, :] = 0
        return T

    def _val_iter(self, util_grid, discount=.9):
        out = np.zeros_like(util_grid)
        M, N = self.shape
        for i in range(M):
            for j in range(N):
                out[i, j] = self._calculate_util((i, j),
                                                 discount,
                                                 util_grid)
        return out

    def _policy_iter(self, *, util_grid,
                     policy_grid, discount=.9):
        r, c = self.grid_indices_to_coords()

        M, N = self.shape
        # Evaluate the current policy
        util_grid = (
                self._prize_grid +
                discount * ((util_grid.reshape((1, 1, 1, M, N)) * self._T)
                            .sum(axis=-1).sum(axis=-1))[r, c, policy_grid.flatten()]
                .reshape(self.shape)
        )

        util_grid[self._trm_msk] = self._prize_grid[self._trm_msk]

        return self.optimal_policy(util_grid), util_grid

    # caculate current value
    def _calculate_util(self, loc, discount, util_grid):
        if self._trm_msk[loc]:
            return self._prize_grid[loc]
        row, col = loc
        return np.max(
            discount * np.sum(
                np.sum(self._T[row, col, :, :, :] * util_grid,
                       axis=-1),
                axis=-1)
        ) + self._prize_grid[loc]

    # plot the action policy
    def plot_policy(self, util_grid, policy_grid=None):
        if policy_grid is None:
            policy_grid = self.optimal_policy(util_grid)
        markers = "^>v<"
        marker_size = 100 // np.max(policy_grid.shape)
        marker_edge_width = marker_size // 10
        marker_fill_color = 'r'

        no_move_mask = self._trm_msk | self._wall_mask

        util_norm = (util_grid - util_grid.min()) / \
                    (util_grid.max() - util_grid.min())

        util_norm = (255 * util_norm).astype(np.uint8)

        for i, marker in enumerate(markers):
            y, x = np.where((policy_grid == i) & np.logical_not(no_move_mask))
            plt.plot(x, y, marker, ms=marker_size, mew=marker_edge_width,
                     color=marker_fill_color)

        y, x = np.where(self._trm_msk)
        plt.plot(x, y, 'o', ms=marker_size, mew=marker_edge_width,
                 color=marker_fill_color)

        tick_step_options = np.array([1, 2, 5, 10, 20, 50, 100])
        tick_step = np.max(policy_grid.shape) / 7
        optimal_option = np.argmin(np.abs(np.log(tick_step) - np.log(tick_step_options)))
        tick_step = tick_step_options[optimal_option]
        plt.xticks(np.arange(0, policy_grid.shape[1] - 0.5, tick_step))
        plt.yticks(np.arange(0, policy_grid.shape[0] - 0.5, tick_step))
        plt.xlim([-0.5, policy_grid.shape[0] - 0.5])
        plt.xlim([-0.5, policy_grid.shape[1] - 0.5])
        plt.savefig('i1_policy_grid.png')


# plot the convergence of iterations
def plot_convergence(util_grids, policy_grids):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    util_ssd = np.sum(np.square(np.diff(util_grids)), axis=(0, 1))
    ax1.plot(util_ssd, 'b.-')
    ax1.set_ylabel('Change in policy value', color='b')

    policy_changes = np.count_nonzero(np.diff(policy_grids), axis=(0, 1))
    ax2.plot(policy_changes, 'r.-')
    ax2.set_ylabel('Change in optimal policy', color='r')
    plt.savefig('i1_plot_convergence.png')


if __name__ == '__main__':
    # import the test data
    file_path = 'i1.txt'
    # definition of the enviroment
    size = int(linecache.getline(file_path, 1).strip())
    discount = float(linecache.getline(file_path, 2).strip())
    noise = linecache.getline(file_path, 3).strip()
    shape = (size, size)
    grid = np.loadtxt(file_path, dtype=str, delimiter=',', skiprows=4, comments='#')
    prize_grid = np.zeros(shape)
    trm_msk = np.zeros_like(prize_grid, dtype=np.bool)
    wall_mask = np.zeros_like(prize_grid, dtype=np.bool)
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            if grid[i - 1, j - 1] != 'X':
                prize_grid[i - 1, j - 1] = int(grid[i - 1, j - 1])
                trm_msk[i - 1, j - 1] = True
    # set the start position
    start = (0, 0)

    gw = GridWorld(prize_grid=prize_grid,
                   wall_mask=wall_mask,
                   trm_msk=trm_msk,
                   move_probs=[
                       (-1, 0.1),
                       (0, 0.8),
                       (1, 0.1),
                   ],
                   no_move_prob=0.0)

    mdp_solvers = {'Value iteration': gw.run_val_iters,
                   'Policy iteration': gw.run_policy_iters}

    # Test the value function and the policy function
    for solver_name, solver_fn in mdp_solvers.items():
        print('Result of {}:'.format(solver_name))
        begintime = time.time()
        policy_grids, util_grids = solver_fn(iters=25, discount=discount)
        endtime = time.time()
        print('Policy grids:')
        print(policy_grids[:, :, -1])
        print('Policy value grids:')
        print(util_grids[:, :, -1])
        print('Solving time of {}:'.format(solver_name))
        print(endtime - begintime)
        plt.figure()
        gw.plot_policy(util_grids[:, :, -1])
        plot_convergence(util_grids, policy_grids)
        plt.show()