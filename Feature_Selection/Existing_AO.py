import numpy as np
from Code.Feature_Selection.optimizer import Optimizer

print("Aquila Optimizer was executing...")
class OriginalAO(Optimizer):


    def __init__(self, epoch=10000, pop_size=100, **kwargs):

        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):

        alpha = delta = 0.1
        g1 = 2 * np.random.rand() - 1  # Eq. 16
        g2 = 2 * (1 - epoch / self.epoch)  # Eq. 17

        dim_list = np.array(list(range(1, self.problem.n_dims + 1)))
        miu = 0.00565
        r0 = 10
        r = r0 + miu * dim_list
        w = 0.005
        phi0 = 3 * np.pi / 2
        phi = -w * dim_list + phi0
        x = r * np.sin(phi)  # Eq.(9)
        y = r * np.cos(phi)  # Eq.(10)
        QF = (epoch + 1) ** ((2 * np.random.rand() - 1) / (1 - self.epoch) ** 2)  # Eq.(15)        Quality function

        pop_new = []
        for idx in range(0, self.pop_size):
            x_mean = np.mean(np.array([item[self.ID_TAR][self.ID_FIT] for item in self.pop]), axis=0)
            levy_step = self.get_levy_flight_step(beta=1.5, multiplier=1.0, case=-1)
            if (epoch + 1) <= (2 / 3) * self.epoch:  # Eq. 3, 4
                if np.random.rand() < 0.5:
                    pos_new = self.g_best[self.ID_POS] * (1 - (epoch + 1) / self.epoch) + \
                              np.random.rand() * (x_mean - self.g_best[self.ID_POS])
                else:
                    idx = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                    pos_new = self.g_best[self.ID_POS] * levy_step + self.pop[idx][self.ID_POS] + np.random.rand() * (y - x)  # Eq. 5
            else:
                if np.random.rand() < 0.5:
                    pos_new = alpha * (self.g_best[self.ID_POS] - x_mean) - np.random.rand() * \
                              (np.random.rand() * (self.problem.ub - self.problem.lb) + self.problem.lb) * delta  # Eq. 13
                else:
                    pos_new = QF * self.g_best[self.ID_POS] - (g2 * self.pop[idx][self.ID_POS] * np.random.rand()) - \
                              g2 * levy_step + np.random.rand() * g1  # Eq. 14
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
print("Aquila Optimizer was executed successfully...")