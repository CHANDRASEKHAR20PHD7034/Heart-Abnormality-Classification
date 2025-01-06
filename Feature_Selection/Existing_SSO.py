import numpy as np
from Code.Feature_Selection.optimizer import Optimizer

print("Salp Swarm Optimization algorithm was executing...")
class OriginalSSO(Optimizer):


    def __init__(self, epoch=10000, pop_size=100, **kwargs):

        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def evolve(self, epoch):

        ## Eq. (3.2) in the paper
        c1 = 2 * np.exp(-((4 * (epoch + 1) / self.epoch) ** 2))
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx < self.pop_size / 2:
                c2_list = np.random.random(self.problem.n_dims)
                c3_list = np.random.random(self.problem.n_dims)
                pos_new_1 = self.g_best[self.ID_POS] + c1 * ((self.problem.ub - self.problem.lb) * c2_list + self.problem.lb)
                pos_new_2 = self.g_best[self.ID_POS] - c1 * ((self.problem.ub - self.problem.lb) * c2_list + self.problem.lb)
                pos_new = np.where(c3_list < 0.5, pos_new_1, pos_new_2)
            else:
                # Eq. (3.4) in the paper
                pos_new = (self.pop[idx][self.ID_POS] + self.pop[idx - 1][self.ID_POS]) / 2

            # Check if salps go out of the search space and bring it back then re-calculate its fitness value
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
print("Salp Swarm Optimization algorithm was executed successfully...")