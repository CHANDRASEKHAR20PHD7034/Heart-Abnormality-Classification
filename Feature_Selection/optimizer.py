import numpy as np
from math import gamma
from copy import deepcopy
from Code.Feature_Selection.utils.history import History
from Code.Feature_Selection.utils.problem import Problem
from Code.Feature_Selection.utils.termination import Termination
from Code.Feature_Selection.utils.logger import Logger
from Code.Feature_Selection.utils.validator import Validator
import concurrent.futures as parallel
import os
import time


class Optimizer:


    ID_POS = 0  # Index of position/location of solution/agent
    ID_TAR = 1  # Index of target list, (includes fitness value and objectives list)

    ID_FIT = 0  # Index of target (the final fitness) in fitness
    ID_OBJ = 1  # Index of objective list in target

    EPSILON = 10E-10

    def __init__(self, **kwargs):
        super(Optimizer, self).__init__()
        self.epoch, self.pop_size, self.solution = None, None, None
        self.mode, self.n_workers, self.name = None, None, None
        self.pop, self.g_best, self.g_worst = None, None, None
        self.problem, self.logger, self.history = None, None, None
        self.__set_keyword_arguments(kwargs)
        self.validator = Validator(log_to="console", log_file=None)

        if self.name is None: self.name = self.__class__.__name__
        self.sort_flag, self.terminate_counter, self.nfe_per_epoch = False, None, self.pop_size
        self.parameters, self.params_name_ordered = {}, None
        self.AVAILABLE_MODES = ["process", "thread", "swarm"]
        self.support_parallel_modes = True

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_parameters(self, parameters):

        if type(parameters) in (list, tuple):
            self.params_name_ordered = tuple(parameters)
            self.parameters = {}
            for name in parameters:
                self.parameters[name] = self.__dict__[name]

        if type(parameters) is dict:
            valid_para_names = set(self.parameters.keys())
            new_para_names = set(parameters.keys())
            if new_para_names.issubset(valid_para_names):
                for key, value in parameters.items():
                    setattr(self, key, value)
                    self.parameters[key] = value
            else:
                raise ValueError(f"Invalid input parameters: {new_para_names} for {self.get_name()} optimizer. "
                                 f"Valid parameters are: {valid_para_names}.")

    def get_parameters(self):

        return self.parameters

    def get_attributes(self):

        return self.__dict__

    def get_name(self):
        return self.name

    def __str__(self):
        temp = ""
        for key in self.params_name_ordered:
            temp += f"{key}={self.parameters[key]}, "
        temp = temp[:-2]
        return f"{self.__class__.__name__}({temp})"

    def before_initialization(self, starting_positions=None):
        if starting_positions is None:
            pass
        elif type(starting_positions) in [list, np.ndarray] and len(starting_positions) == self.pop_size:
            if isinstance(starting_positions[0], np.ndarray) and len(starting_positions[0]) == self.problem.n_dims:
                self.pop = [self.create_solution(self.problem.lb, self.problem.ub, pos) for pos in starting_positions]
            else:
                raise ValueError("Starting positions should be a list of positions or 2D matrix of positions only.")
        else:
            raise ValueError("Starting positions should be a list/2D matrix of positions with same length as pop_size hyper-parameter.")

    def initialization(self):
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)

    def after_initialization(self):
        # The initial population is sorted or not depended on algorithm's strategy
        pop_temp, best, worst = self.get_special_solutions(self.pop, best=1, worst=1)
        self.g_best, self.g_worst = best[0], worst[0]
        # pop_temp, self.g_best = self.get_global_best_solution(self.pop)
        if self.sort_flag: self.pop = pop_temp
        ## Store initial best and worst solutions
        self.history.store_initial_best_worst(self.g_best, self.g_worst)

    def initialize_variables(self):
        pass

    def get_target_wrapper(self, position):

        objs = self.problem.fit_func(position)
        if not self.problem.obj_is_list:
            objs = [objs]
        fit = np.dot(objs, self.problem.obj_weights)
        return [fit, objs]

    def create_solution(self, lb=None, ub=None, pos=None):

        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        return [position, target]

    def evolve(self, epoch):
        pass

    def check_problem(self, problem):
        self.problem = problem if isinstance(problem, Problem) else Problem(**problem)
        self.amend_position = self.problem.amend_position
        self.generate_position = self.problem.generate_position
        self.logger = Logger(self.problem.log_to, log_file=self.problem.log_file).create_logger(name=f"{self.__module__}.{self.__class__.__name__}")
        self.logger.info(self.problem.msg)
        self.history = History(log_to=self.problem.log_to, log_file=self.problem.log_file)

    def check_mode_and_workers(self, mode, n_workers):
        self.mode = self.validator.check_str("mode", mode, ["single", "swarm", "thread", "process"])
        if self.mode in ("process", "thread"):
            if not self.support_parallel_modes:
                self.logger.warning(f"{self.__class__.__name__} doesn't support parallelization. The default mode 'single' is activated.")
                self.mode = "single"
            elif n_workers is not None:
                if self.mode == "process":
                    self.n_workers = self.validator.check_int("n_workers", n_workers, [2, min(61, os.cpu_count() - 1)])
                if self.mode == "thread":
                    self.n_workers = self.validator.check_int("n_workers", n_workers, [2, min(32, os.cpu_count() + 4)])
            else:
                self.logger.warning(f"The parallel mode: {self.mode} is selected. But n_workers is not set. The default n_workers = 4 is used.")
                self.n_workers = 4

    def check_termination(self, mode="start", termination=None, epoch=None):
        if mode == "start":
            self.termination = termination
            if termination is not None:
                if isinstance(termination, Termination):
                    self.termination = termination
                elif type(termination) == dict:
                    self.termination = Termination(log_to=self.problem.log_to, log_file=self.problem.log_file, **termination)
                else:
                    raise ValueError("Termination needs to be a dict or an instance of Termination class.")
                self.terminate_counter = self.termination.get_default_counter(self.epoch)
                self.logger.warning(f"Stopping condition mode: {self.termination.name}, with maximum value is: {self.termination.quantity}")
        else:
            finished = False
            if self.termination is not None:
                if self.termination.mode == 'TB':
                    finished = self.termination.is_finished(time.perf_counter() - self.terminate_counter)
                elif self.termination.mode == 'FE':
                    self.terminate_counter += self.nfe_per_epoch
                    finished = self.termination.is_finished(self.terminate_counter)
                elif self.termination.mode == 'MG':
                    finished = self.termination.is_finished(epoch)
                else:  # Early Stopping
                    temp = self.terminate_counter + self.history.get_global_repeated_times(self.ID_TAR, self.ID_FIT, self.EPSILON)
                    finished = self.termination.is_finished(temp)
                if finished:
                    self.logger.warning(f"Stopping criterion with mode {self.termination.name} occurred. End program!")
            return finished

    def solve(self, problem=None, mode='single', starting_positions=None, n_workers=None, termination=None):

        self.check_problem(problem)
        self.check_mode_and_workers(mode, n_workers)
        self.check_termination("start", termination, None)
        self.initialize_variables()

        self.before_initialization(starting_positions)
        self.initialization()
        self.after_initialization()

        for epoch in range(0, self.epoch):
            time_epoch = time.perf_counter()

            ## Evolve method will be called in child class
            self.evolve(epoch)

            # Update global best position, the population is sorted or not depended on algorithm's strategy
            pop_temp, self.g_best = self.update_global_best_solution(self.pop)
            if self.sort_flag: self.pop = pop_temp

            time_epoch = time.perf_counter() - time_epoch
            self.track_optimize_step(self.pop, epoch + 1, time_epoch)
            if self.check_termination("end", None, epoch+1):
                break
        self.track_optimize_process()
        return self.solution[self.ID_POS], self.solution[self.ID_TAR][self.ID_FIT]

    def track_optimize_step(self, population=None, epoch=None, runtime=None):

        ## Save history data
        pop = deepcopy(population)
        if self.problem.save_population:
            self.history.list_population.append(pop)
        self.history.list_epoch_time.append(runtime)
        self.history.list_global_best_fit.append(self.history.list_global_best[-1][self.ID_TAR][self.ID_FIT])
        self.history.list_current_best_fit.append(self.history.list_current_best[-1][self.ID_TAR][self.ID_FIT])
        # Save the exploration and exploitation data for later usage
        pos_matrix = np.array([agent[self.ID_POS] for agent in pop])
        div = np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0)
        self.history.list_diversity.append(np.mean(div, axis=0))
        ## Print epoch
        self.logger.info(f">Problem: {self.problem.name}, Epoch: {epoch}, Current best: {self.history.list_current_best[-1][self.ID_TAR][self.ID_FIT]}, "
                         f"Global best: {self.history.list_global_best[-1][self.ID_TAR][self.ID_FIT]}, Runtime: {runtime:.5f} seconds")

    def track_optimize_process(self):

        self.history.epoch = len(self.history.list_diversity)
        div_max = np.max(self.history.list_diversity)
        self.history.list_exploration = 100 * (np.array(self.history.list_diversity) / div_max)
        self.history.list_exploitation = 100 - self.history.list_exploration
        self.history.list_global_best = self.history.list_global_best[1:]
        self.history.list_current_best = self.history.list_current_best[1:]
        self.solution = self.history.list_global_best[-1]
        self.history.list_global_worst = self.history.list_global_worst[1:]
        self.history.list_current_worst = self.history.list_current_worst[1:]

    def create_population(self, pop_size=None):

        if pop_size is None:
            pop_size = self.pop_size
        pop = []
        if self.mode == "thread":
            with parallel.ThreadPoolExecutor(self.n_workers) as executor:
                list_executors = [executor.submit(self.create_solution, self.problem.lb, self.problem.ub) for _ in range(pop_size)]
                # This method yield the result everytime a thread finished their job (not by order)
                for f in parallel.as_completed(list_executors):
                    pop.append(f.result())
        elif self.mode == "process":
            with parallel.ProcessPoolExecutor(self.n_workers) as executor:
                list_executors = [executor.submit(self.create_solution, self.problem.lb, self.problem.ub) for _ in range(pop_size)]
                # This method yield the result everytime a cpu finished their job (not by order).
                for f in parallel.as_completed(list_executors):
                    pop.append(f.result())
        else:
            pop = [self.create_solution(self.problem.lb, self.problem.ub) for _ in range(0, pop_size)]
        return pop

    def update_target_wrapper_population(self, pop=None):

        pos_list = [agent[self.ID_POS] for agent in pop]
        if self.mode == "thread":
            with parallel.ThreadPoolExecutor(self.n_workers) as executor:
                list_results = executor.map(self.get_target_wrapper, pos_list)  # Return result as original order, not the future object
                for idx, target in enumerate(list_results):
                    pop[idx][self.ID_TAR] = target
        elif self.mode == "process":
            with parallel.ProcessPoolExecutor(self.n_workers) as executor:
                list_results = executor.map(self.get_target_wrapper, pos_list)  # Return result as original order, not the future object
                for idx, target in enumerate(list_results):
                    pop[idx][self.ID_TAR] = target
        elif self.mode == "swarm":
            for idx, pos in enumerate(pos_list):
                pop[idx][self.ID_TAR] = self.get_target_wrapper(pos)
        return pop

    def get_global_best_solution(self, pop: list):

        sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT])  # Already returned a new sorted list
        if self.problem.minmax == "min":
            return sorted_pop, deepcopy(sorted_pop[0])
        else:
            return sorted_pop, deepcopy(sorted_pop[-1])

    def get_better_solution(self, agent1: list, agent2: list, reverse=False):

        if self.problem.minmax == "min":
            if agent1[self.ID_TAR][self.ID_FIT] < agent2[self.ID_TAR][self.ID_FIT]:
                return deepcopy(agent1) if reverse is False else deepcopy(agent2)
            return deepcopy(agent2) if reverse is False else deepcopy(agent1)
        else:
            if agent1[self.ID_TAR][self.ID_FIT] < agent2[self.ID_TAR][self.ID_FIT]:
                return deepcopy(agent2) if reverse is False else deepcopy(agent1)
            return deepcopy(agent1) if reverse is False else deepcopy(agent2)

    def compare_agent(self, agent_new: list, agent_old: list):

        if self.problem.minmax == "min":
            if agent_new[self.ID_TAR][self.ID_FIT] < agent_old[self.ID_TAR][self.ID_FIT]:
                return True
            return False
        else:
            if agent_new[self.ID_TAR][self.ID_FIT] < agent_old[self.ID_TAR][self.ID_FIT]:
                return False
            return True

    def get_special_solutions(self, pop=None, best=3, worst=3):

        if self.problem.minmax == "min":
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT])
        else:
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=True)
        if best is None:
            if worst is None:
                raise ValueError("Best and Worst can not be None in get_special_solutions function!")
            else:
                return pop, None, deepcopy(pop[::-1][:worst])
        else:
            if worst is None:
                return pop, deepcopy(pop[:best]), None
            else:
                return pop, deepcopy(pop[:best]), deepcopy(pop[::-1][:worst])

    def get_special_fitness(self, pop=None):

        total_fitness = np.sum([agent[self.ID_TAR][self.ID_FIT] for agent in pop])
        if self.problem.minmax == "min":
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT])
        else:
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=True)
        return total_fitness, pop[0][self.ID_TAR][self.ID_FIT], pop[-1][self.ID_TAR][self.ID_FIT]

    def update_global_best_solution(self, pop=None, save=True):

        if self.problem.minmax == "min":
            sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT])
        else:
            sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=True)
        current_best = sorted_pop[0]
        current_worst = sorted_pop[-1]
        if save:
            ## Save current best
            self.history.list_current_best.append(current_best)
            better = self.get_better_solution(current_best, self.history.list_global_best[-1])
            self.history.list_global_best.append(better)
            ## Save current worst
            self.history.list_current_worst.append(current_worst)
            worse = self.get_better_solution(current_worst, self.history.list_global_worst[-1], reverse=True)
            self.history.list_global_worst.append(worse)
            return deepcopy(sorted_pop), deepcopy(better)
        else:
            ## Handle current best
            local_better = self.get_better_solution(current_best, self.history.list_current_best[-1])
            self.history.list_current_best[-1] = local_better
            global_better = self.get_better_solution(current_best, self.history.list_global_best[-1])
            self.history.list_global_best[-1] = global_better
            ## Handle current worst
            local_worst = self.get_better_solution(current_worst, self.history.list_current_worst[-1], reverse=True)
            self.history.list_current_worst[-1] = local_worst
            global_worst = self.get_better_solution(current_worst, self.history.list_global_worst[-1], reverse=True)
            self.history.list_global_worst[-1] = global_worst
            return deepcopy(sorted_pop), deepcopy(global_better)

    def get_index_best(self, pop):
        fit_list = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in pop])
        if self.problem.minmax == "min":
            return np.argmin(fit_list)
        else:
            return np.argmax(fit_list)

    ## Selection techniques
    def get_index_roulette_wheel_selection(self, list_fitness: np.array):

        size = len(list_fitness)
        if np.any(list_fitness) < 0:
            list_fitness = list_fitness - np.min(list_fitness)
        final_fitness = list_fitness
        if self.problem.minmax == "min":
            final_fitness = np.max(list_fitness) - list_fitness
        prob = final_fitness / np.sum(final_fitness)
        return np.random.choice(range(0, size), p=prob)

    def get_index_kway_tournament_selection(self, pop=None, k_way=0.2, output=2, reverse=False):

        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))
        list_id = np.random.choice(range(len(pop)), k_way, replace=False)
        list_parents = [[idx, pop[idx][self.ID_TAR][self.ID_FIT]] for idx in list_id]
        if self.problem.minmax == "min":
            list_parents = sorted(list_parents, key=lambda agent: agent[1])
        else:
            list_parents = sorted(list_parents, key=lambda agent: agent[1], reverse=True)
        if reverse:
            return [parent[0] for parent in list_parents[-output:]]
        return [parent[0] for parent in list_parents[:output]]

    def get_levy_flight_step(self, beta=1.0, multiplier=0.001, size=None, case=0):

        # u and v are two random variables which follow np.random.normal distribution
        # sigma_u : standard deviation of u
        sigma_u = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        size = 1 if size is None else size
        u = np.random.normal(0, sigma_u ** 2, size)
        v = np.random.normal(0, sigma_v ** 2, size)
        s = u / np.power(np.abs(v), 1 / beta)
        if case == 0:
            step = multiplier * s * np.random.uniform()
        elif case == 1:
            step = multiplier * s * np.random.normal(0, 1)
        else:
            step = multiplier * s
        return step[0] if size == 1 else step

    ### Survivor Selection
    def greedy_selection_population(self, pop_old=None, pop_new=None):

        len_old, len_new = len(pop_old), len(pop_new)
        if len_old != len_new:
            raise ValueError("Greedy selection of two population with different length.")
        if self.problem.minmax == "min":
            return [pop_new[i] if pop_new[i][self.ID_TAR][self.ID_FIT] < pop_old[i][self.ID_TAR][self.ID_FIT]
                    else pop_old[i] for i in range(len_old)]
        else:
            return [pop_new[i] if pop_new[i][self.ID_TAR] > pop_old[i][self.ID_TAR]
                    else pop_old[i] for i in range(len_old)]

    def get_sorted_strim_population(self, pop=None, pop_size=None, reverse=False):

        if self.problem.minmax == "min":
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=reverse)
        else:
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=reverse)
        return pop[:pop_size]

    def create_opposition_position(self, agent=None, g_best=None):

        return self.problem.lb + self.problem.ub - g_best[self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - agent[self.ID_POS])

    def create_pop_group(self, pop, n_groups, m_agents):
        pop_group = []
        for i in range(0, n_groups):
            group = pop[i * m_agents: (i + 1) * m_agents]
            pop_group.append(deepcopy(group))
        return pop_group

    ### Crossover
    def crossover_arithmetic(self, dad_pos=None, mom_pos=None):

        r = np.random.uniform()  # w1 = w2 when r =0.5
        w1 = np.multiply(r, dad_pos) + np.multiply((1 - r), mom_pos)
        w2 = np.multiply(r, mom_pos) + np.multiply((1 - r), dad_pos)
        return w1, w2

    def improved_ms(self, pop=None, g_best=None):  ## m: mutation, s: search
        pop_len = int(len(pop) / 2)
        ## Sort the updated population based on fitness
        pop = sorted(pop, key=lambda item: item[self.ID_TAR][self.ID_FIT])
        pop_s1, pop_s2 = pop[:pop_len], pop[pop_len:]

        ## Mutation scheme
        pop_new = []
        for i in range(0, pop_len):
            agent = deepcopy(pop_s1[i])
            pos_new = pop_s1[i][self.ID_POS] * (1 + np.random.normal(0, 1, self.problem.n_dims))
            agent[self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append(agent)
        pop_new = self.update_target_wrapper_population(pop_new)
        pop_s1 = self.greedy_selection_population(pop_s1, pop_new)  ## Greedy method --> improved exploitation

        ## Search Mechanism
        pos_s1_list = [item[self.ID_POS] for item in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        pop_new = []
        for i in range(0, pop_len):
            agent = deepcopy(pop_s2[i])
            pos_new = (g_best[self.ID_POS] - pos_s1_mean) - np.random.random() * \
                      (self.problem.lb + np.random.random() * (self.problem.ub - self.problem.lb))
            agent[self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append(agent)
        ## Keep the diversity of populatoin and still improved the exploration
        pop_s2 = self.update_target_wrapper_population(pop_new)
        pop_s2 = self.greedy_selection_population(pop_s2, pop_new)

        ## Construct a new population
        pop = pop_s1 + pop_s2
        return pop