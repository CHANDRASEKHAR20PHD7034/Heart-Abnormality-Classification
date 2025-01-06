import random
import sys

#from Code.Feature_Selection.Existing_DSO import Dove,Behavior


class dso:
    def __init__(self):
        pass

    @staticmethod
    def run(num_iterations, function, num_doves, MR, num_dimensions, v_max, pos, vel):
        num_seeking = (int)((MR * num_doves) / 100)
        best = sys.maxsize
        best_pos = None
        dove_population = []

        behavior_pattern = dso.generate_behavior(num_doves, num_seeking)
        '''for idx in range(num_doves):
            dove_population.append(Dove(
                behavior=behavior_pattern[idx],
                position=pos[idx][:num_dimensions],
                velocities=vel[idx][:num_dimensions],
                vmax=v_max
            ))'''
        score_doves = {}

        for _ in range(num_iterations):
            # evaluate

            for dove in dove_population:
                score, pos = dove.evaluate(function)
                score_doves[dove] = max(score_doves.get(dove, 0), score)

                if score < best:
                    best = score
                    best_pos = pos.copy()

            # apply behavior
            for dove in dove_population:
                dove.move(function, best_pos)

            # change behavior
            behavior_pattern = dso.generate_behavior(num_doves, num_seeking)
            for idx, dove in enumerate(dove_population):
                dove.behavior = behavior_pattern[idx]

        return best, best_pos, score_doves

    #@staticmethod
    '''def generate_behavior(num_doves, num_seeking):
        behavior_pattern = [Behavior.TRACING] * num_doves
        for _ in range(num_seeking):
            behavior_pattern[random.randint(0, num_doves - 1)] = Behavior.SEEKING

        return behavior_pattern'''