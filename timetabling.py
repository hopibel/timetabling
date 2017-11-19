"""Timetabling using a genetic algorithm"""

import random
from deap import algorithms, base, creator, tools


SLOTS = 90
POP = 100


def gen_ind(courses):
    """Generate individual."""
    ind = []
    for course in courses:
        if course['split'] and random.random() < 80:  # 80% of classes are twice a week
            slot = [random.randrange(SLOTS) for i in range(2)]
        else:
            slot = [random.randrange(SLOTS)]
        ind.append(slot)
    return ind


def main():
    """Entry point if called as executable."""

    # dummy course table
    courses = []
    for name in 'abcdefghij':
        for sec in range(1, 3):
            course = {
                'name': name,
                'section': sec,
                'length': 6,
                'split': 1,
            }
            if random.random() < 0.2:
                course['split'] = 0
            courses.append(course)

    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("ind", gen_ind, courses)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


if __name__ == '__main__':
    main()
