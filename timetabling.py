"""Timetabling using a genetic algorithm"""

import random
import itertools
import numpy
from deap import algorithms, base, creator, tools

SLOTS = 90


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


def eval_timetable(individual, courses):
    """Calculate timetable cost.

    Currently returns:
    Number of overlapping classes
    """
    overlap = 0

    for i, course in enumerate(individual):
        length = courses[i]['length']
        sessions = len(course)
        lps = int(length / sessions)

        if length % sessions != 0:
            raise ValueError(
                "Course {} with length {} cannot be divided into {} sessions per week".format(
                    i, length, sessions))

        # self-overlap
        for a, b in itertools.combinations(course, 2):
            width = max(a, b)+lps - min(a, b)
            if width < 2*lps:
                overlap += 2*lps - width

        # overlap
        for j in range(i+1, len(individual)):
            for a, b in itertools.product(course, individual[j]):
                blps = int(courses[j]['length'] / len(individual[j]))
                width = max(a+lps, b+blps) - min(a, b)
                if width < lps + blps:
                    overlap += lps + blps - width

    return (overlap,)


def mut_timetable(individual, courses):
    """Move a class up or down a single time slot."""
    course = random.choice(range(len(individual)))
    session = random.randrange(len(individual(course)))
    course[session] += random.choice((1, -1))
    # TODO: bounds checking
    return (individual,)


def main():
    """Entry point if called as executable."""

    random.seed(64)

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

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("ind", gen_ind, courses=courses)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_timetable, courses=courses)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", mut_timetable, courses=courses)
    toolbox.register("select", tools.selTournament, tournsize=2)

    gens = 1000  # generations
    mu = 1000  # population size
    lambd = 1000  # offspring to create
    cxpb = 0.7  # crossover probability
    mutpb = 0.2  # mutation probability

    pop = toolbox.population(n=mu)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    algorithms.eaMuPlusLambda(
        pop, toolbox, mu, lambd, cxpb, mutpb, gens, stats=stats, halloffame=hof, verbose=True)


if __name__ == '__main__':
    main()
