"""Timetabling using a genetic algorithm"""

import random
import numpy
from deap import algorithms, base, creator, tools

SLOTS = 90


def gen_ind(courses, rooms):
    """Generate individual."""
    ind = []
    for course in courses:
        length = course['length']
        room = random.choice(rooms)
        if course['split'] and random.random() < 80:  # 80% of classes are twice a week
            ind.append([
                {
                    'slot': random.randrange(SLOTS - length // 2),
                    'len': length // 2,
                    'room': room,
                } for _ in range(2)
            ])
        else:
            ind.append([
                {
                    'slot': random.randrange(SLOTS - length),
                    'len': length,
                    'room': room,
                },
            ])
    return ind


def eval_timetable(individual):
    """Calculate timetable cost.

    Currently returns:
    Number of overlapping classes
    """
    overlap = 0

    # O(n**2). average time complexity is probably around O(nlogn)
    times = []
    for course in individual:
        for session in course:
            times.append(session)
    times.sort(key=lambda x: x['slot'])

    for i, _ in enumerate(times):
        a = times[i]
        for j in range(i + 1, len(times)):
            b = times[j]
            if a['room'] != b['room']:
                continue
            if b['slot'] >= a['slot'] + a['len']:  # b and everything after don't overlap with a
                break
            width = max(a['slot']+a['len'], b['slot']+b['len']) - min(a['slot'], b['slot'])
            if width < a['len'] + b['len']:
                overlap += a['len'] + b['len'] - width

    return (overlap,)


def mut_timetable(individual, rooms):
    """Mutate a timetable.

    Shift a class timeslot by 1
    Change classrooms
    """
    course = random.choice(individual)
    session = random.choice(course)

    x = random.random()
    if x < 0.5:
        # shift timeslot
        session['slot'] += random.choice((1, -1))
    else:
        # change room
        session['room'] = random.choice(rooms)

    # TODO: change split to single or vice versa

    # bounds checking
    # if moving one way goes out of bounds, move the other way
    if session['slot'] < 0:
        session['slot'] = 1
    elif session['slot'] + session['len'] >= SLOTS:
        session['slot'] = SLOTS - session['len'] - 1

    # day boundary checking
    # randomly put it on one side of the boundary
    start = session['slot']
    end = start + session['len'] - 1
    if start // 18 != end // 18:
        bound = int(end / 18) * 18
        session['slot'] = random.choice((bound - session['len'], bound))

    return (individual,)


def main():
    """Entry point if called as executable."""

    random.seed('feffy')

    # dummy course table
    courses = []
    for name in 'abcdefghijklmnopqr':
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

    # dummy room table
    rooms = []
    for i in range(3):
        rooms.append(i)

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("ind", gen_ind, courses=courses, rooms=rooms)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_timetable)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mut_timetable, rooms=rooms)
    toolbox.register("select", tools.selTournament, tournsize=2)

    gens = 100  # generations
    mu = 1000  # population size
    lambd = 1000  # offspring to create
    cxpb = 0.8  # crossover probability
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
