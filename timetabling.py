"""Timetabling using a genetic algorithm"""

import random
import numpy
from deap import algorithms, base, creator, tools

DAY_SLOTS = 28
SLOTS = DAY_SLOTS * 5


def gen_ind(courses, rooms):
    """Generate individual."""
    ind = []
    for course in courses:
        length = course['length']
        room = random.choice(rooms)
        if course['split'] and random.random() < 80:  # 80% of classes are twice a week
            if length % 2 != 0:
                raise ValueError("Splittable class '{}' has odd length".format(course['name']))

            sessions = []
            for _ in range(2):
                day = DAY_SLOTS * random.randrange(5)
                slot = day + random.randrange(DAY_SLOTS - length // 2)
                sessions.append({
                    'slot': slot,
                    'len': length // 2,
                    'room': room,
                })
            ind.append(sessions)
        else:
            day = DAY_SLOTS + random.randrange(5)
            slot = day + random.randrange(DAY_SLOTS - length)
            ind.append([
                {
                    'slot': slot,
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


def mut_timetable(ind, rooms, courses):
    """Mutate a timetable.

    Shift a class timeslot by 1
    Change classrooms
    Toggle split status
    """
    i = random.randrange(len(ind))
    course = ind[i]
    session = random.choice(course)

    def shift_slot():
        """Shift class forward or back by one time slot."""
        shift = random.choice((1, -1))

        # bounds checking
        # if moving one way goes out of bounds, move the other way
        if session['slot'] + shift < 0 or session['slot'] + session['len'] + shift > SLOTS:
            shift = -shift

        session['slot'] += shift

        # day boundary checking
        # fully move across boundary
        start = session['slot']
        end = start + session['len'] - 1
        if start // DAY_SLOTS != end // DAY_SLOTS:
            if shift == 1:
                session['slot'] = end // DAY_SLOTS * DAY_SLOTS
            else:
                session['slot'] = end // DAY_SLOTS * DAY_SLOTS - session['len']

            if session['slot'] < 0:
                print("{} {} {} {}".format(session, start, end, shift))

    def change_room():
        """Change a course's room assignment."""
        room = random.choice(rooms)
        for sess in course:
            sess['room'] = room

    def toggle_split():
        """If a course is split, unsplit it and vice versa."""
        if len(course) > 1:
            # merge other sessions into this one
            session['len'] = sum([s['len'] for s in course])
            ind[i] = [session]
        else:
            # split into two half length sessions
            session['len'] //= 2
            ind[i].append(session.copy())

    # call a random mutator
    muts = [shift_slot, change_room]
    if courses[i]['split']:
        muts.append(toggle_split)
    random.choice(muts)()

    return (ind,)


def main():
    """Entry point if called as executable."""

    random.seed('feffy')

    # dummy course table
    courses = []
    for name in range(18):
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
    toolbox.register("mutate", mut_timetable, rooms=rooms, courses=courses)
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

    print(hof[0])


if __name__ == '__main__':
    main()
