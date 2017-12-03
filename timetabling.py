"""Timetabling using a genetic algorithm"""

import random
import numpy
from deap import algorithms, base, creator, tools

DAY_SLOTS = 28
SLOTS = DAY_SLOTS * 5


def gen_ind(courses, rooms, faculty):
    """Generate individual."""
    # assume faculty availability is enough for classes
    # choose random slot where faculty is available
    ind = []
    for course in courses:
        prof = course['faculty']

        length = course['length']
        room = random.choice(rooms)

        # TODO: do away with random splitting entirely. all split classes have 2 meetings
        # lets us determine if faculty availability >= faculty load
        # TODO: duplication. refactor
        if course['split'] and random.random() < 1:  # 80% of classes are twice a week (100% for now)
            # indexes of availability blocks large enough to hold this class
            blocks = [i for i, r in enumerate(faculty[prof]) if r[1]-r[0]+1 >= length // 2]
            if blocks == []:
                raise ValueError("{} doesn't fit in {}'s schedule".format(course['name'], prof))

            if length % 2 != 0:
                raise ValueError("Splittable class '{}' has odd length".format(course['name']))

            sessions = []
            for _ in range(2):
                # day = DAY_SLOTS * random.randrange(5)
                # slot = day + random.randrange(DAY_SLOTS - length // 2)
                i = random.choice(blocks)
                start, end = faculty[prof][i]
                slot = random.randrange(start, end+1 - length // 2)
                sessions.append({
                    'slot': slot,
                    'len': length // 2,
                    'room': room,
                    'prof': prof,
                    'block': i,
                })
            ind.append(sessions)
        else:
            # indexes of availability blocks large enough to hold this class
            blocks = [i for i, r in enumerate(faculty[prof]) if r[1]-r[0]+1 >= length]
            if blocks == []:
                raise ValueError("{} doesn't fit in {}'s schedule".format(course['name'], prof))

            # day = DAY_SLOTS + random.randrange(5)
            # slot = day + random.randrange(DAY_SLOTS - length)
            i = random.choice(blocks)
            start, end = faculty[prof][i]
            slot = random.randrange(start, end+1 - length // 2)
            ind.append([
                {
                    'slot': slot,
                    'len': length,
                    'room': room,
                    'prof': prof,
                    'block': i,
                },
            ])
    return ind


def eval_timetable(individual):
    """Calculate timetable cost.

    Currently returns:
    Number of overlapping classes
    """
    # TODO: also count faculty conflict (multiple classes at once)
    overlap = 0

    times = []
    for course in individual:
        for session in course:
            times.append(session)
    times.sort(key=lambda x: x['slot'])

    for i, _ in enumerate(times):
        a = times[i]
        for j in range(i + 1, len(times)):
            b = times[j]
            if a['room'] != b['room'] and a['prof'] != b['prof']:
                continue
            if b['slot'] >= a['slot'] + a['len']:  # b and everything after don't overlap with a
                break
            width = max(a['slot']+a['len'], b['slot']+b['len']) - min(a['slot'], b['slot'])
            if width < a['len'] + b['len']:
                overlap += a['len'] + b['len'] - width

    return (overlap,)


def mut_timetable(ind, rooms, faculty):
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
        # TODO: keep classes within availability blocks
        # assume we already confirmed that availability >= load
        shift = random.choice((1, -1))

        blocks = faculty[session['prof']]

        # bounds checking
        # if moving one way goes out of bounds, move the other way
        before_first = session['slot'] + shift < 0
        after_last = session['slot'] + session['len'] + shift > SLOTS
        if before_first or after_last:
            shift = -shift

        def move_session():
            """Move a class session in the shift direction."""
            before_block = session['slot'] + shift < blocks[session['block']][0]
            after_block = session['slot'] + session['len']-1 + shift > blocks[session['block']][1]

            if before_block or after_block:
                # leaving block. is there a suitable adjacent one?
                if shift > 0:
                    block_range = range(session['block']+1, len(blocks))
                else:
                    block_range = range(session['block']-1, -1, -1)

                for i in block_range:
                    if blocks[i][1]+1 - blocks[i][0] >= session['len']:
                        if shift > 0:
                            session['slot'] = blocks[i][0]
                        else:
                            session['slot'] = blocks[i][1]+1 - session['len']
                        session['block'] = i
                        return True
                return False
            else:
                session['slot'] += shift
            return True

        if not move_session():
            # try the other direction
            shift = -shift
            move_session()

        # day boundary checking
        # fully move across boundary
        # TODO: ensure faculty blocks don't cross boundaries
#        start = session['slot']
#        end = start + session['len'] - 1
#        if start // DAY_SLOTS != end // DAY_SLOTS:
#            if shift == 1:
#                session['slot'] = end // DAY_SLOTS * DAY_SLOTS
#            else:
#                session['slot'] = end // DAY_SLOTS * DAY_SLOTS - session['len']
#
#            if session['slot'] < 0:
#                print("{} {} {} {}".format(session, start, end, shift))

    def change_room():
        """Change a course's room assignment."""
        room = random.choice(rooms)
        for sess in course:
            sess['room'] = room

#    def toggle_split():
#        """If a course is split, unsplit it and vice versa."""
#        # NOTE: disabled for now
#        if len(course) > 1:
#            # merge other sessions into this one
#            session['len'] = sum([s['len'] for s in course])
#            ind[i] = [session]
#        else:
#            # split into two half length sessions
#            session['len'] //= 2
#            ind[i].append(session.copy())

    # call a random mutator
    muts = [shift_slot, change_room]
    # if courses[i]['split']:
    #     muts.append(toggle_split)
    random.choice(muts)()

    return (ind,)


def main():
    """Entry point if called as executable."""

    random.seed('feffy')

    # dummy faculty availability
    faculty = {}
    for i in range(9):
        faculty[i] = []
        for day in range(5):
            day *= DAY_SLOTS
            faculty[i].append((day, day+10))
            faculty[i].append((day+12, day+25))

    # dummy course table
    courses = []
    for name in range(18):
        for sec in range(1, 3):
            course = {
                'name': name,
                'section': sec,
                'length': 6,
                'split': 1,
                'faculty': name // 2,
            }
            if random.random() < 0.2:
                course['split'] = 0
            courses.append(course)

    # dummy room table
    rooms = tuple(range(3))

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("ind", gen_ind, courses=courses, rooms=rooms, faculty=faculty)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_timetable)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mut_timetable, rooms=rooms, faculty=faculty)
    toolbox.register("select", tools.selTournament, tournsize=2)

    gens = 500  # generations
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
