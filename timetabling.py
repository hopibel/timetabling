"""Timetabling using a genetic algorithm"""

import sys
import pprint

import copy
import random
from collections import defaultdict, Counter, namedtuple
import math
import numpy
from deap import algorithms, base, creator, tools
import plan_gen

DAY_SLOTS = 28
SLOTS = DAY_SLOTS * 5


# TODO: instructor class
class Session(object):
    """Details of a scheduled session for a course."""

    def __init__(self, course, timeslot, length, room, instructor, block):  # TODO: calculate block internally
        self.name = course['name']
        self.section = course['section']
        self.timeslot = timeslot
        self.length = length
        self.room = room
        self.instructor = instructor
        self.block = block

    def can_overlap(self, session):
        """Check whether session has the same room or instructor."""
        return self.room == session.room or self.instructor == session.instructor

    def calculate_overlap(self, session):
        """Calculate extent of overlap with another session."""
        if not self.can_overlap(session):
            return 0
        if session.timeslot >= self.timeslot + self.length:  # no time overlap
            return 0

        # overlap = minimum width required - actual width
        width = max(self.timeslot+self.length, session.timeslot+session.length) - min(self.timeslot, session.timeslot)
        return self.length + session.length - width


def gen_ind(course_table, rooms, faculty):
    """Generate individual."""
    ind = []
    for course in course_table:
        prof = course['faculty']

        length = course['length']
        room = random.choice(rooms)
        meetings = 1

        if length % 2 != 0:
            raise ValueError("Splittable class '{}' has odd length".format(course['name']))

        if course['twice_a_week']:
            length //= 2
            meetings = 2

        # indexes of availability blocks large enough to hold this class
        blocks = [i for i, r in enumerate(faculty[prof]) if r[1]-r[0]+1 >= length]

        sessions = []
        for _ in range(meetings):
            i = random.choice(blocks)
            start, end = faculty[prof][i]
            slot = random.randrange(start, end+1 - length)
            sessions.append(Session(course, slot, length, room, prof, i))
        ind.append(sessions)

    return ind


def eval_timetable(individual, course_table, class_counts, program_sizes, study_plans):
    """Calculate timetable cost.

    Currently calculates:
    Number of overlapping classes
    Excess classes in a timeslot (overallocation)
    """

    overlap = 0

    times = []
    for course in individual:
        for session in course:
            times.append(session)
    times.sort(key=lambda x: x.timeslot)

    # the time covered by two classes must be at least equal to their combined duration to avoid overlaps
    for i, _ in enumerate(times):
        a = times[i]
        for j in range(i + 1, len(times)):
            b = times[j]
            if a.can_overlap(b):
                ab_overlap = a.calculate_overlap(b)
                if ab_overlap == 0:  # b and everything afterwards don't overlap with a. break early
                    break
                else:
                    overlap += ab_overlap

    # TODO: write efficient data classes
    # A lot of list lookups happening here for static information. data structures *must* be faster than this
    overallocation = 0
    for slot in range(SLOTS):
        # find classes sharing this timeslot
        classes = []
        for course in individual:
            for session in course:
                if slot >= session.timeslot and slot < session.timeslot + session.length:
                    classes.append(session)

        # sort by restrictions to ensure most restricted are allocated first
        def count_allowed(course_name, course_table):
            """
            Count the number of groups allowed to take a course.
            Returns infinity for no restrictions
            """
            course = next(course for course in course_table if course['name'] == course_name)

            # we want classes with no restrictions to be last in the list
            if not course['restrictions']:
                return float('inf')

            allowed = 0
            for restriction in course['restrictions']:
                if restriction.year is not None:
                    allowed += 4
                else:
                    allowed += 1

            return allowed
        classes.sort(key=lambda x: count_allowed(x.name, course_table))

        # TODO: take class size/capacity into account
        program_capacities = {}
        for section in classes:
            # assume all are overallocated and subtract later
            overallocation += 1

            course = next(course for course in course_table if course['name'] == section.name)

            for restriction in course['restrictions']:
                # add only programs/years that can take this course
                if restriction.year is not None:
                    key = '{}-{}'.format(*restriction)
                    program_capacities[key] = program_sizes[key]
                else:
                    for key, count in program_sizes.items():
                        if key.startswith('{}-'.format(restriction.program)):
                            program_capacities[key] = count

            if not course['restrictions']:
                # no restrictions. add everything
                program_capacities = copy.deepcopy(program_sizes)

        for section in classes:
            # allocate program size to classes from most to least restricted. unused class capacity gets penalized
            restrictions = next(course for course in course_table if course['name'] == section.name)['restrictions']
            for restriction in restrictions:
                if restriction.year is not None:
                    restriction_name = '{}-{}'.format(restriction.program, restriction.year)
                else:
                    restriction_name = restriction.program

                if section.name.startswith(restriction_name) and program_capacities[restriction_name] > 0:
                    program_capacities[restriction_name] -= 1
                    overallocation -= 1
                    break
            else:
                for program, size in program_capacities.items():
                    if size > 0:
                        program_capacities[program] -= 1
                        overallocation -= 1
                        break

    return (overlap + overallocation,)


def mut_timetable(ind, rooms, faculty):
    """Mutate a timetable.

    Shift a class timeslot by 1
    Change classrooms
    Swap two timeslots
    """
    i = random.randrange(len(ind))
    course = ind[i]
    session = random.choice(course)

    def shift_slot():
        """Shift class forward or back by one time slot."""
        # TODO: keep classes within availability blocks
        # assume we already confirmed that availability >= load
        shift = random.choice((1, -1))

        blocks = faculty[session.instructor]

        # bounds checking
        # if moving one way goes out of bounds, move the other way
        before_first = session.timeslot + shift < 0
        after_last = session.timeslot + session.length + shift > SLOTS
        if before_first or after_last:
            shift = -shift

        def move_session():
            """Move a class session in the shift direction."""
            before_block = session.timeslot + shift < blocks[session.block][0]
            after_block = session.timeslot + session.length-1 + shift > blocks[session.block][1]

            if before_block or after_block:
                # leaving block. is there a suitable adjacent one?
                if shift > 0:
                    block_range = range(session.block+1, len(blocks))
                else:
                    block_range = range(session.block-1, -1, -1)

                for i in block_range:
                    if blocks[i][1]+1 - blocks[i][0] >= session.length:
                        if shift > 0:
                            session.timeslot = blocks[i][0]
                        else:
                            session.timeslot = blocks[i][1]+1 - session.length
                        session.block = i
                        return True
                return False
            else:
                session.timeslot += shift
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
            sess.room = room

    def swap_slot():
        """Swap a session's timeslot with another's."""
        # pick a second session
        course_b = ind[random.randrange(len(ind))]
        session_b = random.choice(course_b)
        session.timeslot, session_b.timeslot = session_b.timeslot, session.timeslot

    # call a random mutator
    muts = [shift_slot, change_room, swap_slot]
    random.choice(muts)()

    return (ind,)


def to_html(ind):
    """Convert timetable to html table."""
    rooms = defaultdict(list)
    for course in ind:
        for session in course:
            rooms[session.room].append({
                'name': "{}-{}".format(session.name, session.section),
                'start': session.timeslot,
                'end': session.timeslot + session.length - 1,
            })

    tables = {}
    for key, room in rooms.items():
        points = []  # list of (offset, plus/minus, name) tuples
        for course in room:
            points.append((course['start'], '+', course['name']))
            points.append((course['end'], '-', course['name']))
        points.sort(key=lambda x: x[1])
        points.sort(key=lambda x: x[0])

        ranges = []  # output list of (start, stop, symbol_set) tuples
        current_set = []
        last_start = None
        offset = points[0][0]
        for offset, pm, name in points:
            if pm == '+':
                if last_start is not None and current_set and offset - last_start > 0:
                    ranges.append((last_start, offset-1, current_set.copy()))
                current_set.append(name)
                last_start = offset
            elif pm == '-':
                if offset >= last_start:
                    ranges.append((last_start, offset, current_set.copy()))
                current_set.remove(name)
                last_start = offset+1

        cells = []
        last_slot = 0
        for r in ranges:  # ranges = list of (start, end, {names})
            if r[0] > last_slot:
                for i in range(last_slot, r[0]):
                    cells.append((i, i, []))
            cells.append(r)
            last_slot = r[1] + 1
        for i in range(last_slot+1, SLOTS):
            cells.append((i, i, []))

        rows = list([] for _ in range(DAY_SLOTS))
        for cell in cells:
            rows[cell[0] % DAY_SLOTS].append(cell)

        table = []
        table.append("<table>")
        table.append("""
<tr>
<th>Time</th>
<th>Monday</th>
<th>Tuesday</th>
<th>Wednesday</th>
<th>Thursday</th>
<th>Friday</th>
</tr>
"""[1:-1])
        for i, row in enumerate(rows):
            table.append("<tr>")
            table.append("<td>{}</td>".format(str(600 + 100*(i//2) + (i % 2)*30).zfill(4)))
            for cell in row:
                if cell[2] == []:
                    table.append("<td>&nbsp;</td>")
                else:
                    line = "<td"
                    if cell[1] > cell[0]:
                        line += " rowspan={}".format(cell[1] - cell[0] + 1)
                    if len(cell[2]) > 1:
                        line += " class=overlap"
                    else:
                        line += " class=course"
                    line += ">"
                    line += "<br>".join(str(x) for x in cell[2])
                    line += "</td>"
                    table.append(line)
            table.append("</tr>")
        table.append("</table>")
        tables[key] = table

    html = []
    # boilerplate
    html.append("""
<head>
<style>
table { border-collapse: collapse; }
table, th, td { border: 1px solid black; }
td { text-align: center; }
.overlap { background-color: orange; }
.course { background-color: #93c572; }
</style>
</head>
"""[1:-1])
    for name, table in tables.items():
        html.append("<b>Room {}</b><br>".format(name))
        for line in table:
            html.append(line)

    with open('room_sched.html', 'w') as outfile:
        for line in html:
            outfile.write(line + "\n")


def main():
    """Entry point if called as executable."""

    random.seed('feffy')

    # dummy study plans (only used to generate classes right now)
    programs = ['CS', 'Bio', 'Stat']
    plans = plan_gen.generate_study_plans(programs)

    program_sizes = {}
    for program, years in plans.items():
        for year in years:
            program_sizes['{}-{}'.format(program, year)] = 1

    classes = []
    for course in plans.values():
        for year in course.values():
            classes.extend(year)
    class_counts = Counter(classes)

    # dummy faculty availability
    faculty = {}
    for i in range(math.ceil(len(classes) / 3)):
        faculty[i] = []
        for day in range(5):
            day *= DAY_SLOTS
            faculty[i].append((day, day+10))
            faculty[i].append((day+12, day+25))

    # dummy course table
    course_table = []
    restriction = namedtuple('restriction', ['program', 'year'])
    faculty_assigned = 0
    for course_name in class_counts.keys():
        for section in range(1, class_counts[course_name]+1):
            # store restrictions as list of namedtuples with program and year
            # setting program or year to None acts as wildcard
            restrictions = []
            year = int(course_name[course_name.index('-')+1])
            for p in programs:
                if course_name.startswith(p):
                    restrictions.append(restriction(program=p, year=year))

            course = {
                'name': course_name,
                'section': section,
                'length': 6,
                'twice_a_week': True,
                'faculty': faculty_assigned // 3,  # three classes per teacher
                'restrictions': restrictions,
            }
            if random.random() < 0.2:
                course['twice_a_week'] = False
            course_table.append(course)
            faculty_assigned += 1

    # dummy room list
    # if a room can hold 0 classes per day, we need 1 room for every 20 classes
    # i have no idea what the 1.8 is for. probably to account for double-length classes
    rooms = tuple(range(math.ceil(len(classes) * 1.8 / 20)))

    # check if faculty have enough contiguous blocks for each class
    for name in faculty:
        avail_length = []
        for block in faculty[name]:
            avail_length.append(block[1] - block[0] + 1)
        avail_length.sort()

        class_length = []
        for course in [x for x in course_table if x['faculty'] == name]:
            if course['twice_a_week'] == 1:
                class_length.append(course['length'] // 2)
                class_length.append(course['length'] // 2)
            else:
                class_length.append(course['length'])
        class_length.sort()

        i = 0
        j = 0
        while i < len(class_length):
            if j >= len(avail_length):
                raise ValueError("Faculty member {} can't teach all their classes".format(name))

            if avail_length[j] >= class_length[i]:
                avail_length[j] -= class_length[i]
                i += 1
            else:
                j += 1

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("ind", gen_ind, course_table=course_table, rooms=rooms, faculty=faculty)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_timetable, course_table=course_table, class_counts=class_counts,
                     program_sizes=program_sizes, study_plans=plans)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mut_timetable, rooms=rooms, faculty=faculty)
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

    to_html(hof[0])


if __name__ == '__main__':
    main()
