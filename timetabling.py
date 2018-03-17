"""Timetabling using a genetic algorithm"""

import sys
import pprint

import random
from collections import defaultdict, Counter, namedtuple
import math
import numpy
from deap import algorithms, base, creator, tools
import plan_gen

DAY_SLOTS = 28
SLOTS = DAY_SLOTS * 5


class Instructor(object):
    """Details and availability of instructor."""
    def __init__(self, name, avail):
        self.name = name
        self.avail = avail

    def valid_blocks(self, length):
        """Return indexes of availability blocks of a minimum length."""
        return [i for i, r in enumerate(self.avail) if r[1]-r[0]+1 >= length]

    def next_valid_block(self, index, length, reverse=False):
        """Return next block of given minimum length in the specified direction."""
        if not reverse:
            block_range = range(index + 1, len(self.avail))
        else:
            block_range = range(index - 1, -1, -1)

        for block_index in block_range:
            if self.avail[block_index][1]+1 - self.avail[block_index][0] >= length:
                return block_index

        raise ValueError("No blocks fulfill minimum length found.")


class Course(object):
    """Course object for identification purposes in study plans and restrictions."""
    def __init__(self, name=None):
        self.name = name

    def matches(self, course):
        """Check whether this course fulfills the given course requirement."""
        return self.name == course.name

    @staticmethod
    def is_wildcard():
        """Return whether this a wildcard."""
        return False

    def __hash__(self):
        return hash((self.name, self.is_wildcard()))

    def __eq__(self, other):
        return self.name == other.name and self.is_wildcard() == other.is_wildcard()


class WildcardCourse(Course):
    """
    Wildcard course for GEs and Electives. Matches any GE or elective in a study plan.

    All GE types and Electives are treated as one category.
    GEs are numerous enough that mutual exclusion is unlikely.
    Electives often have restrictions which take effect before wildcards.
    """
    def matches(self, course):
        return course.is_wildcard() or self.name == course.name

    @staticmethod
    def is_wildcard():
        return True


class Section(object):
    """Specific section from a Course used to build course table and schedule."""
    def __init__(self, course, id_, instructor, length, restrictions, twice_a_week):
        """
        course (Course): Course that this Section belongs to
        id_ (hashable): section ID
        instructor (Instructor): Instructor details
        twice_a_week (bool): whether the section is split into two weekly sessions
        """
        self.course = course
        self.id_ = id_
        self.instructor = instructor
        self.length = length
        self.restrictions = restrictions
        self.twice_a_week = twice_a_week


class SessionSchedule(object):
    """Details of a scheduled session for a course."""

    def __init__(self, section, timeslot, length, room, block):  # TODO: calculate block internally
        self.section = section
        self.timeslot = timeslot
        self.length = length
        self.room = room
        self.block = block

    def can_overlap(self, session):
        """Check whether session has the same room or instructor."""
        return self.room == session.room or self.section.instructor == session.section.instructor

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
    for section in course_table:
        length = section.length
        room = random.choice(rooms)
        meetings = 1

        if length % 2 != 0:
            raise ValueError("Splittable class '{}' has odd length".format(section.course['name']))

        # TODO: sync meetings
        if section.twice_a_week:
            length //= 2
            meetings = 2

        # indexes of availability blocks large enough to hold this class
        blocks = section.instructor.valid_blocks(length)

        sessions = []
        for _ in range(meetings):
            i = random.choice(blocks)
            start, end = section.instructor.avail[i]
            slot = random.randrange(start, end+1 - length)
            sessions.append(SessionSchedule(section, slot, length, room, i))
        ind.append(sessions)

    return ind


# TODO: apparently i'm not using class_counts and study_plans at all. uh oh
# treat all GEs and electives as one category. GEs are numerous enough that there will usually be an available class.
# electives are often course specific so they're covered by restrictions already
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
            course = next(course for course in course_table if course.course.name == course_name)

            # we want classes with no restrictions to be last in the list
            if not course.restrictions:
                return float('inf')

            allowed = 0
            for restriction in course.restrictions:
                if restriction.year is not None:
                    allowed += 4
                else:
                    allowed += 1

            return allowed
        classes.sort(key=lambda x: count_allowed(x.section.course.name, course_table))

        # TODO: take class size/capacity into account
        program_capacities = {}
        for schedule in classes:
            # assume all are overallocated and subtract later
            overallocation += 1

            # add programs/years that this course is restricted to
            for restriction in schedule.section.restrictions:
                if restriction.year is not None:
                    program_year = (restriction.program, restriction.year)
                    program_capacities[program_year] = program_sizes[program_year]
                else:
                    for program_year, count in program_sizes.items():
                        if program_year[0] == restriction.program:
                            program_capacities[program_year] = count

            # no restrictions. add all programs/years with this course in their study plan
            if not schedule.section.restrictions:
                for program_year, plan in study_plans.items():
                    for course in plan:
                        if schedule.section.course.matches(course):
                            program_capacities[program_year] = program_sizes[program_year]
                            break

        for schedule in classes:
            # allocate program size to classes from most to least restricted. unused class capacity gets penalized
            for restriction in schedule.section.restrictions:
                if program_capacities[restriction] > 0:
                    program_capacities[restriction] -= 1
                    overallocation -= 1
                    break

            if not schedule.section.restrictions:
                for program_year, size in program_capacities.items():
                    if size > 0:
                        program_capacities[program_year] -= 1
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

        blocks = session.section.instructor.avail

        # bounds checking
        # if moving one way goes out of bounds, move the other way
        before_first = session.timeslot + shift < 0
        after_last = session.timeslot + session.length + shift > SLOTS
        if before_first or after_last:
            shift = -shift

        def move_session(shift):
            """Move a class session in the shift direction."""
            before_block = session.timeslot + shift < blocks[session.block][0]
            after_block = session.timeslot + session.length-1 + shift > blocks[session.block][1]

            if before_block or after_block:
                # leaving block. is there a suitable adjacent one?
                if shift > 0:
                    session.block = session.section.instructor.next_valid_block(session.block, session.length)
                    session.timeslot = blocks[session.block][0]
                else:
                    session.block = session.section.instructor.next_valid_block(session.block, session.length, True)
                    session.timeslot = blocks[session.block][1] + 1 - session.length
            else:
                session.timeslot += shift

        # try to shift in the given direction, otherwise try the opposite
        try:
            move_session(shift)
        except ValueError:
            try:
                move_session(-shift)
            except ValueError:
                pass

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
                'name': "{}-{}".format(session.section.course.name, session.section.id_),
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
    programs = ['CS', 'Bio']
    plans = plan_gen.generate_study_plans(programs)

    program_sizes = {}
    for program_year in plans:
        program_sizes[program_year] = 1

    classes = []
    for course in plans.values():
        classes.extend(course)
    class_counts = Counter(classes)

    # dummy faculty availability
    faculty = {}
    for i in range(math.ceil(len(classes) / 3)):
        faculty[i] = []
        for day in range(5):
            day *= DAY_SLOTS
            faculty[i].append((day, day+10))
            faculty[i].append((day+12, day+25))

    # generate teachers with availability of 20 timeslots per day split into morning and afternoon
    # TODO: sanity check that no availability crosses a day boundary
    # possibly implemented as a db constraint rather than here
    faculty = []
    for i in range(math.ceil(len(classes) / 3)):
        blocks = []
        for day in range(5):
            day *= DAY_SLOTS
            blocks.append((day, day + 10))
            blocks.append((day + 12, day + 25))
        faculty.append(Instructor(name=i, avail=blocks))

    # dummy course table
    course_table = []
    restriction = namedtuple('restriction', ['program', 'year'])
    faculty_assigned = 0
    for course in class_counts.keys():
        for section_number in range(1, class_counts[course]+1):
            # store restrictions as list of namedtuples with program and year
            # setting program or year to None acts as wildcard
            restrictions = []
            year = int(course.name[course.name.index('-')+1])
            for p in programs:
                if course.name.startswith(p):
                    restrictions.append(restriction(program=p, year=year))

            twice_a_week = random.random() > 0.2
            section = Section(course, section_number, faculty[faculty_assigned // 3], 6, restrictions, twice_a_week)
            course_table.append(section)
            faculty_assigned += 1

    # dummy room list
    # if a room can hold 4 classes per day, we need 1 room for every 20 classes
    # i have no idea what the 1.8 is for. probably to account for double-length classes
    rooms = tuple(range(math.ceil(len(classes) * 1.8 / 20)))

    # check if faculty have enough contiguous blocks for each class
    for instructor in faculty:
        avail_length = []
        for block in instructor.avail:
            avail_length.append(block[1] - block[0] + 1)
        avail_length.sort()

        class_length = []
        for course in [x for x in course_table if x.instructor.name == instructor.name]:
            if course.twice_a_week:
                class_length.append(course.length // 2)
                class_length.append(course.length // 2)
            else:
                class_length.append(course.length)
        class_length.sort()

        i = 0
        j = 0
        while i < len(class_length):
            if j >= len(avail_length):
                raise ValueError("Faculty member {} can't teach all their classes".format(instructor.name))

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

    gens = 150  # generations
    mu = 300  # population size
    lambd = mu  # offspring to create
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
