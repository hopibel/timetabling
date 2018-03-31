"""Timetabling using a genetic algorithm"""

import sys
import pprint

import random
from collections import Counter, namedtuple
import math
import numpy
from deap import algorithms, base, creator, tools
#import plan_gen
import to_html

DAY_SLOTS = 28
SLOTS = DAY_SLOTS * 5


Room = namedtuple('Room', ['name', 'category'])


class Instructor(object):
    """Details and availability of instructor."""
    def __init__(self, name, avail, max_consecutive):
        self.name = name
        self.avail = avail
        self.max_consecutive = max_consecutive

        # avail is always sorted and gaps are cached
        self.avail.sort(key=lambda x: x[0])

        self.gaps = {}
        for i in range(5):
            self.gaps[i] = set()

        for i in range(len(self.avail) - 1):
            a = self.avail[i]
            b = self.avail[i + 1]

            # check if same day
            if a[0] // DAY_SLOTS == b[0] // DAY_SLOTS:
                day = a[0] // DAY_SLOTS
                self.gaps[day].update(range(a[-1] + 1, b[0]))

    def valid_blocks(self, length):
        """Return indexes of availability blocks of a minimum length."""
        return [i for i, r in enumerate(self.avail) if len(r) >= length]

    def next_valid_block(self, index, length, reverse=False):
        """Return next block of given minimum length in the specified direction."""
        if not reverse:
            block_range = range(index + 1, len(self.avail))
        else:
            block_range = range(index - 1, -1, -1)

        for block_index in block_range:
            if len(self.avail[block_index]) >= length:
                return block_index

        raise ValueError("No blocks fulfill minimum length found.")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class Course(object):
    """Course object for identification purposes in study plans and restrictions."""
    def __init__(self, name=None, room_type=None):
        self.name = name
        self.room_type = room_type

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

    def is_room_compatible(self, room):
        """Return whether this course can be held in a given room."""
        if self.course.room_type is None:
            return True
        elif self.course.room_type == room.category:
            return True

        return False


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
        room = random.choice([room for room in rooms if section.is_room_compatible(room)])

        if length % 2 != 0:
            raise ValueError("Splittable class '{}' has odd length".format(section.course['name']))

        # TODO: sync meetings
        sessions = []
        if section.twice_a_week:
            length //= 2
            # indexes of availability blocks large enough to hold this class
            blocks = section.instructor.valid_blocks(length)
            first_blocks = [b for b in blocks if section.instructor.avail[b][0] < DAY_SLOTS * 2]

            # choose second blocks three days after a first block and overlap >= length
            # filter out first blocks with no corresponding second
            block_pairs = []
            for first in first_blocks:
                f = section.instructor.avail[first]
                shifted = range(f[0] + DAY_SLOTS * 3, f[-1] + 1 + DAY_SLOTS * 3)
                for b in blocks:
                    if len(set(shifted).intersection(section.instructor.avail[b])) >= length:
                        block_pairs.append((first, b))
                        break

            pair = random.choice(block_pairs)
            first = section.instructor.avail[pair[0]]
            second = section.instructor.avail[pair[1]]

            # pick a random block and schedule the second session at the same time 3 days later
            start = max(first[0], second[0] - DAY_SLOTS * 3)
            end = min(first[-1], second[-1] - DAY_SLOTS * 3)
            slot = random.randrange(start, end + 1 - length)
            sessions.append(SessionSchedule(section, slot, length, room, pair[0]))
            sessions.append(SessionSchedule(section, slot + DAY_SLOTS * 3, length, room, pair[1]))
        else:
            # single session scheduling. no sync needed
            blocks = section.instructor.valid_blocks(length)

            i = random.choice(blocks)
            start = section.instructor.avail[i][0]
            end = section.instructor.avail[i][-1]
            slot = random.randrange(start, end+1 - length)
            sessions.append(SessionSchedule(section, slot, length, room, i))

        ind.append(sessions)

    return ind


# treat all GEs and electives as one category. GEs are numerous enough that there will usually be an available class.
# electives are often course specific so they're covered by restrictions already
def eval_timetable(individual, course_table, program_sizes, study_plans):
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

    # soft constraint fitness
    soft_score = soft_fitness(individual)

    return (overlap + overallocation, soft_score)


def soft_fitness(individual):
    """
    Calculate soft constraint fitness penalties.
    Value is a single sum for simplicity.

    Current constraints:
    Minimize unused timeslots between classes per instructor.
    Minimize runs of consecutive classes that are longer than instructor's preference
    """
    # group sessions by instructor
    instructors = {}
    for course in individual:
        if course[0].section.instructor not in instructors:
            instructors[course[0].section.instructor] = []
        for section in course:
            instructors[section.section.instructor].append(section)

    # count gaps between classes occurring on the same day
    # count consecutive classes, adding 1 to penalty if it goes above the instructor's preference
    gap_length = 0
    marathon_penalty = 0
    for instructor, sections in instructors.items():
        # sort by timeslot
        sections.sort(key=lambda x: x.timeslot)

        consecutive = 1
        for i in range(len(sections) - 1):
            a = sections[i]
            b = sections[i + 1]

            # check if same day
            # ignore gap if it is present in availability schedule already
            if a.timeslot // DAY_SLOTS == b.timeslot // DAY_SLOTS:
                gap = range(a.timeslot + a.length, b.timeslot)
                day = a.timeslot // DAY_SLOTS

                gap_length += len(gap)
                gap_length -= len(instructor.gaps[day].intersection(gap))

                # count consecutive
                if a.timeslot + a.length == b.timeslot:
                    consecutive += 1
                    if consecutive > instructor.max_consecutive:
                        marathon_penalty += 1
                else:
                    consecutive = 1
            else:
                # reset consecutive count between days
                consecutive = 1

    return gap_length + marathon_penalty


def mut_timetable(ind, rooms, faculty):
    """Mutate a timetable.

    Shift a class timeslot by 1
    Change classrooms
    Swap two timeslots
    """
    i = random.randrange(len(ind))
    course = ind[i]

    def shift_slot():
        """Shift class forward or back by one time slot."""
        # TODO: keep classes within availability blocks
        # assume we already confirmed that availability >= load
        shift = random.choice((1, -1))

        blocks = course[0].section.instructor.avail

        # bounds checking
        # if moving one way goes out of bounds, move the other way
        before_first = False
        after_last = False
        for sess in course:
            before_first = before_first or sess.timeslot + shift < 0
            after_last = after_last or sess.timeslot + sess.length + shift > SLOTS

        if before_first or after_last:
            shift = -shift

        def move_session(shift):
            """
            Move a class session in the shift direction.
            Throws ValueError when session cannot be shifted.
            """
            before_block = False
            after_block = False
            for sess in course:
                before_block = before_block or sess.timeslot + shift < blocks[sess.block][0]
                after_block = after_block or sess.timeslot + sess.length - 1 + shift > blocks[sess.block][-1]

            instructor = course[0].section.instructor
            length = course[0].section.length
            assert len(course) == 1 or instructor == course[1].section.instructor
            assert len(course) == 1 or length == course[1].section.length
            if before_block or after_block:
                # leaving block. is there a suitable adjacent one?
                reverse = shift < 0

                if len(course) > 1:  # 2 sessions
                    block_a = instructor.next_valid_block(course[0].block, length, reverse)
                    block_b = instructor.next_valid_block(course[1].block, length, reverse)

                    first_block = blocks[block_a]
                    second_block = blocks[block_b]

                    # advance blocks until a valid pair is found for MTh or TF
                    while True:
                        shifted = range(second_block[0] - DAY_SLOTS * 3, second_block[-1] + 1 - DAY_SLOTS * 3)
                        if len(set(shifted).intersection(first_block)) < length:
                            if not reverse:
                                shift_first = first_block[0] < shifted[0]
                            else:
                                shift_first = first_block[-1] >= shifted[-1]

                            if shift_first:
                                block_a = instructor.next_valid_block(block_a, length, reverse)
                                first_block = blocks[block_a]
                            else:
                                block_b = instructor.next_valid_block(block_b, length, reverse)
                                second_block = blocks[block_b]
                        else:
                            break

                    # assign new blocks and timeslots
                    course[0].block = block_a
                    course[1].block = block_b

                    # get correct slot for direction
                    if not reverse:
                        slot = max(blocks[block_a][0], blocks[block_b][0] - DAY_SLOTS * 3)
                    else:
                        slot = min(blocks[block_a][-1], blocks[block_b][-1] - DAY_SLOTS * 3)
                        slot = slot + 1 - length

                    course[0].timeslot = slot
                    course[1].timeslot = slot + DAY_SLOTS * 3
                else:  # 1 session
                    sess = course[0]
                    sess.block = instructor.next_valid_block(sess.block, length, reverse)

                    if not reverse:  # forward
                        sess.timeslot = blocks[sess.block][0]
                    else:  # backward
                        sess.timeslot = blocks[sess.block][-1] + 1 - length
            else:  # not leaving block. just shift the timeslots
                for sess in course:
                    sess.timeslot += shift

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
        room = random.choice([room for room in rooms if course[0].section.is_room_compatible(room)])
        for sess in course:
            sess.room = room

    # call a random mutator
    muts = [shift_slot, change_room]
    random.choice(muts)()

    return (ind,)


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

    # generate teachers with availability of 20 timeslots per day split into morning and afternoon
    faculty = []
    for i in range(math.ceil(len(classes) / 3)):
        blocks = []
        for day in range(5):
            day *= DAY_SLOTS
            blocks.append(range(day, day + 10))
            blocks.append(range(day + 12, day + 25))
        # randomly choose between 2 or 3 max consecutive sessions
        faculty.append(Instructor(name=i, avail=blocks, max_consecutive=random.choice((2, 3))))

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
    # if a room can hold 6 classes per day, we need 1 room for every 30 classes
    # i have no idea what the 1.8 is for. probably to account for double-length classes
    # rooms = tuple(range(math.ceil(len(classes) * 1.8 / 30)))
    rooms = []
    rooms_needed = math.ceil(len(classes) / 30 / len(programs))
    room_number = 0
    for program in programs:
        for _ in range(rooms_needed):
            rooms.append(Room(name=room_number, category=program))

    # check if faculty have enough contiguous blocks for each class
    for instructor in faculty:
        avail_length = []
        for block in instructor.avail:
            avail_length.append(len(block))
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

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))  # minimize hard and soft constraint violations
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("ind", gen_ind, course_table=course_table, rooms=rooms, faculty=faculty)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_timetable,
                     course_table=course_table, program_sizes=program_sizes, study_plans=plans)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mut_timetable, rooms=rooms, faculty=faculty)
    toolbox.register('select', tools.selNSGA2)

    gens = 100  # generations
    mu = 100  # population size
    lambd = mu  # offspring to create
    cxpb = 0.7  # crossover probability
    mutpb = 0.2  # mutation probability

    pop = toolbox.population(n=mu)
    hof = tools.ParetoFront()
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    algorithms.eaMuPlusLambda(
        pop, toolbox, mu, lambd, cxpb, mutpb, gens, stats=stats, halloffame=hof, verbose=True)

    to_html.to_html(hof[0], SLOTS, DAY_SLOTS)


if __name__ == '__main__':
    main()
