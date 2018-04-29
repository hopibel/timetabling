"""Timetabling using a genetic algorithm"""

import sys
import pprint

import os
import argparse
import random
import sqlite3
import pickle
import numpy
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from multiprocessing import Event, Pipe, Process, Queue

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import to_html

DAY_START_TIME = 0
DAY_SLOTS = 48
SLOTS = DAY_SLOTS * 5

Room = namedtuple('Room', ['name', 'capacity', 'category'])
Restriction = namedtuple('Restriction', ['program', 'year'])


class SectionGene(object):
    __slots__ = ('section_id', 'slot', 'room', 'is_twice_weekly')

    def __init__(self, section_id, slot, room, is_twice_weekly):
        self.section_id = section_id
        self.slot = slot
        self.room = room
        self.is_twice_weekly = is_twice_weekly

    def copy(self):
        return SectionGene(self.section_id, self.slot, self.room, self.is_twice_weekly)


class Instructor(object):
    """Details and availability of instructor."""
    def __init__(self, name, avail, max_consecutive):
        self.name = name
        self.avail = avail
        self.max_consecutive = max_consecutive
        self.gaps = set()

        # cache gaps
        lavail = list(avail)
        lavail.sort()
        for i in range(len(lavail) - 1):
            same_day = lavail[i] // DAY_SLOTS == lavail[i+1] // DAY_SLOTS
            consecutive = lavail[i] + 1 == lavail[i + 1]
            if same_day and not consecutive:
                self.gaps.update(range(lavail[i]+1, lavail[i+1]))

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
    def is_nonmajor():
        """Return whether this a nonmajor course."""
        return False

    def __hash__(self):
        return hash((self.name, self.is_nonmajor()))

    def __eq__(self, other):
        return self.name == other.name and self.is_nonmajor() == other.is_nonmajor()


class NonmajorCourse(Course):
    """
    Nonmajor course for GEs and Electives. Matches any GE or elective in a study plan.

    All GE types and Electives are treated as one category.
    GEs are numerous enough that mutual exclusion is unlikely.
    Electives often have restrictions which take effect before Nonmajors.
    """
    def matches(self, course):
        return course.is_nonmajor() or self.name == course.name

    @staticmethod
    def is_nonmajor():
        return True


class Section(object):
    """Specific section from a Course used to build course table and schedule."""
    def __init__(self, course, id_, instructor, length, size, restrictions, is_twice_weekly):
        """
        course (Course): Course that this Section belongs to
        id_ (hashable): section ID
        instructor (Instructor): Instructor details
        is_twice_weekly (bool): whether the section is split into two weekly sessions
        """
        self.course = course
        self.id_ = id_
        self.instructor = instructor
        self.length = length
        self.size = size
        self.restrictions = restrictions
        self.is_twice_weekly = is_twice_weekly

    def is_room_compatible(self, room):
        """Return whether this course can be held in a given room."""
        large_enough = room.capacity >= self.size
        right_type = (self.course.room_type is None
                      or self.course.room_type == room.category)

        return large_enough and right_type


def gen_ind(sections, rooms):
    """
    Generate individual.

    Chromosome is a list of tuples with 4 values:
    [(C, S, R, T), ...]

    C = section id
    S = timeslot (half hour offsets from beginning of the day)
    R = room id
    T = twice a week flag
    """
    ind = []
    for section_id, section in enumerate(sections):
        room_id = random.choice([
            room_id
            for room_id, room
            in enumerate(rooms) if section.is_room_compatible(room)])

        # valid timeslots where this section can be scheduled
        available = section.instructor.avail
        valid_slots = []
        for slot in available:
            # ensure we don't cross day boundaries
            same_day = slot // DAY_SLOTS == (slot + section.length - 1) // DAY_SLOTS
            slot_is_valid = set(range(slot, slot + section.length)).issubset(available) and same_day

            if section.is_twice_weekly:
                meeting2 = slot + DAY_SLOTS * 3
                same_day = meeting2 // DAY_SLOTS == (meeting2 + section.length - 1) // DAY_SLOTS
                meeting2_is_valid = set(range(meeting2, meeting2 + section.length)).issubset(available) and same_day
            else:
                meeting2_is_valid = True

            if slot_is_valid and meeting2_is_valid:
                valid_slots.append(slot)

        slot = random.choice(valid_slots)
        ind.append(SectionGene(section_id, slot, room_id, section.is_twice_weekly))
    return ind


# treat all GEs and electives as one category. GEs are numerous enough that there will usually be an available class.
# electives are often course specific so they're covered by restrictions already
def eval_timetable(individual, sections, program_sizes, study_plans):
    """Calculate timetable cost.

    Currently calculates:
    Number of overlapping classes
    Excess classes in a timeslot (overallocation/density)

    To be compared in lexicographical order. First value is total penalties.
    Second value is hard penalty, so in case of a tie the individual with lower
    hard penalty is preferred.
    """
    overlap = 0

    times = []  # list of (section_id, timeslot, room_id)
    for section in individual:
        times.append(section)
        if section.is_twice_weekly:
            # append second meeting
            second_meeting = section.copy()
            second_meeting.slot += DAY_SLOTS * 3
            times.append(second_meeting)
    times.sort(key=lambda x: x.slot)

    # total slots covered by two classes must be >= total length of both classes to avoid overlap
    for i in range(len(times) - 1):
        a = times[i]
        a_section = sections[a.section_id]
        for j in range(i + 1, len(times)):
            b = times[j]
            b_section = sections[b.section_id]

            # check for overlap if a and b share a room or instructor
            if a.room == b.room or a_section.instructor == b_section.instructor:
                if b.slot >= a.slot + a_section.length:
                    break  # no overlap from here onward since times are sorted

                # overlap = min required width - actual width
                width = (max(a.slot + a_section.length,
                            b.slot + b_section.length)
                         - min(a.slot, b.slot))
                overlap += a_section.length + b_section.length - width

    overallocated = 0
    for slot in range(SLOTS):
        # find classes sharing this timeslot, cover both meetings
        classes = []
        for section in individual:
            section_data = sections[section.section_id]
            start = section.slot
            end = start + section_data.length

            if slot in range(start, end):
                classes.append(section)
            if section.is_twice_weekly:
                start += DAY_SLOTS * 3
                end += DAY_SLOTS * 3
                if slot in range(start, end):
                    classes.append(section)

        # sort by section size then by most restricted
        def count_allowed(section):
            """
            Count the number of groups allowed to take a course.
            Returns infinity for no restrictions
            """
            # unrestricted should be allocated last
            if not section.restrictions:
                return float('inf')

            allowed = 0
            for restriction in section.restrictions:
                # no year restriction means all years of a course can take
                if restriction.year is None:
                    allowed += 4
                else:
                    allowed += 1
            return allowed
        classes.sort(key=lambda x: sections[x.section_id].size)
        classes.sort(key=lambda x: count_allowed(sections[x.section_id]))

        program_alloc = {}
        for section in classes:
            section_data = sections[section.section_id]

            # sum section capacities and subtract allocation later. anything left is overallocation
            overallocated += section_data.size

            # only consider programs/years that this section is restricted to
            for restriction in section_data.restrictions:
                if restriction.year is None:
                    for program_year, count in program_sizes.items():
                        if program_year[0] == restriction.program:
                            program_alloc[program_year] = count
                else:
                    program_alloc[tuple(restriction)] = program_sizes[tuple(restriction)]

            # if unrestricted, add all programs/years whose study plan contains the course
            if not section_data.restrictions:
                for program_year, plan in study_plans.items():
                    for requirement in plan:
                        if section_data.course.matches(requirement):
                            program_alloc[program_year] = program_sizes[program_year]
                            break

        # allocate program size to classes from most to least restricted
        # unused class capacity is overallocation
        for section in classes:
            section_data = sections[section.section_id]

            unallocated = section_data.size
            for restriction in section_data.restrictions:
                if program_alloc[restriction] > 0:
                    if program_alloc[restriction] <= unallocated:
                        unallocated -= program_alloc[restriction]
                        overallocated -= program_alloc[restriction]
                        program_alloc[restriction] = 0
                    else:
                        program_alloc[restriction] -= unallocated
                        overallocated -= unallocated
                        break

            if not section_data.restrictions:
                for program_year, size in program_alloc.items():
                    if size > 0:
                        if size <= unallocated:
                            unallocated -= size
                            overallocated -= size
                            program_alloc[program_year] = 0
                        else:
                            program_alloc[program_year] -= unallocated
                            overallocated -= unallocated
                            break

    # total hard constraint fitness
    hard_penalty = overlap + overallocated

    # soft constraint fitness (penalty. minimize)
    soft_penalty = soft_fitness(individual, sections)

    return (hard_penalty + soft_penalty, hard_penalty, soft_penalty)


def soft_fitness(individual, sections):
    """
    Calculate soft constraint fitness penalties.
    Value is a single sum for simplicity.

    Current constraints:
    Minimize unused timeslots between classes per instructor.
    Minimize runs of consecutive classes that are longer than instructor's preference
    """
    # group sections by instructor
    instructors = {}
    for section in individual:
        section_data = sections[section.section_id]

        if section_data.instructor not in instructors:
            instructors[section_data.instructor] = []
            instructors[section_data.instructor].append(section)
        if section.is_twice_weekly:
            second_meeting = section.copy()
            second_meeting.slot += DAY_SLOTS * 3
            instructors[section_data.instructor].append(second_meeting)

    # count gaps between classes held on the same day
    # count consecutive classes, penalize if above instructor's preference
    gap_length = 0
    consecutive_penalty = 0
    for instructor, meetings in instructors.items():
        # sort by timeslot
        meetings.sort(key=lambda x: x.slot)

        consecutive = 1
        for i in range(len(meetings) - 1):
            a = meetings[i]
            b = meetings[i + 1]

            a_data = sections[a.section_id]

            # check if same day
            # ignore gap if it is from the instructor's availability schedule
            if a.slot // DAY_SLOTS == b.slot // DAY_SLOTS:
                gap = range(a.slot + a_data.length, b.slot)

                gap_length += len(gap)
                gap_length -= len(instructor.gaps.intersection(gap))

                # count_consecutive
                if a.slot + a_data.length == b.slot:
                    consecutive += 1
                    if consecutive > instructor.max_consecutive:
                        consecutive_penalty += 1
                else:
                    consecutive = 1
            else:
                # reset consecutive count between days
                consecutive = 1

    return gap_length + consecutive_penalty


def mut_timetable(ind, sections, rooms, faculty):
    """Mutate a timetable.

    Shift a class timeslot by 1, small chance of completely random slot.
    Change classrooms
    """
    section = random.choice(ind)
    section_data = sections[section.section_id]

    def get_valid_slots():
        """
        Get valid timeslots for mutation.
        """
        # assume we already confirmed that availability >= load

        # valid timeslots where this section can be scheduled
        available = section_data.instructor.avail
        valid_slots = []
        for slot in available:
            # ensure we don't cross day boundaries
            same_day = slot // DAY_SLOTS == (slot + section_data.length - 1) // DAY_SLOTS
            slot_is_valid = set(range(slot, slot + section_data.length)).issubset(available) and same_day

            if section.is_twice_weekly:
                meeting2 = slot + DAY_SLOTS * 3
                same_day = meeting2 // DAY_SLOTS == (meeting2 + section_data.length - 1) // DAY_SLOTS
                meeting2_is_valid = set(range(meeting2, meeting2 + section_data.length)).issubset(available) and same_day
            else:
                meeting2_is_valid = True

            if slot_is_valid and meeting2_is_valid:
                valid_slots.append(slot)
        return valid_slots

    def shift_slot():
        """
        Shift class forward or back by one time slot.
        """
        valid_slots = get_valid_slots()

        slot_index = valid_slots.index(section.slot)
        if slot_index == 0:
            slot = valid_slots[1]
        elif slot_index == len(valid_slots) - 1:
            slot = valid_slots[-2]
        else:
            slot = valid_slots[random.choice((slot_index + 1, slot_index - 1))]

        section.slot = slot

    def random_slot():
        """
        Move a class to a completely random timeslot.
        Used to avoid stagnation towards the end of a run.
        """
        valid_slots = get_valid_slots()
        section.slot = random.choice(valid_slots)
        change_room()

    def change_room():
        """Change a course's room assignment."""
        section.room = random.choice([room_id for room_id, room in enumerate(rooms)
                                      if section_data.is_room_compatible(room)])

    # call a random mutator with the given weights
    mutators = [shift_slot, random_slot, change_room]
    weights = [0.45, 0.1, 0.45]
    numpy.random.choice(mutators, p=weights)()

    return (ind,)


def validate_faculty_load(faculty, sections):
    # check if faculty have enough contiguous blocks for each class
    for instructor in faculty.values():
        avail_length = []
        block_list = list(instructor.avail)
        block_list.sort()

        # get lengths of contiguous availability slots
        length = 1
        for i in range(1, len(block_list)):
            if block_list[i] > block_list[i - 1] + 1:
                avail_length.append(length)
                length = 1
            else:
                length += 1
        if length > 1:
            avail_length.append(length)

        avail_length.sort()

        class_length = []
        for section in [s for s in sections if s.instructor == instructor]:
            if section.is_twice_weekly:
                class_length.extend([section.length] * 2)
            else:
                class_length.append(section.length)
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


def day_time_to_slot(day, time):
    """
    Calculate slot based on integer representing 24 hour time format.
    """
    slot = (day - 1) * DAY_SLOTS
    slot += (time - DAY_START_TIME) // 100 * 2
    slot += (time % 100) // 30
    return slot


def mig_pipe(deme, k, pipe_in, pipe_out, selection, replacement=None):
    """Migration using pipes between processes. It first selects
    *k* individuals from the *deme* and writes them in *pipe_out*. Then it
    reads the individuals from *pipe_in* and replaces some individuals in
    the deme. The *replacement* function must sample without repetition.

    Parameters
    ----------
    deme : list of individuals
    k : int
        Number of individuals to migrate.
    pipe_in : multiprocessing.Pipe
        Pipe from which to read immigrants.
    pipe_out : multiprocessing.Pipe
        Pipe in which to write emigrants.
    selection : function
        Function to use for selecting emigrants.
    replacement : function
        Function to select individuals to replace with immigrants.
        If set to None immigrants directly replace emigrants.
    """
    emigrants = selection(deme, k)
    if replacement is None:
        # If no replacement strategy is selected, replace those who migrate
        immigrants = emigrants
    else:
        # otherwise select those who will be replaced
        immigrants = replacement(deme, k)

    pipe_out.send(emigrants)
    buf = pipe_in.recv()

    for place, immigrant in zip(immigrants, buf):
        i = deme.index(place)
        deme[i] = immigrant


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pop", help="population size per deme", type=int)
    parser.add_argument("gens", help="number of generations", type=int)
    parser.add_argument("-r", "--runs", help="average results over multiple runs",
                        type=int, default=1)
    parser.add_argument("-c", "--continue", help="result from last checkpoint",
                        action='store_true')
    parser.add_argument("-d", "--database",
                        help="sqlite3 database to use",
                        type=str, default='database.sqlite3')
    parser.add_argument("-o", "--outdir", help="output directory name",
                        default="output")
    parser.add_argument("-v", "--verbose", help="log per-generation stats to console",
                        action='store_true')

    return parser.parse_args()


def main():
    """Entry point if called as executable."""

    args = parse_args()

    # set up database connection
    conn = sqlite3.connect(args.database)
    c = conn.cursor()

    # import courses
    courses = {}
    for name, room_type, is_nonmajor in c.execute('select * from courses'):
        if is_nonmajor:
            courses[name] = NonmajorCourse(name, room_type)
        else:
            courses[name] = Course(name, room_type)

    # import study plans
    plans = {}
    for program, year, course in c.execute('select * from study_plans'):
        if (program, year) not in plans:
            plans[(program, year)] = set()
        plans[(program, year)].add(courses[course])

    # import program sizes
    program_sizes = {}
    for program, year, size in c.execute('select * from program_sizes'):
        program_sizes[(program, year)] = size

    # import instructor availability times
    availabilities = {}
    for instructor, day, start, end in c.execute('select * from availability'):
        if instructor not in availabilities:
            availabilities[instructor] = set()

        availabilities[instructor].update(range(
            day_time_to_slot(day, start),
            day_time_to_slot(day, end),
        ))

    # import instructors
    faculty = {}
    for name, max_consecutive in c.execute('select * from instructors'):
        faculty[name] = Instructor(name, availabilities[name], max_consecutive)

    # import restrictions
    restrictions = {}
    for section, program, year in c.execute('select * from restrictions'):
        if section not in restrictions:
            restrictions[section] = []
        restrictions[section].append(Restriction(program, year))

    # import sections
    sections = []
    query = """
        select
            rowid,
            course,
            section_id,
            instructor,
            length,
            size,
            is_twice_weekly
        from sections
        """
    for row in c.execute(query):
        rowid, course, id_, instructor, length, size, is_twice_weekly = row
        sections.append(Section(
            courses[course],
            id_,
            faculty[instructor],
            length,
            size,
            restrictions.get(rowid, []),
            is_twice_weekly
        ))

    rooms = []
    for name, capacity, category in c.execute('select * from rooms'):
        rooms.append(Room(name, capacity, category))

    validate_faculty_load(faculty, sections)

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("ind", gen_ind, sections=sections, rooms=rooms)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_timetable,
                     sections=sections,
                     program_sizes=program_sizes,
                     study_plans=plans)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mut_timetable,
                     sections=sections,
                     rooms=rooms,
                     faculty=faculty)
    toolbox.register('select', tools.selTournament, tournsize=2)

    # number of processes to run in parallel
    NBR_DEMES = 4

    # create output directory
    os.makedirs(args.outdir)

    # perform multiple runs and average the results
    runs = {
        'min': [],
        'avg': [],
        'max': [],
    }
    for run in range(1, args.runs+1):
        print("Started run {}/{}".format(run, args.runs))

        # set up migration pipes in ring topology
        pipes = [Pipe(False) for _ in range(NBR_DEMES)]
        pipes_in = deque(p[0] for p in pipes)
        pipes_out = deque(p[1] for p in pipes)
        pipes_in.rotate(1)
        pipes_out.rotate(-1)

        e = Event()
        out_queue = Queue()

        processes = [
            Process(target=mp_evolve,
                    args=(toolbox.population(args.pop), args.gens, toolbox,
                          i, ipipe, opipe, e, out_queue, random.random(), args.verbose))
            for i, (ipipe, opipe)
            in enumerate(zip(pipes_in, pipes_out))
        ]

        for proc in processes:
            proc.start()

        results = []
        for i in range(NBR_DEMES):
            results.append(out_queue.get())

        for proc in processes:
            proc.join()

        print("Collecting run {} statistics".format(run))

        fit_min = []
        fit_avg = []
        fit_max = []

        hard_min = []
        hard_avg = []
        hard_max = []

        soft_min = []
        soft_avg = []
        soft_max = []
        for r in results:
            log = r['logbook']
            fit_min.append(numpy.array(log.select('min'))[:, 0])
            fit_avg.append(numpy.array(log.select('avg'))[:, 0])
            fit_max.append(numpy.array(log.select('max'))[:, 0])

            hard_min.append(numpy.array(log.select('min'))[:, 1])
            hard_avg.append(numpy.array(log.select('avg'))[:, 1])
            hard_max.append(numpy.array(log.select('max'))[:, 1])

            soft_min.append(numpy.array(log.select('min'))[:, 2])
            soft_avg.append(numpy.array(log.select('avg'))[:, 2])
            soft_max.append(numpy.array(log.select('max'))[:, 2])

        # combine results from each deme
        fit_min = numpy.min(fit_min, axis=0)
        fit_avg = numpy.mean(fit_avg, axis=0)
        fit_max = numpy.max(fit_max, axis=0)

        hard_min = numpy.min(hard_min, axis=0)
        hard_avg = numpy.mean(hard_avg, axis=0)
        hard_max = numpy.max(hard_max, axis=0)

        soft_min = numpy.min(soft_min, axis=0)
        soft_avg = numpy.mean(soft_avg, axis=0)
        soft_max = numpy.max(soft_max, axis=0)

        # save run results
        runs['min'].append(numpy.column_stack((fit_min, hard_min, soft_min)))
        runs['avg'].append(numpy.column_stack((fit_avg, hard_avg, soft_avg)))
        runs['max'].append(numpy.column_stack((fit_max, hard_max, soft_max)))

        # checkpoint run results since this'll take a while
        with open(os.path.join(args.outdir, 'run.pkl'), 'wb') as cp_file:
            pickle.dump(runs, cp_file)

    # average results of all runs
    for stat in runs.keys():
        runs[stat] = numpy.mean(runs[stat], axis=0)

    gen = results[0]['logbook'].select('gen')

    # plot final averaged results
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    ax1.plot(gen, numpy.array(runs['min'])[:, 0], 'g-', label='Minimum')
    ax1.plot(gen, numpy.array(runs['avg'])[:, 0], 'b-', label='Average')
    ax1.plot(gen, numpy.array(runs['max'])[:, 0], 'r-', label='Maximum')
    ax1.set_title('Hard + Soft')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_ylim(ymin=0)
    ax1.legend()

    ax2.plot(gen, numpy.array(runs['min'])[:, 1], 'g-', label='Minimum')
    ax2.plot(gen, numpy.array(runs['avg'])[:, 1], 'b-', label='Average')
    ax2.plot(gen, numpy.array(runs['max'])[:, 1], 'r-', label='Maximum')
    ax2.set_title('Hard')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness')
    ax2.set_ylim(ymin=0)
    ax2.legend()

    ax3.plot(gen, numpy.array(runs['min'])[:, 2], 'g-', label='Minimum')
    ax3.plot(gen, numpy.array(runs['avg'])[:, 2], 'b-', label='Average')
    ax3.plot(gen, numpy.array(runs['max'])[:, 2], 'r-', label='Maximum')
    ax3.set_title('Soft')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Fitness')
    ax3.set_ylim(ymin=0)
    ax3.legend()

    fig1.savefig(os.path.join(args.outdir, '{}-both.png'.format(args.outdir)))
    fig2.savefig(os.path.join(args.outdir, '{}-hard.png'.format(args.outdir)))
    fig3.savefig(os.path.join(args.outdir, '{}-soft.png'.format(args.outdir)))

    # # TODO: save state before exiting
    # # TODO: replace with matplotlib graph and sqlite export
    # to_html.to_html(solution, sections, SLOTS, DAY_SLOTS)


def mp_evolve(pop, ngen, toolbox, procid, pipe_in, pipe_out, sync, out_queue, seed=None,
              verbose=__debug__):
    """Evolve timetables with the (mu + lambda) evolutionary algorithm.
    The next generation is selected from a pool of previous population
    mu + offspring population lambda.

    Roughly based on deap.eaMuPlusLambda()

    Parameters
    ----------
    toolbox : deap.base.Toolbox
        Contains the evolution operators.
    procid : int
        process ID
    pipe_in : multiprocessing.Pipe
        Pipe for immigrants.
    pip_out : multiprocessing.Pipe
        Pipe for emigrants.
    sync : multiprocessing.Event
        Synchronization channel.
    out_queue: multiprocessing.Queue
        Queue for final hall of fame, population, and logbook output.
    seed : random seed
    verbose : bool
        Whether or not to log the statistics to stdout.
    """

    NGEN = ngen         # number of generations
    MU = len(pop)       # population size
    LAMBDA = MU         # number of offspring to generate each gen
    CXPB = 0.6          # crossover probability
    MUTPB = 0.3         # mutation probability
    MIG_RATE = 5        # migration rate (generations)
    MIG_K = 5           # number of individuals migrated
    RR_THRESH = 20      # random restart if no improvement/stagnated
    RR_KEEP = 5         # keep the best individuals during a restart
    CHKPOINT = 50       # checkpoint frequency
    LOG_RATE = 25       # hall of fame log frequency

    # start each deme with a different seed
    random.seed(seed)
    toolbox.register("migrate", mig_pipe, k=MIG_K, pipe_in=pipe_in,
                     pipe_out=pipe_out, selection=tools.selBest,
                     replacement=random.sample)

    deme = pop
    hof = tools.HallOfFame(maxsize=MU)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    # TODO: pickle and checkpoint regularly
    # TODO: save logbooks until we have matplotlib output
    # TODO: argparse optional --continue
    logbook = tools.Logbook()
    logbook.header = ('gen', 'deme', 'nevals', 'std', 'min', 'avg', 'max')

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in deme if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(deme) if stats is not None else {}
    logbook.record(gen=0, deme=procid, nevals=len(invalid_ind), **record)
    if hof is not None:
        hof.update(deme)

    if verbose:
        if procid == 0:
            # Synchronization needed to log header on top exactly once
            print(logbook.stream)
            sync.set()
        else:
            logbook.log_header = False  # never output the header
            sync.wait()
            print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN + 1):
        # Select the next generation population
        offspring = toolbox.select(deme, len(deme))

        # Vary the population
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if hof is not None:
            hof.update(offspring)

        deme[:] = offspring

        # Update the statistics with the new population
        record = stats.compile(deme) if stats else {}
        logbook.record(gen=gen, deme=procid, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # perform migration every MIG_RATE generations
        if gen % MIG_RATE == 0:
            toolbox.migrate(deme)

        # random restart if best solution hasn't improved (stagnation)
        if gen % RR_THRESH == 0:
            current_avg = tuple(logbook[-1]['avg'])
            old_avg = tuple(logbook[-RR_THRESH]['avg'])

            if not current_avg < old_avg:
                print("restarting {}".format(procid))

                # save some of current population and generate a new one
                elite = tools.selBest(deme, RR_KEEP)
                new_pop = toolbox.population(n=MU-len(elite))

                # evaluate fitnesses of new population
                invalid_ind = [ind for ind in new_pop if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                if hof is not None:
                    hof.update(new_pop)

                # update population
                deme[:] = new_pop + elite

        # log current hall of fame individual regularly
        if gen % LOG_RATE == 0:
            print("Deme {} best: {}".format(procid, hof[0].fitness.values))

    result = {
        'logbook': logbook,
        'population': deme,
        'halloffame': hof,
    }
    out_queue.put(result)


if __name__ == '__main__':
    main()

    # possible improvements:
    # focus mutation on genes/sections involved in conflicts
