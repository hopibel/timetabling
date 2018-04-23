"""Timetabling using a genetic algorithm"""

import sys
import pprint

import random
from collections import Counter, namedtuple
import math
import sqlite3
import numpy
from deap import algorithms, base, creator, tools
import plan_gen
import to_html

DAY_START_TIME = 600
DAY_SLOTS = 28
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
    Excess classes in a timeslot (overallocation)
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

        # TODO: take program-year size and section capacity into account
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

    # soft constraint fitness (penalty. minimize)
    soft_penalty = soft_fitness(individual, sections)

    return (overlap, overallocated, soft_penalty)


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


def main():
    """Entry point if called as executable."""

    # set up database connection
    conn = sqlite3.connect('database.sqlite3')
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
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mut_timetable,
                     sections=sections,
                     rooms=rooms,
                     faculty=faculty)
    toolbox.register('select', tools.selNSGA2, nd='log')

    ngen = 100  # generations
    mu = 100  # population size
    lambda_ = mu  # offspring to create
    cxpb = 0.7  # crossover probability
    mutpb = 0.2  # mutation probability

    pop = toolbox.population(n=mu)
    # hof = tools.ParetoFront()  # causes memory ballooning
    hof = None
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    # TODO: save state before exiting
    try:
        pop, logbook = evolve(pop, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                              stats=stats, halloffame=hof, verbose=True)
    except KeyboardInterrupt:
        pass

    solution = sel_best(pop)
    print(eval_timetable(solution, sections, program_sizes, plans))

    # TODO: replace with matplotlib graph and sqlite export
    to_html.to_html(solution, sections, SLOTS, DAY_SLOTS)


def evolve(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
           stats=None, halloffame=None, verbose=__debug__):
    """Evolve timetables with the (mu + lambda) evolutionary algorithm.
    The next generation is selected from a pool of previous population
    mu + offspring population lambda.

    Roughly based on deap.eaMuPlusLambda()

    Parameters
    ----------
    population : list of individuals
    toolbox : deap.base.Toolbox
        Contains the evolution operators.
    mu : int
        Number of individuals to select for the next generation.
    lambda : int
        Number of offspring to produce at each generation.
    cxpb : float
        Probability that an offspring is produced by crossover.
    mutpb : float
        Probability that an offspring is produced by mutation.
    ngen : int
        Number of generations.
    stats : deap.tools.Statistics, optional
        Saves statistics about the population.
    halloffame : deap.tools.HallOfFame, optional
        Will contain best individuals.
    verbose : bool
        Whether or not to log the statistics to stdout.

    Returns
    -------
    list
        The final population.
    deap.tools.Logbook
        Log of the statistics of the evolution.
    """

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Return early if an individual has fitness (0, 0, 0)
    for ind in population:
        if not any(ind.fitness.values):
            return population, logbook

    # Begin the generational process
    # Stop on perfect solution or ngen reached
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluat the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        for ind in invalid_ind:
            if not any(ind.fitness.values):
                break

    return population, logbook


def sel_best(population):
    """Select individual with lowest total hard penalty with soft
    penalty as tiebreaker.
    """
    best = population[0]
    best_penalty = sum(best.fitness.values[:-1])
    for ind in population:
        hard_penalty = sum(ind.fitness.values[:-1])
        if hard_penalty < best_penalty:
            best = ind
            best_penalty = hard_penalty
        elif hard_penalty == best_penalty:
            if ind.fitness.values[-1] < best.fitness.values[-1]:
                best = ind
                best_penalty = hard_penalty

    return best


if __name__ == '__main__':
    main()
