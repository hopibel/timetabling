"""Timetabling using a genetic algorithm"""

import sys
import pprint

import random
from collections import Counter, namedtuple
import math
import numpy
from deap import algorithms, base, creator, tools
import plan_gen
import to_html

DAY_SLOTS = 28
SLOTS = DAY_SLOTS * 5


Room = namedtuple('Room', ['name', 'category'])
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
    def __init__(self, course, id_, instructor, length, restrictions, is_twice_weekly):
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
        self.restrictions = restrictions
        self.is_twice_weekly = is_twice_weekly

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


# TODO: switch back to integer ID tuples. complex objects in the chromosome cause memory ballooning
# individual generation
# rewrite evaluation to use new chromosome
# TODO: rewrite mutation to use new chromosome
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
        room_id = random.choice([room_id for room_id, room in enumerate(rooms) if section.is_room_compatible(room)])

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

        # sort by most restricted
        # TODO: sort by section size then restrictions
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
        classes.sort(key=lambda x: count_allowed(sections[x.section_id]))

        # TODO: take program-year size and section capacity into account
        program_alloc = {}
        for section in classes:
            # sum section capacities and subtract allocation later. anything left is overallocation
            overallocated += 1  # TODO: replace 1 with section size, i think

            # only consider programs/years that this section is restricted to
            section_data = sections[section.section_id]
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

            for restriction in section_data.restrictions:
                if program_alloc[restriction] > 0:
                    # TODO: subtract section size instead of 1
                    program_alloc[restriction] -= 1
                    overallocated -= 1
                    break

            if not section_data.restrictions:
                for program_year, size in program_alloc.items():
                    if size > 0:
                        # TODO: subtract section size
                        program_alloc[program_year] -= 1
                        overallocated -= 1
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


# TODO: think of a better way to mutate. shifting by 1 timeslot seems to cause stagnation
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
    for instructor in faculty:
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


def main():
    """Entry point if called as executable."""

    random.seed('feffy')

    # dummy study plans (only used to generate classes right now)
    programs = ['CS', 'Bio', 'Stat']
#    programs = ['CS']
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
        blocks = set()
        for day in range(5):
            day *= DAY_SLOTS
            blocks.update(range(day, day + 10))
            blocks.update(range(day + 12, day + 22))
        # randomly choose between 2 or 3 max consecutive sessions
        faculty.append(Instructor(name=i, avail=blocks, max_consecutive=random.choice((2, 3))))

    # dummy section table. a section is an instance of a course
    sections = []
    faculty_assigned = 0
    for course in class_counts.keys():
        for section_number in range(1, class_counts[course]+1):
            # store restrictions as list of namedtuples with program and year
            # setting program or year to None acts as wildcard
            restrictions = []
            year = int(course.name[course.name.index('-')+1])
            for p in programs:
                if course.name.startswith(p):
                    restrictions.append(Restriction(program=p, year=year))

            is_twice_weekly = random.random() > 0.2
            section = Section(
                course,
                section_number,
                faculty[faculty_assigned // 3],
                3 if is_twice_weekly else 6,
                restrictions,
                is_twice_weekly)
            sections.append(section)
            faculty_assigned += 1

    # dummy room list
    rooms = []
    twice = 0
    once = 0
    for section in sections:
        if section.is_twice_weekly:
            twice += 1
        else:
            once += 1
    # i did the math, ok? don't worry about it, it's just dummy data
    rooms_needed = math.ceil(max(twice / 16, once / 2) / len(programs))
    room_number = 0
    for program in programs:
        for _ in range(rooms_needed):
            rooms.append(Room(name=room_number, category=program))
            room_number += 1

    validate_faculty_load(faculty, sections)

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("ind", gen_ind, sections=sections, rooms=rooms)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.ind)
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

    try:
        eaMuPlusLambda(
            pop, toolbox, mu, lambd, cxpb, mutpb, ngen, stats=stats, halloffame=hof, verbose=True)
    except KeyboardInterrupt:
        pass

    print(eval_timetable(hof[0], sections, program_sizes, plans))
    # final set of solutions
    # final_front = tools.sortLogNondominated(pop, 1, first_front_only=True)

    to_html.to_html(hof[0], sections, SLOTS, DAY_SLOTS)


# TODO: clean up
def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
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

    # Begin the generational process
    # Stopping criteria: perfect solution, hard penalty == 0 and best fitness unchanged for > 200 generations
    best_fitness = invalid_ind[0].fitness.values
    last_improvement = 0
    gen = 1
    while True:
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        old_best = best_fitness
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

            # Update best fitness for stopping criteria
            # Assumes minimization fitness functions
            as_good = all([new <= old for new, old in zip(fit, best_fitness)])
            strictly_better = any([new < old for new, old in zip(fit, best_fitness)])
            if as_good and strictly_better:
                best_fitness = fit

        as_good = all([new <= old for new, old in zip(best_fitness, old_best)])
        better = any([new < old for new, old in zip(best_fitness, old_best)])
        if as_good and better:
            last_improvement = 0
        else:
            last_improvement += 1

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

        # Check for stopping criteria
        if sum(best_fitness) == 0 or last_improvement > 200:
            break
        # TODO: remove logbook and print best fitness

        gen += 1

    return population, logbook


if __name__ == '__main__':
    main()
