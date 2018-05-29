"""Timetabling using a genetic algorithm"""

import sys
import os
import signal
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

    return (hard_penalty, soft_penalty)


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

        consecutive = 0
        for i in range(len(meetings) - 1):
            a = meetings[i]
            b = meetings[i + 1]

            a_data = sections[a.section_id]
            b_data = sections[b.section_id]

            # initialize consecutive
            if i == 0:
                consecutive += a_data.length

            # check if same day
            # ignore gap if it is from the instructor's availability schedule
            if a.slot // DAY_SLOTS == b.slot // DAY_SLOTS:
                gap = range(a.slot + a_data.length, b.slot)
                instructor_avail_gap = len(instructor.gaps.intersection(gap))

                old_gap = gap_length
                gap_length += len(gap) - instructor_avail_gap

                # count_consecutive
                if a.slot + a_data.length == b.slot:
                    consecutive += b_data.length
                    if consecutive > instructor.max_consecutive:
                        consecutive_penalty += consecutive - instructor.max_consecutive
                else:
                    # if gap prevents consecutive_penalty, reduce gap penalty
                    if consecutive == instructor.max_consecutive and len(gap) > instructor_avail_gap:
                        gap_length -= 1
                    consecutive = 0
                assert gap_length - old_gap >= 0
            else:
                # reset consecutive count between days
                consecutive = 0

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


def slot_to_day_time(slot):
    """
    Calculate day and time based on time slot.
    """
    day = slot // DAY_SLOTS
    hour = slot % DAY_SLOTS
    time = DAY_START_TIME
    time += hour // 2 * 100
    time += hour % 2 * 30
    return day, time


def plot_results(gen, runs, outdir):
    # average results of all runs
    for stat in runs.keys():
        runs[stat] = numpy.mean(runs[stat], axis=0)

    # plot final averaged results
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.plot(gen, numpy.array(runs['min'])[:, 0], 'g-', label='Minimum')
    ax1.plot(gen, numpy.array(runs['avg'])[:, 0], 'b-', label='Average')
    ax1.plot(gen, numpy.array(runs['max'])[:, 0], 'r-', label='Maximum')
    ax1.set_title('Hard constraints')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_ylim(ymin=0)
    ax1.legend()

    ax2.plot(gen, numpy.array(runs['min'])[:, 1], 'g-', label='Minimum')
    ax2.plot(gen, numpy.array(runs['avg'])[:, 1], 'b-', label='Average')
    ax2.plot(gen, numpy.array(runs['max'])[:, 1], 'r-', label='Maximum')
    ax2.set_title('Soft constraints')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness')
    ax2.set_ylim(ymin=0)
    ax2.legend()

    fig1.savefig(os.path.join(outdir, 'fitness_hard.png'), dpi=600)
    fig2.savefig(os.path.join(outdir, 'fitness_soft.png'), dpi=600)


def export_to_image(best, sections, rooms, outdir):
    """
    Save schedules in tabular form to outdir
    """
    os.makedirs(outdir, exist_ok=True)

    # group by instructor and room
    by_instructor = {}
    by_room = {}
    for section in best:
        prof = sections[section.section_id].instructor
        if prof not in by_instructor:
            by_instructor[prof] = []
        by_instructor[prof].append(section)

        if section.room not in by_room:
            by_room[section.room] = []
        by_room[section.room].append(section)

    day_start = DAY_START_TIME
    day_end = day_start + DAY_SLOTS // 2 * 100 + (DAY_SLOTS % 2) * 30
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # generate instructor schedules
    # based on http://masudakoji.github.io/2015/05/23/generate-timetable-using-matplotlib/en/
    for prof in by_instructor.keys():
        fig, ax = plt.subplots(figsize=(12, 12))
        for section in by_instructor[prof]:
            section_data = sections[section.section_id]
            day, time = slot_to_day_time(section.slot)
            day += 1  # 1-based indexing
            day -= 0.5
            hour = time // 100
            minute = time % 100
            start = hour * 100 + minute / 60 * 100
            _, end = slot_to_day_time(section.slot + section_data.length)
            end = end // 100 * 100 + (end % 100) / 60 * 100

            # plot classes
            ax.fill_between(
                [day, day+1], [start, start], [end, end],
                facecolor='lightgreen', edgecolor='k', linewidth=2, alpha=0.7)
            # show time in top left corner
            ax.text(day+0.05, start+10, '{0}:{1:0>2}'.format(hour, minute),
                    va='top', fontsize=8)
            # show class and room name
            text = "{} - {}\n{}".format(section_data.course.name, section_data.id_, rooms[section.room].name)
            ax.text(day+0.5, (start+end)*0.5, text,
                    ha='center', va='center', fontsize=11)

            if section.is_twice_weekly:
                day += 3
                # plot classes
                ax.fill_between(
                    [day, day+1], [start, start], [end, end],
                    facecolor='lightgreen', edgecolor='k', linewidth=2, alpha=0.7)
                # show time in top left corner
                ax.text(day+0.05, start+10, '{0}:{1:0>2}'.format(hour, minute),
                        va='top', fontsize=8)
                # show class and room name
                text = "{} - {}\n{}".format(section_data.course.name, section_data.id_, rooms[section.room].name)
                ax.text(day+0.5, (start+end)*0.5, text,
                        ha='center', va='center', fontsize=11)

        # offset title to avoid getting covered by day labels
        ax.set_title('Instructor: {}'.format(prof.name), y=1.07)

        ax.yaxis.grid()
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(day_end, day_start)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(days)
        ax.set_ylabel('Time')
        ax.set_yticks(range(day_start, day_end, 50))
        time_labels = []
        for time in range(day_start, day_end, 100):
            hour = time // 100
            time_labels.append('{0}:{1:0>2}'.format(hour, 0))
            time_labels.append('{0}:{1:0>2}'.format(hour, 30))
        ax.set_yticklabels(time_labels)

        # show labels on both sides
        ax2 = ax.twiny().twinx()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylim(ax.get_ylim())
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels(days)
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(time_labels)

        filename = os.path.join(outdir, 'prof_{}.png'.format(prof.name.replace(' ', '_')))
        print("Writing {}".format(filename))
        fig.savefig(filename)
        plt.close(fig)

    # generate room schedules
    for room_id in by_room:
        room = rooms[room_id]
        fig, ax = plt.subplots(figsize=(12, 12))
        for section in by_room[room_id]:
            section_data = sections[section.section_id]
            day, time = slot_to_day_time(section.slot)
            day += 1  # 1-based indexing
            day -= 0.5
            hour = time // 100
            minute = time % 100
            start = hour * 100 + minute / 60 * 100
            _, end = slot_to_day_time(section.slot + section_data.length)
            end = end // 100 * 100 + (end % 100) / 60 * 100

            # plot classes
            ax.fill_between(
                [day, day+1], [start, start], [end, end],
                facecolor='lightgreen', edgecolor='k', linewidth=2, alpha=0.7)
            # show time in top left corner
            ax.text(day+0.05, start+10, '{0}:{1:0>2}'.format(hour, minute),
                    va='top', fontsize=8)
            # show class and instructor
            text = "{} - {}\n{}".format(section_data.course.name, section_data.id_, section_data.instructor.name)
            ax.text(day+0.5, (start+end)*0.5, text,
                    ha='center', va='center', fontsize=11)

            if section.is_twice_weekly:
                day += 3
                # plot classes
                ax.fill_between(
                    [day, day+1], [start, start], [end, end],
                    facecolor='lightgreen', edgecolor='k', linewidth=2, alpha=0.7)
                # show time in top left corner
                ax.text(day+0.05, start+10, '{0}:{1:0>2}'.format(hour, minute),
                        va='top', fontsize=8)
                # show class and room name
                text = "{} - {}\n{}".format(section_data.course.name, section_data.id_, section_data.instructor.name)
                ax.text(day+0.5, (start+end)*0.5, text,
                        ha='center', va='center', fontsize=11)

        # offset title to avoid getting covered by day labels
        ax.set_title('Room: {}'.format(room.name), y=1.07)

        ax.yaxis.grid()
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(day_end, day_start)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(days)
        ax.set_ylabel('Time')
        ax.set_yticks(range(day_start, day_end, 50))
        time_labels = []
        for time in range(day_start, day_end, 100):
            hour = time // 100
            time_labels.append('{0}:{1:0>2}'.format(hour, 0))
            time_labels.append('{0}:{1:0>2}'.format(hour, 30))
        ax.set_yticklabels(time_labels)

        # show labels on both sides
        ax2 = ax.twiny().twinx()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylim(ax.get_ylim())
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels(days)
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(time_labels)

        filename = os.path.join(outdir, 'room_{}.png'.format(room.name.replace(' ', '_')))
        print("Writing {}".format(filename))
        fig.savefig(filename)
        plt.close(fig)


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
    parser.add_argument("--pop", help="population size per deme", type=int, default=100)
    parser.add_argument("--gens", help="number of generations. If not provided, run until optimal timetable found", type=int)
    parser.add_argument("--fast", help="If set, terminate on the first feasible timetable",
                        action='store_true')
    parser.add_argument("--crossover", help="crossover probability", type=float, default=0.6)
    parser.add_argument("--mutation", help="mutation probability", type=float, default=0.4)
    parser.add_argument("-r", "--runs", help="average results over multiple runs",
                        type=int, default=1)
    parser.add_argument("-c", "--resume", help="result from last checkpoint",
                        action='store_true')
    parser.add_argument("database",
                        help="sqlite3 database to use",
                        type=str)
    parser.add_argument("-o", "--outdir", help="output directory name",
                        default="output")
    parser.add_argument("-v", "--verbose", help="log per-generation stats to console",
                        action='store_true')

    return parser.parse_args()


def main():
    """Entry point if called as executable."""

    args = parse_args()

    # suppress errors when not in debug mode
    def exit_msg():
        print("Received ctrl-c")
        sys.exit()
    signal.signal(signal.SIGINT, lambda signal_number, stack_frame: exit_msg())

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
        # also convert max_consecutive hours to time slots
        faculty[name] = Instructor(name, availabilities[name], max_consecutive * 2)

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

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
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

    # create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    start_run = 1
    runs = {
        'min': [],
        'avg': [],
        'max': [],
    }
    halloffame = tools.HallOfFame(maxsize=1)

    # resume previous run if applicable
    checkpoint = os.path.join(args.outdir, 'runs_cp.pkl')
    if args.resume:
        try:
            with open(checkpoint, 'rb') as cp_file:
                cp = pickle.load(cp_file)
                start_run = cp['start_run']
                runs = cp['runs']
                halloffame = cp['halloffame']
                gens = cp['gens']

            # discard results if number of generations changed
            # can't collect stats on different length runs or endless ones
            if args.gens != gens or args.gens is None:
                runs = {
                    'min': [],
                    'avg': [],
                    'max': [],
                }
            # start_run = min(start_run, args.runs)
        except FileNotFoundError:
            print("No checkpoint found. Starting a new run")
            args.resume = False

    with open(checkpoint, 'wb') as cp_file:
        cp = dict(
            start_run=start_run,
            runs=runs,
            halloffame=halloffame,
            gens=args.gens
        )
        pickle.dump(cp, cp_file)

    # perform multiple runs and average the results
    for run in range(start_run, args.runs+1):
        print("Started run {}/{}".format(run, args.runs))

        # endless runs if args.gens is None
        # resume runs in increments of 100 generations until 0 conflicts
        current_best = (float('inf'), float('inf'))
        while True:
            # set up migration pipes in ring topology
            pipes = [Pipe(False) for _ in range(NBR_DEMES)]
            pipes_in = deque(p[0] for p in pipes)
            pipes_out = deque(p[1] for p in pipes)
            pipes_in.rotate(1)

            e = Event()
            out_queue = Queue()

            processes = [
                Process(target=mp_evolve,
                        args=(args, toolbox, i, ipipe, opipe, e, out_queue,
                              random.random(), args.verbose))
                for i, (ipipe, opipe)
                in enumerate(zip(pipes_in, pipes_out))
            ]

            for proc in processes:
                proc.start()

            # prevent succeeding runs from trying to resume with old checkpoint
            args.resume = False

            results = []
            for i in range(NBR_DEMES):
                results.append(out_queue.get())

            for proc in processes:
                proc.join()

            print("Collecting run {} statistics".format(run))

            hard_min = []
            hard_avg = []
            hard_max = []

            soft_min = []
            soft_avg = []
            soft_max = []

            for r in results:
                log = r['logbook']
                hard_min.append(numpy.array(log.select('min'))[:, 0])
                hard_avg.append(numpy.array(log.select('avg'))[:, 0])
                hard_max.append(numpy.array(log.select('max'))[:, 0])

                soft_min.append(numpy.array(log.select('min'))[:, 1])
                soft_avg.append(numpy.array(log.select('avg'))[:, 1])
                soft_max.append(numpy.array(log.select('max'))[:, 1])

                # collect hall of fame members
                halloffame.update(r['halloffame'])

            # combine results from each deme
            hard_min = numpy.min(hard_min, axis=0)
            hard_avg = numpy.mean(hard_avg, axis=0)
            hard_max = numpy.max(hard_max, axis=0)

            soft_min = numpy.min(soft_min, axis=0)
            soft_avg = numpy.mean(soft_avg, axis=0)
            soft_max = numpy.max(soft_max, axis=0)

            # save run results. append to last run if endless
            if args.gens is None and len(runs['min']) > 0:
                runs['min'][0] = numpy.column_stack((hard_min, soft_min))
                runs['avg'][0] = numpy.column_stack((hard_avg, soft_avg))
                runs['max'][0] = numpy.column_stack((hard_max, soft_max))
            else:
                runs['min'].append(numpy.column_stack((hard_min, soft_min)))
                runs['avg'].append(numpy.column_stack((hard_avg, soft_avg)))
                runs['max'].append(numpy.column_stack((hard_max, soft_max)))

            # checkpoint run results since this'll take a while
            with open(os.path.join(args.outdir, 'runs_cp.pkl'), 'wb') as cp_file:
                cp = dict(
                    start_run=run,
                    runs=runs,
                    halloffame=halloffame,
                    gens=args.gens
                )
                pickle.dump(cp, cp_file)

            if args.gens is not None:
                break
            else:
                fit = halloffame[0].fitness.values
                if fit[0] == 0 and (args.fast or fit[1] == 0 or not fit < current_best):
                    break
                else:
                    # resume evolution until next multiple of 100 generations
                    args.resume = True
                    current_best = fit

        # only one run if endless
        if args.gens is None:
            break

    # average and plot results
    gen = list(range(1, len(runs['min'][0]) + 1))
    plot_results(gen, runs, args.outdir)

    # get best solution, minimizing hard penalty first
    # best = min(halloffame, key=lambda ind: tuple(ind.fitness.values[1:]))
    best = halloffame[0]
    print("Fittest timetable found: {}".format(toolbox.evaluate(best)))

    # export best solution to an sqlite3 db
    print("Saving timetable to {}".format(os.path.join(args.outdir, 'final_timetable.sqlite3')))
    conn = sqlite3.connect(os.path.join(args.outdir, 'final_timetable.sqlite3'))
    c = conn.cursor()
    c.execute("""
    DROP TABLE IF EXISTS timetable
    """)
    c.execute("""
    CREATE TABLE timetable(
        instructor text NOT NULL,
        course text NOT NULL,
        section text NOT NULL,
        schedule text NOT NULL,
        start int NOT NULL,
        end int NOT NULL,
        room text NOT NULL
    )""")

    # add each section to the database
    rows = []
    for section in best:
        section_data = sections[section.section_id]

        # convert slot to days and times
        day, start = slot_to_day_time(section.slot)
        length = section_data.length
        _, end = slot_to_day_time(section.slot + length)
        if section.is_twice_weekly:
            schedule = ('MTh', 'TF')[day]
        else:
            schedule = ('M', 'T', 'W', 'Th', 'F')[day]

        row = (
            section_data.instructor.name,
            section_data.course.name,
            section_data.id_,
            schedule,
            start,
            end,
            rooms[section.room].name,
        )
        rows.append(row)
    c.executemany(
        'INSERT OR REPLACE INTO timetable VALUES (?, ?, ?, ?, ?, ?, ?)', rows)

    conn.commit()
    conn.close()

    # export to images
    image_output = os.path.join(args.outdir, 'img')
    print("Exporting schedules to {}".format(image_output))
    export_to_image(best, sections, rooms, image_output)


def mp_evolve(args, toolbox, procid, pipe_in, pipe_out, sync, out_queue, seed=None,
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

    NGEN = args.gens    # number of generations
    MU = args.pop       # population size
    LAMBDA = MU         # number of offspring to generate each gen
    CXPB = args.crossover  # 0.6          # crossover probability
    MUTPB = args.mutation  # 0.4         # mutation probability
    MIG_RATE = 5        # migration rate (generations)
    MIG_K = 5           # number of individuals migrated
    RR_THRESH = 50      # random restart if no improvement/stagnated
    RR_KEEP = 5         # keep the best individuals during a restart
    CHKPOINT = 10       # checkpoint frequency
    LOG_RATE = 10       # hall of fame log frequency

    # set up migration pipes
    toolbox.register("migrate", mig_pipe, k=MIG_K, pipe_in=pipe_in,
                     pipe_out=pipe_out, selection=tools.selBest,
                     replacement=random.sample)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    checkpoint = os.path.join(args.outdir, 'deme{}_cp.pkl'.format(procid))
    if args.resume:
        # resume with checkpoint in resume_prefix folder
        with open(checkpoint, 'rb') as cp_file:
            cp = pickle.load(cp_file)
        deme = cp['population']
        start_gen = cp['generation']
        hof = cp['halloffame']
        logbook = cp['logbook']
        random.setstate(cp['rngstate'])
    else:
        # start a new evolution
        deme = toolbox.population(MU)
        start_gen = 1
        random.seed(seed)
        hof = tools.HallOfFame(maxsize=1)
        logbook = tools.Logbook()
        logbook.header = ('gen', 'deme', 'nevals', 'std', 'min', 'avg', 'max')

        # initial checkpoint
        cp = dict(
            population=deme,
            generation=start_gen,
            halloffame=hof,
            logbook=logbook,
            rngstate=random.getstate()
        )
        with open(checkpoint, 'wb') as cp_file:
            pickle.dump(cp, cp_file)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in deme if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(deme) if stats is not None else {}
        logbook.record(gen=start_gen, deme=procid, nevals=len(invalid_ind), **record)
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

    # update NGEN if endless
    if NGEN is None:
        NGEN = start_gen // 100 * 100 + 100

    # Begin the generational process
    for gen in range(start_gen + 1, NGEN + 1):
        # Select the next generation population
        offspring = toolbox.select(deme, len(deme))

        # Vary the population
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            # # feasible solutions take priority over infeasible
            # if fit[1] == 0:
            #     fit = list(fit)
            #     fit[0] = 0
            #     fit = tuple(fit)
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
        # NOTE: disabled. didn't seem to help much
        if False and gen % RR_THRESH == 0:
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
                    ## feasible solutions take priority over infeasible
                    #if fit[1] == 0:
                    #    fit = list(fit)
                    #    fit[0] = 0
                    #    fit = tuple(fit)
                    ind.fitness.values = fit

                if hof is not None:
                    hof.update(new_pop)

                # update population
                deme[:] = new_pop + elite

        # log current hall of fame individual regularly
        if gen % LOG_RATE == 0:
            print("Gen {} Deme {} best: {}".format(gen, procid, hof[0].fitness.values))

        # checkpoint
        if gen % CHKPOINT == 0 or gen == NGEN:
            cp = dict(
                population=deme,
                generation=gen,
                halloffame=hof,
                logbook=logbook,
                rngstate=random.getstate()
            )
            with open(checkpoint, 'wb') as cp_file:
                pickle.dump(cp, cp_file)

    result = {
        'logbook': logbook,
        'population': deme,
        'halloffame': hof,
    }
    out_queue.put(result)


if __name__ == '__main__':
    main()
