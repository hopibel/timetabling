import sqlite3
import random
import math
from timetabling import Course, WildcardCourse, Instructor

conn = sqlite3.connect('database.sqlite3')

c = conn.cursor()

# global constant. needs to match the one in timetabling.py
DAY_SLOTS = 28
SLOTS = DAY_SLOTS * 5

# populate list of degree programs
programs = ['CS', 'Bio', 'Stat']
c.executemany('INSERT or REPLACE INTO programs VALUES (?)', [(name,) for name in programs])

# populate room types
c.executemany('INSERT or REPLACE INTO room_types VALUES (?)', [(name,) for name in programs])

# generate courses and study plans
plans = {}
courses = set()
for program in programs:
    for year in range(1, 5):
        shared = set([WildcardCourse('shared-{}{}'.format(year, shared)) for shared in 'ab'])
        plans[(program, year)] = shared
        courses.update(shared)

        major = [Course('{}-{}{}'.format(program, year, major), random.choice([program, None])) for major in 'abcd']
        plans[(program, year)].update(major)
        courses.update(major)

for course in courses:
    is_wildcard = 0
    if course.is_wildcard():
        is_wildcard = 1
    c.execute('INSERT or REPLACE INTO courses VALUES (?, ?, ?)', (course.name, course.room_type, is_wildcard))

for program_year, plan in plans.items():
    program, year = program_year
    for course in plan:
        c.execute('INSERT or REPLACE INTO study_plans VALUES (?, ?, ?)', (program, year, course.name))

class_count = 0
for course in plans.values():
    class_count += 1

# generate teachers with availability of 20 timeslots per day split into morning and afternoon
faculty = []
for i in range(math.ceil(class_count / 3)):
    # randomly choose between 2 or 3 max consecutive sessions
    max_sess = random.choice((2, 3))
    c.execute('INSERT or REPLACE INTO instructors VALUES (?, ?)', (i, max_sess))

    blocks = []
    for day in range(1, 6):
        c.execute('INSERT or REPLACE INTO availability VALUES (?, ?, ?, ?)', (i, day, 600, 1100))
        c.execute('INSERT or REPLACE INTO availability VALUES (?, ?, ?, ?)', (i, day, 1200, 1700))

        day = (day - 1) * DAY_SLOTS
        blocks.append(range(day, day + 10))
        blocks.append(range(day + 12, day + 22))

    faculty.append(Instructor(name=i, avail=blocks, max_consecutive=max_sess))

conn.commit()
conn.close()
