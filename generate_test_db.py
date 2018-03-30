import sqlite3
import random
from timetabling import Course, WildcardCourse

# names of degree programs to generate
programs = ['CS', 'Bio', 'Stat']

conn = sqlite3.connect('database.sqlite3')

c = conn.cursor()

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

conn.commit()
conn.close()
