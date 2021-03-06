import sqlite3
import random

conn = sqlite3.connect('database_small.sqlite3')

c = conn.cursor()

# populate list of degree programs
programs = [
    'Computer Science',
    'Applied Mathematics'
]
c.executemany(
    'INSERT or REPLACE INTO programs VALUES (?)',
    [(name,) for name in programs])

# populate room types
room_types = [
    'computer lab',
]
c.executemany(
    'INSERT or REPLACE INTO room_types VALUES (?)',
    [(room,) for room in room_types])

# generate courses
c.executemany(
    'INSERT or REPLACE INTO courses VALUES (?, ?, ?)',
    [
        ('Math 100', None, 0),
        ('Math 101', None, 0),
        ('Math 11A', None, 0),
        ('Math 114', None, 0),
        ('Math 121', None, 0),
        ('Math 123', None, 0),
        ('Math 131', None, 0),
        ('Math 131 lab', 'computer lab', 0),
        ('Math 14', None, 0),
        ('Math 143', 'computer lab', 0),
        ('Math 143 lab', 'computer lab', 0),
        ('Math 174', 'computer lab', 0),
        ('Math 174 lab', 'computer lab', 0),
        ('Math 178', None, 0),
        ('Math 183', None, 0),
        ('Math 184', None, 0),
        ('Math 190', None, 0),
        ('Math 20', None, 0),
        ('Math 54', None, 0),
        ('Math 55', None, 0),

        ('CMSC 11', 'computer lab', 0),
        ('CMSC 11 lab', 'computer lab', 0),
        ('CMSC 127', 'computer lab', 0),
        ('CMSC 127 lab', 'computer lab', 0),
        ('CMSC 128', 'computer lab', 0),
        ('CMSC 128 lab', 'computer lab', 0),
        ('CMSC 131', 'computer lab', 0),
        ('CMSC 131 lab', 'computer lab', 0),
        ('CMSC 137', 'computer lab', 0),
        ('CMSC 137 lab', 'computer lab', 0),
        ('CMSC 141', 'computer lab', 0),
        ('CMSC 142', 'computer lab', 0),
        ('CMSC 151', 'computer lab', 0),
        ('CMSC 162', 'computer lab', 0),
        ('CMSC 162 lab', 'computer lab', 0),
        ('CMSC 192', 'computer lab', 0),
        ('CMSC 198.1', 'computer lab', 0),
        ('CMSC 22', 'computer lab', 0),
        ('CMSC 22 lab', 'computer lab', 0),
        ('CMSC 57', 'computer lab', 0),

        ('AH 1', None, 1),
        ('AH 2', None, 1),
        ('AH 3', None, 1),

        ('SSP 1', None, 1),
        ('SSP 2', None, 1),
        ('SSP 3', None, 1),

        ('MST 1', None, 1),
        ('MST 2', None, 1),
    ]
)

# generate study plans
plans = [
    ('Computer Science', 1, 'CMSC 11'),
    ('Computer Science', 1, 'CMSC 11 lab'),
    ('Computer Science', 1, 'CMSC 127'),
    ('Computer Science', 1, 'CMSC 127 lab'),
    ('Computer Science', 1, 'CMSC 57'),
    ('Computer Science', 1, 'AH 1'),
    ('Computer Science', 1, 'SSP 1'),

    ('Computer Science', 2, 'CMSC 128'),
    ('Computer Science', 2, 'CMSC 128 lab'),
    ('Computer Science', 2, 'CMSC 131'),
    ('Computer Science', 2, 'CMSC 131 lab'),
    ('Computer Science', 2, 'CMSC 141'),
    ('Computer Science', 2, 'AH 2'),
    ('Computer Science', 2, 'MST 1'),

    ('Computer Science', 3, 'CMSC 142'),
    ('Computer Science', 3, 'CMSC 137'),
    ('Computer Science', 3, 'CMSC 137 lab'),
    ('Computer Science', 3, 'CMSC 162'),
    ('Computer Science', 3, 'CMSC 162 lab'),
    ('Computer Science', 3, 'MST 2'),
    ('Computer Science', 3, 'SSP 2'),

    ('Computer Science', 4, 'CMSC 151'),
    ('Computer Science', 4, 'CMSC 192'),
    ('Computer Science', 4, 'CMSC 198.1'),
    ('Computer Science', 4, 'CMSC 22'),
    ('Computer Science', 4, 'CMSC 22 lab'),
    ('Computer Science', 4, 'AH 3'),
    ('Computer Science', 4, 'SSP 3'),

    ('Applied Mathematics', 1, 'Math 100'),
    ('Applied Mathematics', 1, 'Math 114'),
    ('Applied Mathematics', 1, 'Math 11A'),
    ('Applied Mathematics', 1, 'Math 121'),
    ('Applied Mathematics', 1, 'AH 1'),
    ('Applied Mathematics', 1, 'SSP 1'),

    ('Applied Mathematics', 2, 'Math 123'),
    ('Applied Mathematics', 2, 'Math 131'),
    ('Applied Mathematics', 2, 'Math 131 lab'),
    ('Applied Mathematics', 2, 'Math 143'),
    ('Applied Mathematics', 2, 'Math 143 lab'),
    ('Applied Mathematics', 2, 'AH 2'),
    ('Applied Mathematics', 2, 'MST 1'),

    ('Applied Mathematics', 3, 'Math 174'),
    ('Applied Mathematics', 3, 'Math 174 lab'),
    ('Applied Mathematics', 3, 'Math 178'),
    ('Applied Mathematics', 3, 'Math 183'),
    ('Applied Mathematics', 3, 'Math 184'),
    ('Applied Mathematics', 3, 'MST 2'),
    ('Applied Mathematics', 3, 'SSP 2'),

    ('Applied Mathematics', 4, 'Math 190'),
    ('Applied Mathematics', 4, 'Math 20'),
    ('Applied Mathematics', 4, 'Math 54'),
    ('Applied Mathematics', 4, 'Math 55'),
    ('Applied Mathematics', 4, 'AH 3'),
    ('Applied Mathematics', 4, 'SSP 3'),
]
c.executemany(
    'INSERT or REPLACE INTO study_plans VALUES (?, ?, ?)', plans)

# generate program_sizes
size = 4
program_sizes = []
for program in programs:
    for year in range(1, 5):
        program_sizes.append((program, year, size))
c.executemany(
    'INSERT or REPLACE INTO program_sizes VALUES (?, ?, ?)', program_sizes)

# generate instructors with max consecutive class preference
instructors = [
    'AAEA',
    'ALT',
    'CFCC',
    'ETB',
    'ETT',
    'FJUC',
    'JCM',
    'JFJS',
    'JRRD',
    'MBSP',
    'MDP',
    'MOO',
    'NCA',
    'RCC',
    'VGN',
    'AH 1 Prof',
    'AH 2 Prof',
    'AH 3 Prof',
    'SSP 1 Prof',
    'SSP 2 Prof',
    'SSP 3 Prof',
    'MST 1 Prof',
    'MST 2 Prof',
]
rows = []
for name in instructors:
    rows.append((name, random.choice((3, 6))))
c.executemany(
    'INSERT or REPLACE INTO instructors VALUES (?, ?)', rows)

# generate instructor availability of 20 slots daily, morning and afternoon
rows = []
for name in instructors:
    for day in range(1, 6):
        rows.append((name, day, 600, 1100))
        rows.append((name, day, 1200, 1700))
c.executemany(
    'INSERT or REPLACE INTO availability VALUES (?, ?, ?, ?)', rows)

# generate sections
sections = [
    ('Math 123', '1', 'ETB', 3, 4, 1),
    ('Math 131', '1', 'ETB', 3, 4, 1),
    ('Math 11A', 'ABM', 'ETB', 3, 1, 1),
    ('Math 11A', 'ABM', 'ETB', 4, 1, 0),
    ('Math 174', '1', 'RCC', 2, 2, 1),
    ('Math 121', '2', 'RCC', 3, 2, 1),
    ('Math 174', '2', 'RCC', 2, 2, 1),
    ('Math 14', '2', 'RCC', 3, 4, 1),
    ('Math 174 lab', '1', 'RCC', 6, 2, 0),
    ('Math 174 lab', '2', 'RCC', 6, 2, 0),
    ('Math 190', '1', 'FJUC', 3, 4, 1),
    ('Math 55', '1', 'FJUC', 3, 4, 1),
    ('Math 121', '1', 'FJUC', 3, 2, 1),
    ('Math 20', '1', 'VGN', 3, 4, 1),
    ('Math 100', '2', 'VGN', 4, 1, 1),
    ('Math 100', '1', 'VGN', 4, 1, 1),
    ('Math 11A', 'STEM', 'MOO', 3, 1, 1),
    ('Math 11A', 'STEM', 'MOO', 4, 1, 0),
    ('Math 114', '1', 'MOO', 3, 2, 1),
    ('Math 100', '5', 'MOO', 4, 1, 1),
    ('Math 114', '1', 'MDP', 3, 2, 1),
    ('Math 100', '3', 'MDP', 4, 1, 1),
    ('Math 183', '1', 'MBSP', 3, 4, 1),
    ('Math 54', '1', 'MBSP', 3, 2, 1),
    ('Math 54', '1', 'MBSP', 4, 2, 0),
    ('Math 100', '4', 'MBSP', 4, 1, 1),
    ('Math 184', '1', 'ALT', 3, 4, 1),
    ('Math 178', '1', 'ALT', 3, 4, 1),
    ('CMSC 142', '1', 'AAEA', 3, 2, 1),
    ('CMSC 142', '2', 'AAEA', 3, 2, 1),
    ('Math 143', '1', 'AAEA', 2, 4, 1),
    ('Math 143 lab', '1', 'AAEA', 3, 2, 1),
    ('Math 143 lab', '2', 'AAEA', 3, 1, 1),
    ('Math 143 lab', '3', 'AAEA', 3, 1, 1),
    ('CMSC 57', '1', 'AAEA', 6, 4, 0),
    ('CMSC 198.1', '2', 'NCA', 2, 1, 1),
    ('CMSC 137', '1', 'NCA', 2, 2, 1),
    ('CMSC 198.1', '1', 'NCA', 2, 1, 1),
    ('CMSC 137', '2', 'NCA', 2, 2, 1),
    ('CMSC 11', '1', 'CFCC', 2, 4, 1),
    ('CMSC 22', '1', 'CFCC', 2, 4, 1),
    ('CMSC 151', '1', 'CFCC', 3, 2, 1),
    ('CMSC 11 lab', '1', 'CFCC', 3, 4, 1),
    ('CMSC 198.1', '4', 'CFCC', 2, 1, 1),
    ('CMSC 22 lab', '1', 'CFCC', 3, 4, 1),
    ('CMSC 151', '2', 'CFCC', 3, 2, 1),
    ('CMSC 198.1', '3', 'JRRD', 2, 1, 1),
    ('CMSC 141', '1', 'JRRD', 3, 2, 1),
    ('CMSC 127 lab', '1', 'JRRD', 3, 2, 1),
    ('CMSC 127 lab', '2', 'JRRD', 3, 1, 1),
    ('CMSC 127 lab', '3', 'JRRD', 3, 1, 1),
    ('CMSC 141', '2', 'JRRD', 3, 2, 1),
    ('CMSC 192', '1', 'JCM', 2, 2, 0),
    ('CMSC 131 lab', '1', 'JCM', 3, 2, 1),
    ('CMSC 131 lab', '2', 'JCM', 3, 1, 1),
    ('CMSC 131 lab', '3', 'JCM', 3, 1, 1),
    ('CMSC 131', '1', 'JCM', 2, 2, 1),
    ('CMSC 131', '2', 'JCM', 2, 2, 1),
    ('CMSC 192', '2', 'JCM', 2, 2, 1),
    ('CMSC 137 lab', '2', 'JCM', 6, 2, 0),
    ('CMSC 137 lab', '1', 'JCM', 6, 1, 0),
    ('CMSC 137 lab', '3', 'JCM', 6, 1, 0),
    ('CMSC 127', '2', 'JFJS', 2, 2, 1),
    ('CMSC 127', '1', 'JFJS', 2, 2, 1),
    ('CMSC 128', '1', 'JFJS', 2, 2, 1),
    ('CMSC 128', '2', 'JFJS', 2, 2, 1),
    ('CMSC 128 lab', '1', 'JFJS', 3, 2, 1),
    ('CMSC 128 lab', '2', 'JFJS', 3, 1, 1),
    ('CMSC 128 lab', '3', 'JFJS', 3, 1, 1),
    ('CMSC 162', '1', 'ETT', 2, 2, 1),
    ('CMSC 162', '2', 'ETT', 2, 2, 1),
    ('CMSC 162 lab', '2', 'ETT', 6, 2, 0),
    ('CMSC 162 lab', '3', 'ETT', 6, 1, 0),
    ('CMSC 162 lab', '4', 'ETT', 6, 1, 0),
    ('AH 1', '1', 'AH 1 Prof', 3, 4, 1),
    ('AH 1', '2', 'AH 1 Prof', 3, 4, 1),
    ('AH 2', '1', 'AH 2 Prof', 3, 4, 1),
    ('AH 2', '2', 'AH 2 Prof', 3, 4, 1),
    ('AH 3', '1', 'AH 3 Prof', 3, 4, 1),
    ('AH 3', '2', 'AH 3 Prof', 3, 4, 1),
    ('SSP 1', '1', 'SSP 1 Prof', 3, 4, 1),
    ('SSP 1', '2', 'SSP 1 Prof', 3, 4, 1),
    ('SSP 2', '1', 'SSP 2 Prof', 3, 4, 1),
    ('SSP 2', '2', 'SSP 2 Prof', 3, 4, 1),
    ('SSP 3', '1', 'SSP 3 Prof', 3, 4, 1),
    ('SSP 3', '2', 'SSP 3 Prof', 3, 4, 1),
    ('MST 1', '1', 'MST 1 Prof', 3, 4, 1),
    ('MST 1', '2', 'MST 1 Prof', 3, 4, 1),
    ('MST 2', '1', 'MST 2 Prof', 3, 4, 1),
    ('MST 2', '2', 'MST 2 Prof', 3, 4, 1),
]
c.executemany(
    'INSERT or REPLACE INTO sections VALUES (?, ?, ?, ?, ?, ?)', sections)

# generate rooms
rooms = [
    ('B6', 4, None),
    ('CL1', 4, 'computer lab'),
    ('CL2', 4, 'computer lab'),
    ('CL3', 4, 'computer lab'),
    ('CL4', 4, 'computer lab'),
    ('PA 2', 4, None),
    ('PA 3', 4, None),
    ('PF 2', 4, None),
    ('PF 3', 4, None),
    ('R104', 4, None),
    ('R111', 4, None),
    ('R203', 4, None),

    # extra rooms for GEs
    ('R996', 4, None),
    ('R997', 4, None),
    ('R998', 4, None),
    ('R999', 4, None),
]
c.executemany(
    'INSERT or REPLACE INTO rooms VALUES (?, ?, ?)', rooms)

conn.commit()

# generate restrictions after commit because we need section rowid
ge_names = set([
    'AH 1',
    'AH 2',
    'AH 3',
    'SSP 1',
    'SSP 2',
    'SSP 3',
    'MST 1',
    'MST 2',
])
for program, year, course in plans:
    # don't add restrictions to GE courses
    if course in ge_names:
        continue

    for row in c.execute('SELECT rowid FROM sections WHERE course = ?',
                         (course,)):
        c.execute(
            'INSERT or REPLACE INTO restrictions VALUES (?, ?, ?)',
            (row[0], program, year)
        )

conn.commit()

conn.close()
