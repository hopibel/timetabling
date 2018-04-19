import sqlite3

conn = sqlite3.connect('database.sqlite3')

c = conn.cursor()

# global constant. needs to match the one in timetabling.py
# days are 28 half hour slots starting at 0600
DAY_SLOTS = 28
SLOTS = DAY_SLOTS * 5

# populate list of degree programs
programs = [
    'Physics',
    'Computer Science',
    'Statistics',
    'Applied Mathematics'
]
c.executemany(
    'INSERT or REPLACE INTO programs VALUES (?)',
    [(name,) for name in programs])

# populate room types
room_types = [
    'computer lab',
    'physics lab',
]
c.executemany(
    'INSERT or REPLACE INTO room_types VALUES (?)',
    [(room,) for room in room_types])

# generate courses
c.executemany(
    'INSERT or REPLACE INTO courses VALUES (?, ?)',
    [
        ('CMSC 11', 0),
        ('CMSC 11 lab', 0),
        ('CMSC 127', 0),
        ('CMSC 127 lab', 0),
        ('CMSC 128', 0),
        ('CMSC 128 lab', 0),
        ('CMSC 131', 0),
        ('CMSC 131 lab', 0),
        ('CMSC 137', 0),
        ('CMSC 137 lab', 0),
        ('CMSC 141', 0),
        ('CMSC 142', 0),
        ('CMSC 151', 0),
        ('CMSC 162', 0),
        ('CMSC 162 lab', 0),
        ('CMSC 192', 0),
        ('CMSC 198.1', 0),
        ('CMSC 22', 0),
        ('CMSC 22 lab', 0),
        ('CMSC 57', 0),
        ('Math 100', 0),
        ('Math 101', 0),
        ('Math 114', 0),
        ('Math 11A', 0),
        ('Math 121', 0),
        ('Math 123', 0),
        ('Math 131', 0),
        ('Math 14', 0),
        ('Math 143', 0),
        ('Math 143 lab', 0),
        ('Math 174', 0),
        ('Math 174 lab', 0),
        ('Math 178', 0),
        ('Math 183', 0),
        ('Math 184', 0),
        ('Math 190', 0),
        ('Math 20', 0),
        ('Math 54', 0),
        ('Math 55', 0),
        ('Physical Science 11', 0),
        ('Physics 21', 0),
        ('Physics 51', 0),
        ('Physics 51.1', 0),
        ('Physics 72', 0),
        ('Physics 72.1', 0),
        ('Physics 74', 0),
        ('Physics 74 lab', 0),
        ('Physics 76', 0),
        ('Physics 76 lab', 0),
        ('Stat 104', 0),
        ('Stat 105', 0),
        ('Stat 105 lab', 0),
        ('Stat 111', 0),
        ('Stat 117', 0),
        ('Stat 121', 0),
        ('Stat 130', 0),
        ('Stat 131', 0),
        ('Stat 131 lab', 0),
        ('Stat 138 lab', 0),
        ('Stat 138', 0),
        ('Stat 141', 0),
        ('Stat 145', 0),
        ('Stat 197', 0),
        ('AH 1', 1),
        ('AH 2', 1),
        ('AH 3', 1),
        ('SSP 1', 1),
        ('SSP 2', 1),
        ('SSP 3', 1),
        ('MST 1', 1),
        ('MST 2', 1),
    ]
)

# generate study plans
plans = [
    ('Physics', 1, 'Physical Science 11'),
    ('Physics', 1, 'Physics 51'),
    ('Physics', 1, 'Physics 51.1'),
    ('Physics', 1, 'AH 1'),
    ('Physics', 1, 'SSP 1'),

    ('Physics', 2, 'Physics 21'),
    ('Physics', 2, 'Physics 74'),
    ('Physics', 2, 'Physics 74 lab'),
    ('Physics', 2, 'AH 2'),
    ('Physics', 2, 'MST 1'),

    ('Physics', 3, 'Physics 72'),
    ('Physics', 3, 'Physics 72.1'),
    ('Physics', 3, 'MST 2'),
    ('Physics', 3, 'SSP 2'),

    ('Physics', 4, 'Physics 76'),
    ('Physics', 4, 'Physics 76 lab'),
    ('Physics', 4, 'AH 3'),
    ('Physics', 4, 'SSP 3'),

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

    ('Statistics', 1, 'Math 101'),
    ('Statistics', 1, 'Stat 104'),
    ('Statistics', 1, 'Stat 105'),
    ('Statistics', 1, 'Stat 105 lab'),
    ('Statistics', 1, 'AH 1'),
    ('Statistics', 1, 'SSP 1'),

    ('Statistics', 2, 'Stat 111'),
    ('Statistics', 2, 'Stat 117'),
    ('Statistics', 2, 'Stat 121'),
    ('Statistics', 2, 'Stat 130'),
    ('Statistics', 2, 'AH 2'),
    ('Statistics', 2, 'MST 1'),

    ('Statistics', 3, 'Stat 131'),
    ('Statistics', 3, 'Stat 131 lab'),
    ('Statistics', 3, 'Stat 138'),
    ('Statistics', 3, 'Stat 138 lab'),
    ('Statistics', 3, 'MST 2'),
    ('Statistics', 3, 'SSP 2'),

    ('Statistics', 4, 'Stat 141'),
    ('Statistics', 4, 'Stat 145'),
    ('Statistics', 4, 'Stat 197'),
    ('Statistics', 4, 'AH 3'),
    ('Statistics', 4, 'SSP 3'),

    ('Applied Mathematics', 1, 'Math 100'),
    ('Applied Mathematics', 1, 'Math 114'),
    ('Applied Mathematics', 1, 'Math 11A'),
    ('Applied Mathematics', 1, 'Math 121'),
    ('Applied Mathematics', 1, 'AH 1'),
    ('Applied Mathematics', 1, 'SSP 1'),

    ('Applied Mathematics', 2, 'Math 123'),
    ('Applied Mathematics', 2, 'Math 131'),
    ('Applied Mathematics', 2, 'Math 13'),
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
max_consecutive = 2
instructors = [
    'AAEA',
    'ALT',
    'CFCC',
    'ETB',
    'ETT',
    'ERA',
    'FJUC',
    'JCM',
    'JFJS',
    'JRRD',
    'JGC',
    'LAA',
    'MBSP',
    'MDP',
    'MBB',
    'MOO',
    'NCA',
    'RCLC',
    'RAD',
    'RCC',
    'VGN',
    'VTB',
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
    rows.append((name, max_consecutive))
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
    ('Math 123', '1', 'ETB', None, 3, 4, 1),
    ('Math 131', '1', 'ETB', None, 3, 4, 1),
    ('Math 11A', 'ABM', 'ETB', None, 3, 1, 1),
    ('Math 11A', 'ABM', 'ETB', None, 4, 1, 0),
    ('Math 174', '1', 'RCC', 'computer lab', 2, 2, 1),
    ('Math 121', '2', 'RCC', None, 3, 2, 1),
    ('Math 174', '2', 'RCC', 'computer lab', 2, 2, 1),
    ('Math 14', '2', 'RCC', None, 3, 4, 1),
    ('Math 174 lab', '1', 'RCC', 'computer lab', 6, 2, 0),
    ('Math 174 lab', '2', 'RCC', 'computer lab', 6, 2, 0),
    ('Math 190', '1', 'FJUC', None, 3, 4, 1),
    ('Math 55', '1', 'FJUC', None, 3, 4, 1),
    ('Math 121', '1', 'FJUC', None, 3, 2, 1),
    ('Math 20', '1', 'VGN', None, 3, 4, 1),
    ('Math 100', '2', 'VGN', None, 4, 1, 1),
    ('Math 100', '1', 'VGN', None, 4, 1, 1),
    ('Math 11A', 'STEM', 'MOO', None, 3, 1, 1),
    ('Math 11A', 'STEM', 'MOO', None, 4, 1, 0),
    ('Math 114', '1', 'MOO', None, 3, 2, 1),
    ('Math 100', '5', 'MOO', None, 4, 1, 1),
    ('Math 114', '1', 'MDP', None, 3, 2, 1),
    ('Math 100', '3', 'MDP', None, 4, 1, 1),
    ('Math 183', '1', 'MBSP', None, 3, 4, 1),
    ('Math 54', '1', 'MBSP', None, 3, 2, 1),
    ('Math 54', '1', 'MBSP', None, 4, 2, 0),
    ('Math 100', '4', 'MBSP', None, 4, 1, 1),
    ('Math 184', '1', 'ALT', None, 3, 4, 1),
    ('Math 178', '1', 'ALT', None, 3, 4, 1),
    ('Stat 141', '1', 'LAA', None, 3, 4, 1),
    ('Stat 111', '1', 'LAA', None, 4, 4, 1),
    ('Stat 104', '1', 'LAA', 'physics lab', 3, 4, 1),
    ('Stat 117', '1', 'LAA', None, 3, 4, 1),
    ('Stat 105', '1', 'LAA', 'computer lab', 4, 2, 0),
    ('Stat 105', '2', 'LAA', 'computer lab', 4, 2, 0),
    ('Stat 145', '1', 'VTB', None, 3, 4, 1),
    ('Stat 197', '1', 'VTB', None, 3, 4, 1),
    ('Math 101', '2', 'MBB', None, 3, 2, 1),
    ('Stat 131', '1', 'MBB', None, 3, 4, 1),
    ('Stat 131 lab', '1', 'MBB', None, 3, 4, 1),
    ('Stat 121', '1', 'MBB', None, 3, 4, 1),
    ('Stat 130', '1', 'MBB', None, 3, 4, 1),
    ('Stat 138 lab', '2', 'RAD', 'computer lab', 3, 2, 1),
    ('Stat 138 lab', '1', 'RAD', 'computer lab', 3, 2, 1),
    ('Stat 105 lab', '1', 'RAD', 'computer lab', 3, 2, 1),
    ('Stat 138', '1', 'RAD', None, 3, 2, 1),
    ('Stat 105 lab', '2', 'RAD', 'computer lab', 3, 1, 1),
    ('Math 101', '1', 'RAD', None, 3, 2, 1),
    ('Stat 105 lab', '3', 'RAD', 'computer lab', 3, 1, 1),
    ('Stat 138', '1', 'RAD', 'computer lab', 4, 2, 0),
    ('CMSC 142', '1', 'AAEA', 'computer lab', 3, 2, 1),
    ('CMSC 142', '2', 'AAEA', 'computer lab', 3, 2, 1),
    ('Math 143', '1', 'AAEA', 'computer lab', 2, 4, 1),
    ('Math 143 lab', '1', 'AAEA', 'computer lab', 3, 2, 1),
    ('Math 143 lab', '2', 'AAEA', 'computer lab', 3, 1, 1),
    ('Math 143 lab', '3', 'AAEA', 'computer lab', 3, 1, 1),
    ('CMSC 57', '1', 'AAEA', 'computer lab', 6, 4, 0),
    ('CMSC 198.1', '2', 'NCA', 'computer lab', 2, 1, 1),
    ('CMSC 137', '1', 'NCA', 'computer lab', 2, 2, 1),
    ('CMSC 198.1', '1', 'NCA', 'computer lab', 2, 1, 1),
    ('CMSC 137', '2', 'NCA', 'computer lab', 2, 2, 1),
    ('CMSC 11', '1', 'CFCC', 'computer lab', 2, 4, 1),
    ('CMSC 22', '1', 'CFCC', 'computer lab', 2, 4, 1),
    ('CMSC 151', '1', 'CFCC', 'computer lab', 3, 2, 1),
    ('CMSC 11 lab', '1', 'CFCC', 'computer lab', 3, 4, 1),
    ('CMSC 198.1', '4', 'CFCC', 'computer lab', 2, 1, 1),
    ('CMSC 22 lab', '1', 'CFCC', 'computer lab', 3, 4, 1),
    ('CMSC 151', '2', 'CFCC', 'computer lab', 3, 2, 1),
    ('CMSC 198.1', '3', 'JRRD', 'computer lab', 2, 1, 1),
    ('CMSC 141', '1', 'JRRD', 'computer lab', 3, 2, 1),
    ('CMSC 127 lab', '1', 'JRRD', 'computer lab', 3, 2, 1),
    ('CMSC 127 lab', '2', 'JRRD', 'computer lab', 3, 1, 1),
    ('CMSC 127 lab', '3', 'JRRD', 'computer lab', 3, 1, 1),
    ('CMSC 141', '2', 'JRRD', 'computer lab', 3, 2, 1),
    ('CMSC 192', '1', 'JCM', 'computer lab', 2, 2, 0),
    ('CMSC 131 lab', '1', 'JCM', 'computer lab', 3, 2, 1),
    ('CMSC 131 lab', '2', 'JCM', 'computer lab', 3, 1, 1),
    ('CMSC 131 lab', '3', 'JCM', 'computer lab', 3, 1, 1),
    ('CMSC 131', '1', 'JCM', 'computer lab', 2, 2, 1),
    ('CMSC 131', '2', 'JCM', 'computer lab', 2, 2, 1),
    ('CMSC 192', '2', 'JCM', 'computer lab', 2, 2, 1),
    ('CMSC 137 lab', '2', 'JCM', 'computer lab', 6, 2, 0),
    ('CMSC 137 lab', '1', 'JCM', 'computer lab', 6, 1, 0),
    ('CMSC 137 lab', '3', 'JCM', 'computer lab', 6, 1, 0),
    ('CMSC 127', '2', 'JFJS', 'computer lab', 2, 2, 1),
    ('CMSC 127', '1', 'JFJS', 'computer lab', 2, 2, 1),
    ('CMSC 128', '1', 'JFJS', 'computer lab', 2, 2, 1),
    ('CMSC 128', '2', 'JFJS', 'computer lab', 2, 2, 1),
    ('CMSC 128 lab', '1', 'JFJS', 'computer lab', 3, 2, 1),
    ('CMSC 128 lab', '2', 'JFJS', 'computer lab', 3, 1, 1),
    ('CMSC 128 lab', '3', 'JFJS', 'computer lab', 3, 1, 1),
    ('CMSC 162', '1', 'ETT', 'computer lab', 2, 2, 1),
    ('CMSC 162', '2', 'ETT', 'computer lab', 2, 2, 1),
    ('CMSC 162 lab', '2', 'ETT', 'computer lab', 6, 2, 0),
    ('CMSC 162 lab', '3', 'ETT', 'computer lab', 6, 1, 0),
    ('CMSC 162 lab', '4', 'ETT', 'computer lab', 6, 1, 0),
    ('Physics 76 lab', '1', 'ERA', 'physics lab', 3, 2, 1),
    ('Physics 76', '1', 'ERA', 'physics lab', 3, 2, 1),
    ('Physics 72', '1', 'ERA', 'physics lab', 4, 4, 1),
    ('Physics 76 lab', '2', 'ERA', 'physics lab', 3, 1, 1),
    ('Physical Science 11', '1', 'ERA', None, 3, 2, 1),
    ('Physical Science 11', '2', 'ERA', None, 3, 2, 1),
    ('Physics 74', '1', 'JGC', 'physics lab', 3, 4, 1),
    ('Physics 51.1', '3', 'JGC', 'physics lab', 4, 1, 1),
    ('Physics 51.1', '4', 'JGC', 'physics lab', 4, 1, 0),
    ('Physics 76', '2', 'JGC', 'physics lab', 3, 2, 1),
    ('Physics 74 lab', '1', 'JGC', 'physics lab', 6, 4, 0),
    ('Physics 72.1', '1', 'JGC', 'physics lab', 4, 4, 0),
    ('Physics 76 lab', '3', 'JGC', 'physics lab', 6, 1, 0),
    ('Physics 51', '1', 'RCLC', None, 3, 2, 1),
    ('Physics 51', '2', 'RCLC', 'physics lab', 3, 2, 1),
    ('Physics 21', '1', 'RCLC', None, 4, 4, 1),
    ('Physics 51.1', '1', 'RCLC', 'physics lab', 4, 1, 0),
    ('Physics 51.1', '2', 'RCLC', 'physics lab', 4, 1, 0),
    ('AH 1', '1', 'AH 1 Prof', None, 3, 4, 1),
    ('AH 1', '2', 'AH 1 Prof', None, 3, 4, 1),
    ('AH 1', '3', 'AH 1 Prof', None, 3, 4, 1),
    ('AH 1', '4', 'AH 1 Prof', None, 3, 4, 1),
    ('AH 2', '1', 'AH 2 Prof', None, 3, 4, 1),
    ('AH 2', '2', 'AH 2 Prof', None, 3, 4, 1),
    ('AH 2', '3', 'AH 2 Prof', None, 3, 4, 1),
    ('AH 2', '4', 'AH 2 Prof', None, 3, 4, 1),
    ('AH 3', '1', 'AH 3 Prof', None, 3, 4, 1),
    ('AH 3', '2', 'AH 3 Prof', None, 3, 4, 1),
    ('AH 3', '3', 'AH 3 Prof', None, 3, 4, 1),
    ('AH 3', '4', 'AH 3 Prof', None, 3, 4, 1),
    ('SSP 1', '1', 'SSP 1 Prof', None, 3, 4, 1),
    ('SSP 1', '2', 'SSP 1 Prof', None, 3, 4, 1),
    ('SSP 1', '3', 'SSP 1 Prof', None, 3, 4, 1),
    ('SSP 1', '4', 'SSP 1 Prof', None, 3, 4, 1),
    ('SSP 2', '1', 'SSP 2 Prof', None, 3, 4, 1),
    ('SSP 2', '2', 'SSP 2 Prof', None, 3, 4, 1),
    ('SSP 2', '3', 'SSP 2 Prof', None, 3, 4, 1),
    ('SSP 2', '4', 'SSP 2 Prof', None, 3, 4, 1),
    ('SSP 3', '1', 'SSP 3 Prof', None, 3, 4, 1),
    ('SSP 3', '2', 'SSP 3 Prof', None, 3, 4, 1),
    ('SSP 3', '3', 'SSP 3 Prof', None, 3, 4, 1),
    ('SSP 3', '4', 'SSP 3 Prof', None, 3, 4, 1),
    ('MST 1', '1', 'MST 1 Prof', None, 3, 4, 1),
    ('MST 1', '2', 'MST 1 Prof', None, 3, 4, 1),
    ('MST 1', '3', 'MST 1 Prof', None, 3, 4, 1),
    ('MST 1', '4', 'MST 1 Prof', None, 3, 4, 1),
    ('MST 2', '1', 'MST 2 Prof', None, 3, 4, 1),
    ('MST 2', '2', 'MST 2 Prof', None, 3, 4, 1),
    ('MST 2', '3', 'MST 2 Prof', None, 3, 4, 1),
    ('MST 2', '4', 'MST 2 Prof', None, 3, 4, 1),
]
c.executemany(
    'INSERT or REPLACE INTO sections VALUES (?, ?, ?, ?, ?, ?, ?)', sections)

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
    ('PL1', 4, 'physics lab'),
    ('PL2', 4, 'physics lab'),
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
