import sqlite3

conn = sqlite3.connect('database.sqlite3')

c = conn.cursor()

# create programs table
c.execute("""
CREATE TABLE programs(
    name text NOT NULL,
    PRIMARY KEY (name)
)""")

# create room_types table
# examples of room types can be "physics lab", "computer lab", etc
c.execute("""
CREATE TABLE room_types(
    name text NOT NULL,
    PRIMARY KEY (name)
)""")

# create courses table
c.execute("""
CREATE TABLE courses(
    name text NOT NULL,
    room_type text,
    is_nonmajor tinyint NOT NULL,
    PRIMARY KEY (name),
    FOREIGN KEY (room_type) REFERENCES room_types(name),
    CHECK (is_nonmajor BETWEEN 0 AND 1)
)""")

# create study_plans table
c.execute("""
CREATE TABLE study_plans(
    program text NOT NULL,
    year int NOT NULL,
    course text NOT NULL,
    FOREIGN KEY (program) REFERENCES programs(name),
    FOREIGN KEY (course) REFERENCES courses(name),
    CHECK (year BETWEEN 1 AND 4)
)""")

# create program_sizes table
c.execute("""
CREATE TABLE program_sizes(
    program text NOT NULL,
    year int NOT NULL,
    size int NOT NULL,
    FOREIGN KEY (program) REFERENCES programs(name),
    CHECK (year BETWEEN 1 AND 4),
    CHECK (size > 0)
)""")

# create instructors table
c.execute("""
CREATE TABLE instructors(
    name text NOT NULL,
    max_consecutive int NOT NULL,
    PRIMARY KEY (name),
    CHECK (max_consecutive > 0)
)""")

# create availability table
# day is an integer from 1-5 for the weekday
c.execute("""
CREATE TABLE availability(
    instructor text NOT NULL,
    day int NOT NULL,
    start int NOT NULL,
    end int NOT NULL,
    FOREIGN KEY (instructor) REFERENCES instructors(name),
    CHECK (day BETWEEN 1 AND 5),
    CHECK (start < end),
    CHECK (start / 100 BETWEEN 0 AND 23),
    CHECK (start % 100 in (0, 30)),
    CHECK (end / 100 BETWEEN 0 AND 23),
    CHECK (end % 100 in (0, 30))
)""")

# create sections table
# section data without time and room
# sections don't have unique identifiers. use default rowid column instead
c.execute("""
CREATE TABLE sections(
    course text NOT NULL,
    section_id text NOT NULL,
    instructor text NOT NULL,
    length int NOT NULL,
    size int NOT NULL,
    is_twice_weekly int NOT NULL,
    FOREIGN KEY (course) REFERENCES courses(name),
    FOREIGN KEY (instructor) REFERENCES instructors(name),
    CHECK (length > 0),
    CHECK (size > 0)
)""")

# create rooms table
c.execute("""
CREATE TABLE rooms(
    name text NOT NULL,
    capacity int NOT NULL,
    category text,
    FOREIGN KEY (category) REFERENCES room_types(name)
    CHECK (capacity > 0)
)""")

# create restrictions table
c.execute("""
CREATE TABLE restrictions(
    section int NOT NULL,
    program text NOT NULL,
    year int NOT NULL,
    FOREIGN KEY (section) REFERENCES sections(rowid),
    FOREIGN KEY (program) REFERENCES programs(name),
    CHECK (year BETWEEN 1 AND 4)
)""")

conn.commit()
conn.close()
