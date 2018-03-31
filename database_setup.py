import sqlite3

conn = sqlite3.connect('database.sqlite3')

c = conn.cursor()

# create programs table
c.execute("""
CREATE TABLE programs (
    name text NOT NULL,
    PRIMARY KEY (name)
)""")

# create room_types table
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
    is_wildcard tinyint NOT NULL,
    PRIMARY KEY (name),
    FOREIGN KEY (room_type) REFERENCES room_types(name)
)""")

# create study_plans table
c.execute("""
CREATE TABLE study_plans(
    program text NOT NULL,
    year int NOT NULL,
    course text NOT NULL,
    FOREIGN KEY (program) REFERENCES programs(name),
    FOREIGN KEY (course) REFERENCES courses(name)
)""")

# create instructors table
c.execute("""
CREATE TABLE instructors (
    name text NOT NULL,
    max_consecutive int NOT NULL,
    PRIMARY KEY (name)
)""")

# create availability table
# day is an integer from 1-5 for the weekday
c.execute("""
CREATE TABLE availability (
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

conn.commit()
conn.close()
