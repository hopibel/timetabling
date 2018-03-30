import sqlite3

conn = sqlite3.connect('database.sqlite3')

c = conn.cursor()

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
    FOREIGN KEY (course) REFERENCES courses(name)
)""")

conn.commit()
conn.close()
