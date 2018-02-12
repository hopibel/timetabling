"""Dummy study plan generator."""


def generate_study_plans(course_names):
    """
    Create dummy study plans for all 4 year levels for each given course name.
    All plans have 4 course-specific classes and 2 "GE" classes that are in more than one study plan.
    """
    plans = {}
    for course in course_names:
        plans[course] = {}
        for year in range(1, 5):
            plans[course][year] = ["shared-{}{}".format(year, shared) for shared in "ab"]
            plans[course][year].extend(["{}-{}{}".format(course, year, major) for major in "abcd"])
    return plans
