"""Dummy study plan generator."""

import random
from timetabling import Course, NonmajorCourse


def generate_study_plans(programs):
    """
    Create dummy study plans for all 4 year levels for each given program name.
    All plans have 4 program-specific classes and 2 "GE" classes that are in more than one study plan.
    Half of major courses have a required room category. Wildcard courses can be held anywhere.
    """
    plans = {}
    for program in programs:
        for year in range(1, 5):
            plans[(program, year)] = set([
                NonmajorCourse('shared-{}{}'.format(year, shared))
                for shared
                in 'ab'])
            plans[(program, year)].update([
                Course('{}-{}{}'.format(program, year, major),
                       random.choice([program, None]))
                for major
                in 'abcd'])
    return plans
