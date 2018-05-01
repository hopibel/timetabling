"""Room view HTML visualizer for timetable."""

from collections import defaultdict
import os


def to_html(ind, sections, slots, day_slots, day_start, outdir):
    """Convert timetable to html table."""
    rooms = defaultdict(list)
    for section in ind:
        section_data = sections[section.section_id]
        meeting = {
            'name': "{}-{}<br>{}".format(section_data.course.name,
                                      section_data.id_,
                                      section_data.instructor.name),
            'start': section.slot,
            'end': section.slot + section_data.length - 1,
        }
        rooms[section.room].append(meeting)

        if section.is_twice_weekly:
            meeting2 = meeting.copy()
            meeting2['start'] += day_slots * 3
            meeting2['end'] += day_slots * 3
            rooms[section.room].append(meeting2)

    tables = {}
    for key, room in rooms.items():
        points = []  # list of (offset, plus/minus, name) tuples
        for course in room:
            points.append((course['start'], '+', course['name']))
            points.append((course['end'], '-', course['name']))
        points.sort(key=lambda x: x[1])
        points.sort(key=lambda x: x[0])

        ranges = []  # output list of (start, stop, symbol_set) tuples
        current_set = []
        last_start = None
        offset = points[0][0]
        for offset, pm, name in points:
            if pm == '+':
                if last_start is not None and current_set and offset - last_start > 0:
                    ranges.append((last_start, offset-1, current_set.copy()))
                current_set.append(name)
                last_start = offset
            elif pm == '-':
                if offset >= last_start:
                    ranges.append((last_start, offset, current_set.copy()))
                current_set.remove(name)
                last_start = offset+1

        cells = []
        last_slot = 0
        for r in ranges:  # ranges = list of (start, end, {names})
            if r[0] > last_slot:
                for i in range(last_slot, r[0]):
                    cells.append((i, i, []))
            cells.append(r)
            last_slot = r[1] + 1
        for i in range(last_slot+1, slots):
            cells.append((i, i, []))

        rows = list([] for _ in range(day_slots))
        for cell in cells:
            rows[cell[0] % day_slots].append(cell)

        table = []
        table.append("<table>")
        table.append("""
<tr>
<th>Time</th>
<th>Monday</th>
<th>Tuesday</th>
<th>Wednesday</th>
<th>Thursday</th>
<th>Friday</th>
</tr>
"""[1:-1])
        for i, row in enumerate(rows):
            table.append("<tr>")
            table.append("<td>{}</td>".format(str(day_start + 100*(i//2) + (i % 2)*30).zfill(4)))
            for cell in row:
                if cell[2] == []:
                    table.append("<td>&nbsp;</td>")
                else:
                    line = "<td"
                    if cell[1] > cell[0]:
                        line += " rowspan={}".format(cell[1] - cell[0] + 1)
                    if len(cell[2]) > 1:
                        line += " class=overlap"
                    else:
                        line += " class=course"
                    line += ">"
                    line += "<br>".join(str(x) for x in cell[2])
                    line += "</td>"
                    table.append(line)
            table.append("</tr>")
        table.append("</table>")
        tables[key] = table

    html = []
    # boilerplate
    html.append("""
<head>
<style>
table { border-collapse: collapse; }
table, th, td { border: 1px solid black; }
td { text-align: center; }
.overlap { background-color: orange; }
.course { background-color: #93c572; }
</style>
</head>
"""[1:-1])
    for name, table in tables.items():
        html.append("<b>Room {}</b><br>".format(name))
        for line in table:
            html.append(line)

    with open(os.path.join(outdir, 'room_sched.html'), 'w') as outfile:
        for line in html:
            outfile.write(line + "\n")
