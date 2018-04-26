# -*- coding: utf-8 -*-
"""
Contributors:
    - Auriane Blarre
"""
import re


def write_line(time, phase):
    return """        <phase duration="{}" state="{}"/>\n""".format(int(time), phase)

def modify_offset(theta_incoming, theta_outgoing, gi_incoming, gi_outgoing, C, trans_time=5):
    # Transition: before light turns red from green it is yellow for 3s
    text = """"""
    t0 = 0

    # First Phase
    t1 = min(theta_incoming, theta_outgoing)
    phase = "GGgsrrrGGgsrrr"
    if t1 - t0 > 0:
       text += write_line(t1 - t0, phase)

    # Second Phase
    t2 = max(theta_incoming, theta_outgoing)
    if t2 == theta_outgoing:
        # incoming is green
        # down transitions yellow
        trans_phase = "GGgsrrryygsrrr"
        main_phase = "GGgsrrrsrrGGGg"
    if t2 == theta_incoming:
        # outgoing is green
        # up transitions yellow
        trans_phase = "yygsrrrGGgsrrr"
        main_phase = "srrGGGgGGgsrrr"

    if t2 - t1 - trans_time > 0:
        text += write_line(trans_time, trans_phase)
        text += write_line(t2 - t1 - trans_time, main_phase)

    # Third Phase
    # Both are green
    t3 = min(theta_incoming + gi_incoming, theta_outgoing + gi_outgoing)
    if t2 == theta_outgoing:
        # up transitions yellow
        trans_phase = "yygsrrrsrrGGGg"
    if t2 == theta_incoming:
        # down transitions yellow
        trans_phase = "GGgsrrryygsrrr"
    main_phase = "srrGGGgsrrGGGg"
    if t3 - t2 - trans_time > 0:
        text += write_line(trans_time, trans_phase)
        text += write_line(t3 - t2 - trans_time, main_phase)

    # Fourth Phase
    t4 = max(theta_incoming + gi_incoming, theta_outgoing + gi_outgoing)
    if t4 == theta_outgoing + gi_outgoing:
        # incoming is red
        trans_phase = "srrGGGgGGgyyyg"
        main_phase = "srrGGGgGGgsrrr"
    if t4 == theta_incoming + gi_incoming:
        # outgoing is red
        trans_phase = "GGgyyygsrrGGGg"
        main_phase = "GGgsrrrsrrGGGg"

    if t4 - t3 - trans_time > 0:
        text += write_line(trans_time, trans_phase)
        text += write_line(t4 - t3 - trans_time, main_phase)

    # Fifth Phase
    # Both are red
    t5 = C
    if t4 == theta_outgoing + gi_outgoing:
        trans_phase = "GGgyyyyGGgsrrr"
    if t2 == theta_incoming:
        trans_phase = "GGgsrrrsrryyyy"
    main_phase = "GGgsrrrGGgsrrr"

    if t5 - t4 - trans_time > 0:
        text += write_line(trans_time, trans_phase)
        text += write_line(t5 - t4 - trans_time, main_phase)
    return text

def modify_offsets(thetas_incoming, thetas_outgoing, gis_incoming, gis_outgoing, C, trans_time=3,
                   network_path="quickstart.net.xml"):
    file = open(network_path, "r")
    network = file.read()
    file.close()

    id = 0
    for theta_incoming, theta_outgoing, gi_incoming, gi_outgoing in zip(thetas_incoming, thetas_outgoing, gis_incoming, gis_outgoing):
        replacement = modify_offset(theta_incoming, theta_outgoing, gi_incoming, gi_outgoing, C, trans_time=trans_time)

        start = """<tlLogic id="{}" type="static" programID="0" offset="0">\n""".format(id)
        end = "</tlLogic>"
        match = re.match(r'(.+%s\s*).+?(\s*%s.+)' % (start, end), network, re.DOTALL)
        network = match.group(1) + replacement + match.group(2)
        id += 1

    file_handle = open(network_path, "w")
    file_handle.write(network)
    file_handle.close()

    return network


if __name__ == '__main__':
    thetas_incoming = [20 for i in range(8)]
    thetas_outgoing = [20 for i in range(8)]
    gis_incoming = [50 for i in range(8)]
    gis_outgoing = [50 for i in range(8)]
    C = 90

    modify_offsets(thetas_incoming, thetas_outgoing, gis_incoming, gis_outgoing, C, trans_time=3,
                   network_path="quickstart.net.xml")