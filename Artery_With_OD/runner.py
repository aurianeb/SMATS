# -*- coding: utf-8 -*-
"""
Contributors:
    - Auriane Blarre
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import subprocess
import sys


def build_configuration_file(input_cfg_path, output_cfg_path):
    file = open(input_cfg_path, "r")
    cfg = file.read()
    file.close()

    cfg = cfg.replace("""<net-file value="quickstart.net.xml"/>""", """<net-file value="optimized.net.xml"/>""")
    cfg = cfg.replace("""<fcd-output value="quickstartod1.output.xml" />""", """<fcd-output value="optimized.output.xml" />""")

    file_handle = open(output_cfg_path, "w")
    file_handle.write(cfg)
    file_handle.close()


def runner(file, stat):
    """
    file: the name of configuration file
    stat: output statistic result such as RouteLength, WaitingTime, etc.
    """
    try:
        sys.path.append(os.path.dirname(
            __file__))  # tutorial in tests
        os.environ['SUMO_HOME'] = "/Users/aurianeblarre/Documents/Berkeley/Capstone/sumo-0.30.0"
        sys.path.append(os.path.join(os.environ.get("SUMO_HOME"), "tools"))
        from sumolib import checkBinary
    except ImportError:
        sys.exit(
            "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

    sumoBinary = checkBinary('sumo')

    if stat:
        com = "--duration-log.statistics"
    else:
        com = "-"
    retcode = subprocess.call(
        [sumoBinary, "-c", file, com], stdout=sys.stdout, stderr=sys.stderr)
    print(">> Simulation closed with status %s" % retcode)
    sys.stdout.flush()


if __name__ == "__main__":
    build_configuration_file("quickstart.sumocfg", "optimized.sumocfg")
    # os.system("sumo-gui -c optimized.sumocfg")

    # runner("optimized.sumocfg", False)