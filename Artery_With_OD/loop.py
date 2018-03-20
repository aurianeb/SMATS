# -*- coding: utf-8 -*-
"""
Contributors:
    - Auriane Blarre
"""

from sensor_detections import sensor_detections
from lp import solver


if __name__ == '__main__':

    C = 200 # Cycle length
    'g_i_inbound': [0.5 * C, 0.4 * C, 0.5 * C],  # portion of cycle
    'g_i_outbound': [0.7 * C, 0.6 * C, 0.4 * C],  # portion of cycle
    n_intersections = 8

    sumo_output = pd.read_csv("quickstartod1.csv", sep=",")
    nb_sensors = 2
    sensors_x = [406.55, 393.45]
    sensors_y = [190.00, 210.00]
    sensors_rad = [200.00, 200.00]

    detection_times = sensor_detections(sumo_output, sensors_x, sensors_y, sensors_rad, perc=0.2)
    detection_times.to_csv("detection_times.csv")

    solver(alpha, n_intersections, C, g_i_inbound, g_i_outbound, delta, print_=False)