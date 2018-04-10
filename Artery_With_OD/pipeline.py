# -*- coding: utf-8 -*-
"""
Contributors:
    - Auriane Blarre
"""
import pandas as pd
from sensor_detections import sensor_detections, travel_times
from lp import get_alpha, get_delta, solve_pulse

def performance_metrics(detection_times):

    detection_times['travel_time_incoming'] = detection_times['detected_sensor_{}'.format(nb_sensors - 1)] - detection_times['detected_sensor_{}'.format(0)]
    detection_times['travel_time_outgoing'] = detection_times['detected_sensor_{}'.format(0)] - detection_times['detected_sensor_{}'.format(nb_sensors - 1)]

    average_incoming_travel_time = detection_times[detection_times['incoming'] == True]['travel_time_incoming'].mean()
    average_outgoing_travel_time = detection_times[detection_times['incoming'] == False]['travel_time_outgoing'].mean()

    max_incoming_travel_time = detection_times[detection_times['incoming'] == True]['travel_time_incoming'].max()
    max_outgoing_travel_time = detection_times[detection_times['incoming'] == False]['travel_time_outgoing'].max()

    min_incoming_travel_time = detection_times[detection_times['incoming'] == True]['travel_time_incoming'].min()
    min_outgoing_travel_time = detection_times[detection_times['incoming'] == False]['travel_time_outgoing'].min()

    print("Average incoming travel time: {}".format(average_incoming_travel_time))
    print("Average outgoing travel time: {}".format(average_outgoing_travel_time))
    print("Average travel time: {}".format((average_incoming_travel_time + average_outgoing_travel_time) / 2))

    print("Max incoming travel time: {}".format(max_incoming_travel_time))
    print("Max outgoing travel time: {}".format(max_outgoing_travel_time))
    print("Max travel time: {}".format((max_incoming_travel_time + max_outgoing_travel_time) / 2))

    print("Min incoming travel time: {}".format(min_incoming_travel_time))
    print("Min outgoing travel time: {}".format(min_outgoing_travel_time))
    print("Min travel time: {}".format((min_incoming_travel_time + min_outgoing_travel_time) / 2))


def optimize(sumo_output, sensors_x, sensors_y, sensors_rad, C, n_intersections, \
             g_i_inbound, g_i_outbound, theta_incoming, theta_outgoing, verbose=True):
    detection_times = sensor_detections(sumo_output, sensors_x, sensors_y, sensors_rad, perc=0.2)
    if verbose == True:
        print("Before optimization")
        # performance_metrics(detection_times)

    alpha = get_alpha(detection_times)
    if verbose == True:
        print("Alpha: {}".format(alpha))
    delta = get_delta(theta_incoming, theta_outgoing, C)
    if verbose == True:
        print("Delta: {}".format(delta))
    travel_time_incoming, travel_time_outgoing = travel_times(detection_times, sensors_x, sensors_y)

    # Solve the problem with kwargs arguments
    theta_inbound, theta_outbound = solve_pulse(alpha, n_intersections, C, g_i_inbound, g_i_outbound, delta, travel_time_incoming, travel_time_outgoing, verbose=verbose)
    return theta_inbound, theta_outbound

if __name__ == '__main__':
    sumo_output = pd.read_csv("quickstartod1.csv", sep=",")

    # Sensor data
    # RATE OF ACQUIREMENT
    # Width 3 lanes 10
    # Width perpendicular lane 6.5
    n_intersections = 8
    n_sensors = n_intersections
    sensors_x = [400 + 200 * k + 7 for k in range(n_sensors)] + [400 + 200 * k - 7 for k in range(n_sensors)]
    sensors_y = [200 - 10 for k in range(n_sensors)] + [200 + 10 for k in range(n_sensors)]
    sensors_rad = [10.0 for k in range(n_sensors)] + [10.0 for k in range(n_sensors)]

    # Network data
    C = 90
    g_i_inbound = [29 for k in range(n_intersections)]
    g_i_outbound = [29 for k in range(n_intersections)]
    theta_incoming = [0 for k in range(n_intersections)]
    theta_outgoing = [0 for k in range(n_intersections)]

    theta_inbound, theta_outbound = optimize(sumo_output, sensors_x, sensors_y, sensors_rad, C, n_intersections, \
             g_i_inbound, g_i_outbound, theta_incoming, theta_outgoing, verbose=True)