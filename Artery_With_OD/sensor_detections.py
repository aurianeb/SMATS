# -*- coding: utf-8 -*-
"""
Contributors:
    - Auriane Blarre
"""
import pandas as pd
import numpy as np
from random import shuffle


def vehicles_detected(sensor_data, perc):
    """
    Pick perc cars from sensor_data that can be detected by the sensors
    :param sensor_data:
    :type sensor_data: pandas DataFrame
    :param perc:
    :type perc: float from 0 to 1
    :return: cars from sensor_data that can be detected by the sensors
    :rtype: pandas DataFrame
    """
    vids = sensor_data['vehicle_id'].unique()
    sample_size = int(perc * len(vids))
    shuffle(vids)
    detected_vehicles = vids[:sample_size]
    sensor_data = sensor_data[sensor_data['vehicle_id'].isin(detected_vehicles)]
    return sensor_data, detected_vehicles

def detected_by_sensor(x, y, x0, y0, r):
    """
    Whether or not vehicule at position (x, y) can be detected by sensor at (x0, y0) with radius r
    :param x:
    :type x:
    :param y:
    :type y:
    :param x0:
    :type x0:
    :param y0:
    :type y0:
    :param r:
    :type r:
    :return:
    :rtype:
    """
    return (x - x0) ** 2 + (y - y0) ** 2 <= r ** 2

def sensor_detections(sumo_output, sensors_x, sensors_y, sensors_rad, perc=0.2):
    """
    List of the vehicles detected by sensors, with detection times (Nan if never detected)
    :param sumo_output: sumo output
    :type sumo_output: pandas DataFrame
    :param sensors_x: x coordinates of sensors
    :type sensors_x: List
    :param sensors_y: y coordinates of sensors
    :type sensors_y: List
    :param sensors_rad: radiuses of sensors
    :type sensors_rad: List
    :param perc: percentage of vehicles that can be detected
    :type perc: float
    :return: vehicles detected by sensors, with detection times
    :rtype: pandas DataFrame
    """
    # Data the sensor gets
    sensor_data = sumo_output[['timestep_time', 'vehicle_id', 'vehicle_x', 'vehicle_y']]
    sensor_data, detected_vehicles = vehicles_detected(sensor_data, perc)

    nb_sensors = len(sensors_x)
    for i in range(nb_sensors):
        sensor_x, sensor_y, sensor_rad = sensors_x[i], sensors_y[i], sensors_rad[i]
        sensor_data['detected_sensor_{}'.format(i)] = detected_by_sensor(sensor_data['vehicle_x'],
                                                                         sensor_data['vehicle_y'],
                                                                         sensor_x,
                                                                         sensor_y,
                                                                         sensor_rad)
    sensor_data.drop(['vehicle_x', 'vehicle_y'], inplace=True, axis=1)

    detection_times = pd.DataFrame(index=detected_vehicles)
    for i in range(nb_sensors):
        col_name = 'detected_sensor_{}'.format(i)
        detection_times[col_name] = sensor_data[sensor_data[col_name] == True].groupby(['vehicle_id'])[
            'timestep_time'].min()

    return detection_times


if __name__ == '__main__':
    sumo_output = pd.read_csv("quickstartod1.csv", sep=",")
    nb_sensors = 2
    sensors_x = [406.55, 393.45]
    sensors_y = [190.00, 210.00]
    sensors_rad = [200.00, 200.00]

    detection_times = sensor_detections(sumo_output, sensors_x, sensors_y, sensors_rad, perc=0.2)
    detection_times.to_csv("detection_times.csv")