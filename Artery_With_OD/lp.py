# -*- coding: utf-8 -*-
"""
Contributors:
    - Auriane Blarre
"""

from amplpy import AMPL, Environment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# AMPL path
path = '/Users/aurianeblarre/Documents/Berkeley/ENGIN296/Projet/amplide.macosx64'

def get_alpha(detection_times):
    alpha = detection_times['incoming'].sum() / detection_times['incoming'].count()
    return alpha

def get_delta(theta_incoming, theta_outgoing, C):
    return [modulo(y - x, C) for x, y in zip(theta_incoming, theta_outgoing)]

def compute_delta0(delta, travel_time_incoming, travel_time_outgoing, C):
    delta0 = [delta[0]]
    running_sum = 0
    n = len(delta)
    for i in range(1, n):
        running_sum += travel_time_incoming[i - 1] - travel_time_outgoing[i - 1]
        delta0.append(modulo(delta[i] + running_sum, C))

    return delta0

def compute_bandwidths(w_incoming, delta0, g_i_inbound, g_i_outbound):
    n = len(w_incoming)
    outgoing_min = 1000000
    for i in range(1, n):
        for j in range(1, n):
            outgoing_min = min(outgoing_min, w_incoming[i] + delta0[i] - (
                w_incoming[j] + delta0[j]) + (g_i_outbound[i] + g_i_outbound[j]) / 2)

    incoming_min = 1000000
    for i in range(1, n):
        for j in range(1, n):
            incoming_min = min(incoming_min, w_incoming[i] + w_incoming[j] + (
                g_i_inbound[i] + g_i_inbound[j]) / 2)

    return incoming_min, outgoing_min

def f_l(w_incoming, delta0, g_i_inbound, g_i_outbound, alpha):
    incoming_min, outgoing_min = compute_bandwidths(w_incoming, delta0, g_i_inbound, g_i_outbound)

    return alpha * incoming_min + (1 - alpha) * outgoing_min

def f_n(w_incoming, delta0, g_i_inbound, g_i_outbound, alpha):
    incoming_min, outgoing_min = compute_bandwidths(w_incoming, delta0, g_i_inbound, g_i_outbound)

    return alpha * max(0, incoming_min) + (1 - alpha) * max(0, outgoing_min)

def solver(alpha, n_intersections, C, g_i_inbound, g_i_outbound, delta, verbose=True):
    """
    Solves the linear problem for the given set of parameters
    :param alpha:
    :type alpha:
    :param n_intersections:
    :type n_intersections:
    :param C:
    :type C:
    :param g_i_inbound:
    :type g_i_inbound:
    :param g_i_outbound:
    :type g_i_outbound:
    :param delta:
    :type delta:
    :return:
    :rtype:
    """
    ampl = AMPL(Environment(path))

    # Set the solver
    ampl.setOption('solver', path + '/cplex')

    # Read the model file
    ampl.read('lp.mod')

    # Set parameters values
    ampl.param['alpha'] = alpha
    ampl.param['N'] = n_intersections
    ampl.param['C'] = C
    ampl.param['g_incoming'] = g_i_inbound
    ampl.param['g_outgoing'] = g_i_outbound
    ampl.param['delta'] = delta

    if verbose == True:
        print("alpha: {}".format(alpha))
        print("N: {}".format(n_intersections))
        print("C: {}".format(C))
        print("g_incoming: {}".format(g_i_inbound))
        print("g_outgoing: {}".format(g_i_outbound))
        print("delta: {}".format(delta))

    # Resolve and display objective
    ampl.solve()
    bandwidth = ampl.getObjective('bandwidth').value()

    # Display the variables
    b_incoming = ampl.getVariable('b_incoming').value()
    b_outgoing = ampl.getVariable('b_outgoing').value()
    wl = ampl.getVariable('w').getValues()

    if verbose == True:
        print("New objective value: {}".format(bandwidth))
        print("New offsets: {}".format(list(wl.toPandas()['w.val'])))
        print("Incoming bandwidth: {}".format(b_incoming))
        print("Outgoing bandwidth: {}".format(b_outgoing))

    # return bandwidths and offset values
    return b_incoming, b_outgoing, list(wl.toPandas()['w.val'])


def solve_pulse(alpha, n_intersections, C, g_i_inbound, g_i_outbound, delta, travel_time_incoming,
                travel_time_outgoing, verbose=True, test=False):
    delta0 = compute_delta0(delta, travel_time_incoming, travel_time_outgoing, C)
    gA = max(min(g_i_inbound), min(g_i_outbound))

    if verbose == True:
        print("Original bandwidth: {}".format(gA))

    b_incoming, b_outgoing, wL = solver(alpha, n_intersections, C, g_i_inbound, g_i_outbound, delta0, verbose=verbose)
    # Evaluate wL in constraints set

    if test == True:
        # f_l is the same as LP objective function
        assert b_incoming, b_outgoing == compute_bandwidths(wL, delta0, g_i_inbound, g_i_outbound)
        test_lp(wL, delta0, g_i_inbound, g_i_outbound, alpha, C, n_tests=1000)

    new_bandwidth = f_l(wL, delta0, g_i_inbound, g_i_outbound, alpha)

    if verbose == True:
        print("New bandwidth: {}".format(new_bandwidth))

    n = len(delta)

    if new_bandwidth >= gA:
        if verbose == True:
            print("Bandwidth improved, updating the offsets")
        wN = wL
    else:
        if min(g_i_inbound) > min(g_i_outbound):
            if verbose == True:
                print("Bandwidth not improved, sychronizing offsets with incoming traffic")
            wN = [0 for i in range(n)]
        else:
            if verbose == True:
                print("Bandwidth not improved, sychronizing offsets with outgoing traffic")
            wN = delta

    if test == True:
        test_offsets(wN, delta0, g_i_inbound, g_i_outbound, alpha, C, n_tests=1000)

    w_outgoing = []
    for i in range(n):
        w_outgoing.append(wN[i] - (delta0[0] - delta0[i]))

    if test == True:
        test_offset_internal_relation(delta0, wN, w_outgoing)

    n = len(wL)
    theta_incoming = [0]
    theta_outgoing = [delta[0]]

    theta1 = theta_incoming[0]
    running_sum = 0

    for i in range(1, n):
        running_sum += travel_time_incoming[i - 1]
        theta_incoming = modulo(theta1 + wN[i] + running_sum, C)

    theta1 = theta_outgoing[0]
    running_sum = 0

    for i in range(1, n):
        running_sum += travel_time_outgoing[i - 1]
        theta_incoming = modulo(theta1 + w_outgoing[i] + running_sum, C)
    return theta_incoming, theta_outgoing

def test_offset_internal_relation(delta0, w_incoming, w_outgoing):
    n = len(delta0)
    for i in range(n):
        assert w_incoming[i] - w_outgoing[i] == delta0[0] - delta0[i], """Internal offset relation not respected at index {}""".format(i)

def test_lp(w_incoming, delta0, g_i_inbound, g_i_outbound, alpha, C, n_tests=1000):
    optimal_bandwidth = f_l(w_incoming, delta0, g_i_inbound, g_i_outbound, alpha)
    n = len(w_incoming)
    # Separate b incoming and outgoing
    it = 0
    while it < n_tests:
        random_offsets = C * np.random.rand(n) - C / 2
        is_feasible = of_filter(random_offsets, delta0, g_i_inbound, g_i_outbound, alpha)
        if not is_feasible:
            continue
        else:
            it += 1
            random_bandwidth = f_l(random_offsets, delta0, g_i_inbound, g_i_outbound, alpha)
            assert random_bandwidth <= optimal_bandwidth, """\n 
            Found better offsets for (L) after {} iterations: {} \n
            Optimal offsets: {} \n
            Optimal bandwidth: {} \n
            New random bandwidth: {}""".format(it, random_offsets, w_incoming, optimal_bandwidth, random_bandwidth)


def test_offsets(w_incoming, delta0, g_i_inbound, g_i_outbound, alpha, C, n_tests=1000):
    optimal_bandwidth = f_n(w_incoming, delta0, g_i_inbound, g_i_outbound, alpha)
    n = len(w_incoming)
    it = 0
    while it < n_tests:
        random_offsets = C * np.random.rand(n) - C / 2
        is_feasible = of_filter(random_offsets, delta0, g_i_inbound, g_i_outbound, alpha)
        if not is_feasible:
            continue
        else:
            it += 1
            random_bandwidth = f_n(random_offsets, delta0, g_i_inbound, g_i_outbound, alpha)
            assert random_bandwidth <= optimal_bandwidth, """\n 
            Found better offsets for (N) after {} iterations: {} \n
            Optimal offsets: {} \n
            Optimal bandwidth: {} \n
            New random bandwidth: {}""".format(it, random_offsets, w_incoming, optimal_bandwidth, random_bandwidth)


def of_filter(offsets, delta0, g_i_inbound, g_i_outbound, alpha):
    """
    Check whether the offset is in feasible area
    """
    b_incoming, b_outgoing = compute_bandwidths(offsets, delta0, g_i_inbound, g_i_outbound)
    N = len(offsets)
    for i in range(N):
        if (b_incoming > g_i_inbound[i] or b_outgoing > g_i_outbound[i]):
            return False


    for i in range(1, N):
        for j in range(1, N):
            if (i != j and b_incoming + offsets[i] - offsets[j] > (g_i_inbound[i] + g_i_inbound[j]) / 2):
                return False
            if (i != j and b_incoming - offsets[i] + offsets[j] > (g_i_inbound[i] + g_i_inbound[j]) / 2):
                return False
            if (i != j and b_outgoing + offsets[i] - offsets[j] > (g_i_outbound[i] + g_i_inbound[j]) / 2 - (delta0[i] - delta0[j])):
                return False
            if (i != j and b_outgoing - offsets[i] + offsets[j] > (g_i_outbound[i] + g_i_inbound[j]) / 2 + (delta0[i] - delta0[j])):
                return False

        if b_incoming + offsets[i] > (g_i_inbound[0] + g_i_inbound[i]) / 2:
            return False
        if b_incoming - offsets[i] > (g_i_inbound[0] + g_i_inbound[i]) / 2:
            return False
        if b_outgoing + offsets[i] > (g_i_outbound[0] + g_i_outbound[i]) / 2 - (delta0[i] - delta0[1]):
            return False
        if b_outgoing - offsets[i] > (g_i_outbound[0] + g_i_outbound[i]) / 2 + (delta0[i] - delta0[1]):
            return False

    return True


def modulo(t, C):
    """
    Custom modulo function
    :param t:
    :type t:
    :param C:
    :type C:
    :return:
    :rtype:
    """
    res = t // C
    if t - res * C < (res + 1) * C - t:
        return t - res * C
    else:
        return t - (res + 1) * C


if __name__ == '__main__':
    # Network Parameters
    # C, n_intersections, g_i_inbound, g_i_outbound, theta_incoming, theta_outgoing
    C = 70
    n_intersections = 2
    g_i_inbound = [0.5 * 70, 0.4 * 70]
    g_i_outbound = [0.7 * 70, 0.6 * 70]
    theta_incoming = [0, 0]
    theta_outgoing = [0, 0]

    detection_times = pd.read_csv('detection_times.csv')
    alpha = get_alpha(detection_times)
    delta = get_delta(theta_incoming, theta_outgoing, C)

    # Solve the problem with kwargs arguments
    solver(alpha, n_intersections, C, g_i_inbound, g_i_outbound, delta)

    # Run test function of alpha
    test_alpha()

    # run test funtion of the number of intersections
    test_n_intersections()

    # plot a sample solution
    # kwargs = generate_arguments(alpha, n_intersections, C)
    # positions, speeds, travel_time = generate_arterial(kwargs['n_intersections'])
    # w_relative =
    # plot_solution(positions, g_i_inbound, g_i_outbound, w_relative, delta, travel_time, C)