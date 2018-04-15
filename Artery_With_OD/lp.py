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

    # Resolve and display objective
    ampl.solve()
    bandwidth = ampl.getObjective('bandwidth').value()

    # Display the variables
    b_incoming = ampl.getVariable('b_incoming').value()
    b_outgoing = ampl.getVariable('b_outgoing').value()
    wl = ampl.getVariable('w').getValues()

    if verbose == True:
        print("New objective value: {}".format(bandwidth))
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

    if test == True:
        # f_l is the same as LP objective function
        assert b_incoming, b_outgoing == compute_bandwidths(wL, delta0, g_i_inbound, g_i_outbound)
        test_lp(wL, delta0, g_i_inbound, g_i_outbound, alpha, C, n_tests=1000)

    new_bandwidth = f_l(wL, delta0, g_i_inbound, g_i_outbound, alpha)

    if verbose == True:
        print("New bandwidth: {}".format(new_bandwidth))

    n = len(delta)

    if new_bandwidth > gA:
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
    for it in range(n_tests):
        random_offsets = C * np.random.rand(n) - C / 2
        random_bandwidth = f_l(random_offsets, delta0, g_i_inbound, g_i_outbound, alpha)
        assert random_bandwidth <= optimal_bandwidth, """\n 
        Found better offsets for (L) after {} iterations: {} \n
        Optimal offsets: {} \n
        Optimal bandwidth: {} \n
        New random bandwidth: {}""".format(it, random_offsets, w_incoming, optimal_bandwidth, random_bandwidth)


def test_offsets(w_incoming, delta0, g_i_inbound, g_i_outbound, alpha, C, n_tests=1000):
    optimal_bandwidth = f_n(w_incoming, delta0, g_i_inbound, g_i_outbound, alpha)
    n = len(w_incoming)
    for it in range(n_tests):
        random_offsets = C * np.random.rand(n) - C / 2
        random_bandwidth = f_n(random_offsets, delta0, g_i_inbound, g_i_outbound, alpha)
        assert random_bandwidth <= optimal_bandwidth, """\n 
        Found better offsets for (N) after {} iterations: {} \n
        Optimal offsets: {} \n
        Optimal bandwidth: {} \n
        New random bandwidth: {}""".format(it, random_offsets, w_incoming, optimal_bandwidth, random_bandwidth)


def generate_arterial(n_intersections):
    """
    Generates a random set of intersection positions
    and speeds between intersections
    :param n_intersections:
    :type n_intersections:
    :return: positions, speeds, travel time
    :rtype: numpy array, numpy array, numpy array
    """
    positions = np.random.randint(1, 500, n_intersections)
    for i in range(1, n_intersections):
        positions[i] += positions[i - 1]
    speeds = np.random.randint(0, 100, n_intersections)
    travel_time = positions / speeds
    return positions, speeds, travel_time

def generate_arguments(alpha, n_intersections, C):
    g_i_inbound = [(0.7 * np.random.rand() + 0.2) * C for i in range(n_intersections)]
    g_i_outbound = [(0.7 * np.random.rand() + 0.2) * C for i in range(n_intersections)]
    delta = [(0.7 * np.random.rand()) * C for i in range(n_intersections)]

    kwargs = {
        'alpha': alpha,  # inbound weight
        'n_intersections': n_intersections,
        'C': C,  # seconds
        'g_i_inbound': g_i_inbound,  # portion of cycle
        'g_i_outbound': g_i_outbound,  # portion of cycle
        'delta': delta
    }
    return kwargs

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

def plot_solution(positions, g_i_inbound, g_i_outbound, w_relative, delta, travel_time, C, output_file='solution_plot.png'):
    """
    Plot green light slots
    :param positions:
    :type positions:
    :param g_i_inbound:
    :type g_i_inbound:
    :param g_i_outbound:
    :type g_i_outbound:
    :param w_relative:
    :type w_relative:
    :param delta:
    :type delta:
    :param travel_time:
    :type travel_time:
    :param C:
    :type C:
    :return:
    :rtype:
    """
    theta_inbound, theta_outbound = convert_offsets(w_relative, delta, travel_time, C)
    eps = positions[-1] / 1000

    # Plot the inbound slots
    it = 0
    for position, theta, g_i in zip(positions, theta_inbound, g_i_inbound):
        inf, sup = theta - g_i / 2, theta + g_i / 2
        inf_mod, sup_mod = modulo(inf, C), modulo(sup, C)
        if (inf, sup) == (inf_mod, sup_mod):
            plt.plot([inf, sup], [position + eps, position + eps], color='b')
        else:
            plt.plot([- C / 2, sup_mod], [position + eps, position + eps], color='b')
            plt.plot([inf_mod, C / 2], [position + eps, position + eps], color='b')

    # Plot the outbound slots
    for position, theta, g_i in zip(positions, theta_outbound, g_i_outbound):
        inf, sup = theta - g_i / 2, theta + g_i / 2
        inf_mod, sup_mod = modulo(inf, C), modulo(sup, C)
        if (inf, sup) == (inf_mod, sup_mod):
            plt.plot([inf, sup], [position - eps, position - eps], color='r')
        else:
            plt.plot([- C / 2, sup_mod], [position - eps, position - eps], color='r')
            plt.plot([inf_mod, C / 2], [position - eps, position - eps], color='r')

    plt.title("Optimal green light time slots for inbound (blue) and outbound (red) traffic")
    plt.ylabel("Position (feet)")
    plt.xlabel("Time (seconds)")
    plt.savefig(output_file)
    plt.close()

def test_alpha(n_points=20):
    alpha_range = np.linspace(0, 1, n_points)
    incoming_bandwidths = []
    outgoing_bandwidths = []
    bandwidths = []
    for alpha in alpha_range:
        kwargs['alpha'] = alpha
        b_incoming, b_outgoing, w_relative = solver(**kwargs, print_=False)
        incoming_bandwidths.append(b_incoming)
        outgoing_bandwidths.append(b_outgoing)
        bandwidths.append(b_incoming + b_outgoing)
    plt.plot(alpha_range, bandwidths, 'b', label='Total bandwidth')
    plt.plot(alpha_range, incoming_bandwidths, 'r', label='Incoming bandwidth')
    plt.plot(alpha_range, outgoing_bandwidths, 'g', label='Outgoing bandwidth')
    plt.title("Optimal Bandwidth and Inbound Weight alpha")
    plt.ylabel("Optimal Bandwidth")
    plt.xlabel("Inbound Weight")
    plt.legend()
    plt.savefig('bandwidth_and_alpha_for_{}_values'.format(n_points))
    plt.close()

def test_n_intersections(n_intersections_max=15, alpha=0.4, C=70):
    n_intersections_range = [k for k in range(1, n_intersections_max + 1)]
    incoming_bandwidths = []
    outgoing_bandwidths = []
    bandwidths = []

    kwargs = {
        'alpha': alpha,  # inbound weight
        'n_intersections': 0,
        'C': C,  # seconds
        'g_i_inbound': [],  # portion of cycle
        'g_i_outbound': [],  # portion of cycle
        'delta': []
    }

    for n_intersections in n_intersections_range:
        print("{} intersections".format(n_intersections))
        kwargs['n_intersections'] += 1
        kwargs['g_i_inbound'].append((0.7 * np.random.rand() + 0.2) * C)
        kwargs['g_i_outbound'].append((0.7 * np.random.rand() + 0.2) * C)
        kwargs['delta'].append((0.7 * np.random.rand()) * C)

        b_incoming, b_outgoing, w_relative = solver(**kwargs, print_=False)
        incoming_bandwidths.append(b_incoming)
        outgoing_bandwidths.append(b_outgoing)
        bandwidths.append(b_incoming + b_outgoing)
    plt.plot(n_intersections_range, bandwidths, 'b', label='Total bandwidth')
    plt.plot(n_intersections_range, incoming_bandwidths, 'r', label='Incoming bandwidth')
    plt.plot(n_intersections_range, outgoing_bandwidths, 'g', label='Outgoing bandwidth')
    plt.title("Optimal Bandwidth and Number of intersections")
    plt.ylabel("Optimal Bandwidth")
    plt.xlabel("Number of intersections")
    plt.legend()
    plt.savefig('bandwidth_and_n_intersections_for_{}_values'.format(n_intersections_max))
    plt.close()

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