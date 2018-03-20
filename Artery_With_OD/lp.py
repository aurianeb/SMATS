# -*- coding: utf-8 -*-
"""
Contributors:
    - Auriane Blarre
"""

from amplpy import AMPL, Environment
import numpy as np
import matplotlib.pyplot as plt

# AMPL path
path = '/Users/aurianeblarre/Documents/Berkeley/ENGIN296/Projet/amplide.macosx64'

def solver(alpha, n_intersections, C, g_i_inbound, g_i_outbound, delta, print_=True):
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
    w_relative = ampl.getVariable('w').getValues()
    w_relative = [w[1] for w in w_relative]

    if print_ == True:
        print("New objective value: {}".format(bandwidth))
        print("Incoming bandwidth: {}".format(b_incoming))
        print("Outgoing bandwidth: {}".format(b_outgoing))
        print("Incoming offsets: {}".format(w_relative))

    # return bandwidths and offset values
    return b_incoming, b_outgoing, w_relative

def convert_offsets(w_relative, delta, travel_time, C):
    """
    Converts relative offsets to absolute
    :param w_relative:
    :type w_relative:
    :param delta:
    :type delta:
    :param travel_time:
    :type travel_time:
    :param theta_1:
    :type theta_1:
    :return:
    :rtype:
    """
    theta_1 = travel_time[0] + w_relative[0]
    theta_inbound = [theta_1]
    for i, w in enumerate(w_relative[1:]):
        t = modulo(theta_1 + w - w_relative[0] + np.sum(travel_time[i:]), C)
        theta_inbound.append(t)
    theta_outbound = [modulo(x + y, C) for x, y in zip(theta_inbound, delta)]
    return theta_inbound, theta_outbound

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
    # Parameters
    kwargs = {
        'alpha': 0.4, # inbound weight
        'n_intersections': 3,
        'C': 70, # seconds
        'g_i_inbound': [0.5 * 70, 0.4 * 70, 0.5 * 70], # portion of cycle
        'g_i_outbound': [0.7 * 70, 0.6 * 70, 0.4 * 70], # portion of cycle
        'delta': [0.5 * 70, 0.2 * 70, 0.1 * 70]
    }

    # Solve the problem with kwargs arguments
    solver(**kwargs)

    # Run test function of alpha
    test_alpha()

    # run test funtion of the number of intersections
    test_n_intersections()

    # plot a sample solution
    # kwargs = generate_arguments(alpha, n_intersections, C)
    # positions, speeds, travel_time = generate_arterial(kwargs['n_intersections'])
    # w_relative =
    # plot_solution(positions, g_i_inbound, g_i_outbound, w_relative, delta, travel_time, C)