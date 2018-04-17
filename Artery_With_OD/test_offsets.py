# -*- coding: utf-8 -*-
"""
Contributors:
    - Auriane Blarre
"""
from lp import get_alpha, get_delta, solve_pulse

if __name__ == "__main__":
    alpha_test = 0.5
    n_intersections_test = 9
    C_test = 100
    g_i_inbound_test = [0.7 * C_test, 0.65 * C_test, 0.50 * C_test, 0.51 * C_test, 0.6 * C_test, 0.67 * C_test, 0.4 * C_test, 0.4 * C_test, 0.7 * C_test]
    g_i_outbound_test = g_i_inbound_test
    delta_test = [0 for i in range(n_intersections_test)]
    travel_time_incoming_test = [371 / 23.7, 371 / 23.7, 390 / 23.7, 276 / 23.7, 472 / 23.7, 364 / 23.7, 968 / 23.7,
                                 377 / 23.7]
    travel_time_outgoing_test = [-x for x in travel_time_incoming_test]
    solve_pulse(alpha_test, n_intersections_test, C_test, g_i_inbound_test, g_i_outbound_test, delta_test,
                travel_time_incoming_test, travel_time_outgoing_test, verbose=True,
                test=True)