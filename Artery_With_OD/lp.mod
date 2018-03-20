# Parameters
param N symbolic;
param C symbolic;
param alpha symbolic;
param g_incoming {i in 1..N};
param g_outgoing {i in 1..N};
param delta {i in 1..N};

# Variables
var b_incoming >= 0;
var b_outgoing >= 0;
var w {i in 1..N};

# Objective function
maximize bandwidth: alpha * b_incoming + (1 - alpha) * b_outgoing;

# Constraints
subject to b_incoming_pos: b_incoming >= 0;
subject to b_outgoing_pos: b_outgoing >= 0;
subject to w_incoming_lower_bound {i in 1..N}: w[i] >= - C / 2;
subject to w_outgoing_upper_bound {i in 1..N}: w[i] <= C / 2;
subject to b_incoming_inferior_gmin {i in 1..N}: b_incoming <= g_incoming[i];
subject to b_outgoing_inferior_gmin {i in 1..N}: b_outgoing <= g_outgoing[i];
subject to incoming_b_g {i in 2..N, j in 2..N}: b_incoming + (w[i] - w[j]) <= (g_incoming[i] + g_incoming[j]) / 2;
subject to outgoing_b_g {i in 2..N, j in 2..N}: b_outgoing + (w[i] - w[j]) <= (g_outgoing[i] + g_outgoing[j]) / 2 - (delta[i] - delta[j]);
subject to incoming_b_g1 {i in 2..N}: b_incoming + w[i] <= (g_incoming[1] + g_incoming[i]) / 2;
subject to incoming_b_g1_ {i in 2..N}: b_incoming - w[i] <= (g_incoming[1] + g_incoming[i]) / 2;
subject to outgoing_b_g1 {i in 2..N}: b_outgoing + w[i] <= (g_outgoing[1] + g_outgoing[i]) / 2 - (delta[i] - delta[1]);
subject to outgoing_b_g1_ {i in 2..N}: b_outgoing - w[i] <= (g_outgoing[1] + g_outgoing[i]) / 2 + (delta[i] - delta[1]);
