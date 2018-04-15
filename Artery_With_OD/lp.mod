# Parameters
param N symbolic;
param C symbolic;
param alpha symbolic;
param g_incoming {i in 1..N};
param g_outgoing {i in 1..N};
param delta {i in 1..N};

# Variables
var b_incoming;
var b_outgoing;
var w {i in 1..N};

# Objective function
maximize bandwidth: alpha * b_incoming + (1 - alpha) * b_outgoing;

# Constraints
subject to incoming_b_g {i in 1..N, j in 1..N}: b_incoming + (w[i] - w[j]) <= (g_incoming[i] + g_incoming[j]) / 2;
subject to outgoing_b_g {i in 1..N, j in 1..N}: b_outgoing + (w[i] - w[j]) <= (g_outgoing[i] + g_outgoing[j]) / 2 - (delta[i] - delta[j]);
subject to outgoing_b_g_ {i in 2..N, j in 2..N}: b_outgoing + (w[i] - w[j]) <= (g_outgoing[i] + g_outgoing[j]) / 2 - (delta[i] + delta[j]);
subject to incoming_b_g1 {i in 2..N}: b_incoming + w[i] <= (g_incoming[1] + g_incoming[i]) / 2;
subject to incoming_b_g1_ {i in 2..N}: b_incoming - w[i] <= (g_incoming[1] + g_incoming[i]) / 2;
subject to outgoing_b_g1 {i in 2..N}: b_outgoing + w[i] <= (g_outgoing[1] + g_outgoing[i]) / 2 - (delta[i] - delta[1]);
subject to outgoing_b_g1_ {i in 2..N}: b_outgoing - w[i] <= (g_outgoing[1] + g_outgoing[i]) / 2 + (delta[i] - delta[1]);
