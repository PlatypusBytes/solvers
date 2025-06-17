import numpy as np

M = np.array([[2, 0], [0, 1]])
K = np.array([[6, -2], [-2, 4]])
C = np.array([[0, 0], [0, 0]])
F = np.zeros((2, 13))
F[1, :] = 10

n_steps = 13
t_step = 0.28
t_total = n_steps * t_step

u = np.zeros((n_steps, 2))
v = np.zeros((n_steps, 2))
a = np.zeros((n_steps, 2))
a[0, :] = [0, 10]
R = np.zeros((n_steps, 2))

p = 0.54
q1 = (1 - 2 * p) / (2 * p * ( 1 - p))
q2 = 0.5 - p * q1
q0 = -q1 - q2 + 0.5
a0 = p * t_step
a1 = 0.5 * (p * t_step) ** 2
a2 = a0 / 2
a3 = (1 - p) * t_step
a4 = 0.5 * ((1-p)*t_step) ** 2
a5 = q0 * a3
a6 = (0.5 + q1) * a3
a7 = q2 * a3


for t in range(n_steps-1):
    u_tp = u[t, :] + a0 * v[t, :] + a1 * a[t, :]
    R_tp = ( 1 - p) * F[:, t].T + p * F[:, t+1].T
    R_tp = R_tp - K.dot(u_tp) - C.dot(v[t, :] + a0 * a[t, :])

    a_tp = np.linalg.inv(M).dot(R_tp)
    v_tp = v[t, :] + a2 * (a[t, :] + a_tp)

    u[t+1, :] = u_tp + a3 * v_tp + a4 * a_tp
    R[t + 1, :] = F[:, t+1].T - K.dot(u[t+1, :]) - C.dot(v_tp + a3 * a_tp)

    a[t+1, :] = np.linalg.inv(M).dot(R[t+1,:])
    v[t+1, :] = v_tp + a5 * a[t, :] + a6 * a_tp + a7 * a[t+1, :]

print(u)