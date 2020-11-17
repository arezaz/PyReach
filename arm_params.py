import numpy as np

# arm parameters
l1, l2 = 0.35, 0.36
m1, m2 = 2.1, 1.65
l1_c, l2_c = 0.152, 0.2465
J1, J2 = 0.0264, 0.0472

Kp = 2*np.array([[32, 16], [16, 21]])
Kd = 2*np.array([[5, 3],[3, 4]])

dt = 0.01

