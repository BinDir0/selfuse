"""
Connector Configuration Parameters
Extracted from visualize_psirobot_with_rgbd_calib.py
"""

import numpy as np

# Left arm connector parameters (from visualize_psirobot_with_rgbd_calib.py line 864-872)
LEFT_CONNECTOR_OFFSET_1_1 = np.array([-0.0004094, 0.0000127, 0.0036907])
LEFT_CONNECTOR_RPY_1_1 = np.array([0.0, 0.0000038, 1.5707998])
LEFT_CONNECTOR_OFFSET_1_2 = np.array([0.0, 0.0, 0.0036596])
LEFT_CONNECTOR_RPY_1_2 = np.array([0.0, 0.0, 0.0])

# Right arm connector parameters (from visualize_psirobot_with_rgbd_calib.py line 883-891)
RIGHT_CONNECTOR_OFFSET_2_1 = np.array([-0.0004101, -0.0000127, 0.0036911])
RIGHT_CONNECTOR_RPY_2_1 = np.array([0.0, -0.0000037, -1.5707928])
RIGHT_CONNECTOR_OFFSET_2_2 = np.array([0.0, -0.0000002, 0.0036597])
RIGHT_CONNECTOR_RPY_2_2 = np.array([0.0000001, -0.0000001, 3.1415925])

# Connector total height (user's latest value: 0.0105m = 10.5mm)
# From visualize_psirobot_with_rgbd_calib.py line 878
CONNECTOR_HEIGHT = 0.011  # meters

# TCP rotation: Z-axis 180 degrees
# From visualize_psirobot_with_rgbd_calib.py line 109-113
TCP_ROTATION = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0,  1.0]
])
