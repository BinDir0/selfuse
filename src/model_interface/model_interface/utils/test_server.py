from websocket_client import WebsocketClientPolicy
import numpy as np

policy = WebsocketClientPolicy(host="81.68.132.224", port=18025)
response = policy.infer({"image": np.random.randint(0, 255, (1, 480, 640, 3)).astype(np.uint8), "depth_image": np.random.randint(0, 10000, (1, 480, 640, 1)).astype(np.uint16), "camera_intrinsics": np.eye(3), "instruction": "test", "states": np.random.rand(1, 48).astype(np.float32)})
print(response["pred_actions"].shape)