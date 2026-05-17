#!/usr/bin/env python3
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class VirtualSmokeTestNode(Node):
    """Throwaway virtual harness for OpenPI no-RTC inference.

    It serves a fixed instruction to model_interface_node and monitors whether
    action topics are published from predicted chunks.
    """

    def __init__(self):
        super().__init__('virtual_smoke_test_node')
        self.declare_parameter('ui_service_port', 18080)
        self.declare_parameter('instruction', 'Pick up the test tube.')
        self.declare_parameter('mode', 'deploy')
        self.declare_parameter('duration_sec', 30.0)
        self.declare_parameter('min_arm_msgs', 3)
        self.declare_parameter('min_hand_msgs', 3)

        self.ui_service_port = int(self.get_parameter('ui_service_port').value)
        self.instruction = self.get_parameter('instruction').value
        self.mode = self.get_parameter('mode').value
        self.duration_sec = float(self.get_parameter('duration_sec').value)
        self.min_arm_msgs = int(self.get_parameter('min_arm_msgs').value)
        self.min_hand_msgs = int(self.get_parameter('min_hand_msgs').value)

        self.counts = {
            'arm': 0,
            'left_hand': 0,
            'right_hand': 0,
        }
        self.first_msg_time = None
        self.start_time = time.time()
        self._server = self._start_http_server()

        self.create_subscription(PoseArray, '/action/both_arms/wrist_poses', self._arm_cb, 10)
        self.create_subscription(PoseArray, '/action/left_hand/keypoints', self._left_hand_cb, 10)
        self.create_subscription(PoseArray, '/action/right_hand/keypoints', self._right_hand_cb, 10)
        self.create_timer(1.0, self._tick)

        self.get_logger().info(
            f"Virtual smoke test started: port={self.ui_service_port}, "
            f"instruction='{self.instruction}', duration={self.duration_sec}s"
        )

    def _start_http_server(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                if self.path == '/status':
                    self._send_json({'waiting': True})
                elif self.path == '/get_input':
                    self._send_json({'instruction': node.instruction, 'mode': node.mode})
                else:
                    self._send_json({'ok': True})

            def do_POST(self):
                if self.path.startswith('/play/'):
                    self.send_response(200)
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()

            def _send_json(self, payload):
                data = json.dumps(payload).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(('0.0.0.0', self.ui_service_port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    def _mark(self, key):
        self.counts[key] += 1
        if self.first_msg_time is None:
            self.first_msg_time = time.time()

    def _arm_cb(self, msg):
        if len(msg.poses) == 2:
            self._mark('arm')

    def _left_hand_cb(self, msg):
        if len(msg.poses) == 5:
            self._mark('left_hand')

    def _right_hand_cb(self, msg):
        if len(msg.poses) == 5:
            self._mark('right_hand')

    def _tick(self):
        elapsed = time.time() - self.start_time
        self.get_logger().info(
            f"virtual smoke progress: elapsed={elapsed:.1f}s, "
            f"arm={self.counts['arm']}, left_hand={self.counts['left_hand']}, right_hand={self.counts['right_hand']}"
        )
        if elapsed < self.duration_sec:
            return

        passed = (
            self.counts['arm'] >= self.min_arm_msgs
            and self.counts['left_hand'] >= self.min_hand_msgs
            and self.counts['right_hand'] >= self.min_hand_msgs
        )
        if passed:
            self.get_logger().info(f"VIRTUAL_SMOKE_PASS counts={self.counts}")
            os._exit(0)
        else:
            self.get_logger().error(f"VIRTUAL_SMOKE_FAIL counts={self.counts}")
            os._exit(1)

    def destroy_node(self):
        self._server.shutdown()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VirtualSmokeTestNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
