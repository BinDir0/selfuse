from typing import Dict
from egovla.policy.base_policy import BasePolicy

class BaseImageRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()
