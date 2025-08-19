import torch
import numpy as np
import os

import mujoco
from dex_retargeting.retargeting_config import RetargetingConfig

from egovla.model.retargeting.base_retargeting import BaseRetargeting

class OptRetargeting(BaseRetargeting):
    """
    A retargeting processor for the ruihand6y dexterous hand.

    This class encapsulates the configuration and calling logic of the dex-retargeting library,
    making it possible to run independently of any simulation environment (such as MuJoCo).
    """
    def __init__(self, 
        retargeting_type, 
        model_config
    ):

        self.retargeting_type = retargeting_type    
        self.scaling_factor = model_config["scaling_factor"] 
        print(f"🎯 Scaling factor: {self.scaling_factor}")

        print(f"Loading model: {model_config['model_path']}")
        try:
            self.m = mujoco.MjModel.from_xml_path(model_config['model_path'])
            self.d = mujoco.MjData(self.m)
            print("Model loaded successfully!")
            print(f"Total DOF: {self.m.nv}, Joints: {self.m.njnt}")
            print("Note: RuiYan Hand has different DOF structure than PSI Hand:")
            print("Thumb: 4 DOF, other fingers: 3 DOF each (Total 16 DOF)")
        except Exception as e:
            print(f"Error: Model loading failed. Please check the path and XML file.")
            print(f"Detailed error: {e}")
            self.m = None; self.d = None; return

        self.urdf_path = model_config['urdf_path']
        # Gravity and collisions are disabled in the XML, no need to set in Python

        self.controllable_joints = {}
        self.joint_name_to_qpos = {}
        
        # vector retargeting system
        self.retargeting = None
        self.hand_meta = model_config['hand_meta']
        self.keypoints_dict = model_config['keypoints_dict']
        print(f"Keypoints dict: {self.keypoints_dict}")
        
        self._identify_controllable_joints()
        self._create_joint_name_map()
        self._setup_vector_retargeting()

    def _identify_controllable_joints(self):
        print("\nIdentifying controllable joints...")
        for i in range(self.m.njnt):
            if self.m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                joint_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i)
                qpos_addr = self.m.jnt_qposadr[i]
                self.controllable_joints[qpos_addr] = {"name": joint_name, "range": self.m.jnt_range[i]}
        print("==========================================\n")

    def _create_joint_name_map(self):
        """build a mapping from joint name to qpos address, for later lookup by name"""
        for qpos_addr, joint_info in self.controllable_joints.items():
            self.joint_name_to_qpos[joint_info['name']] = qpos_addr
        print("Joint name mapping created.")
        print("==========================================\n")

    def _setup_vector_retargeting(self):
        """setup vector retargeting system"""
        print("Setting up vector retargeting system...")
        
        try:
            # create Vector retargeting config (more suitable for real-time teleoperation, no sudden switch)
            # we only implement vector retargeting for now
            config_dict = {
                "type": self.retargeting_type,  # Vector retargeting for smooth real-time control
                "urdf_path": self.urdf_path,
                
                # hand link structure - match fingertip sites in XML
                "target_origin_link_names": [
                    v["origin_link_name"] for _, v in self.keypoints_dict.items()
                ],
                "target_task_link_names": [
                    v["link_name"] for _, v in self.keypoints_dict.items()
                ],
                
                # Human indices for vector calculation - follows AnyTeleop format
                # [origin_indices, task_indices] where each vector goes from origin[i] to task[i]
                "target_link_human_indices": [
                    [0] * len(self.keypoints_dict),  # All vectors start from wrist (human joint index 0)  
                    [v["mano_id"] for _, v in self.keypoints_dict.items()]  # End at fingertips (MANO indices)
                ],
                
                # Explicitly specify target joints (exclude mimic joints)
                "target_joint_names": [
                    [joint for _, v in self.hand_meta.items() for joint in v["primary_joints"]]
                ],
                
                # Scaling and filtering
                "scaling_factor": self.scaling_factor,
                "low_pass_alpha": 0.2,  # Standard smoothing from AnyTeleop (0.1=smooth, 0.9=responsive)
            }
            
            # build retargeting system
            config = RetargetingConfig.from_dict(config_dict)
            self.retargeting = config.build()
            
            print("✓ Vector retargeting system created successfully!")
            print(f"✓ Robot DOF: {len(self.retargeting.joint_names)}")
            print(f"✓ Joint order: {self.retargeting.joint_names[:8]}...")  # Show first 8
            print("==========================================\n")
            
        except Exception as e:
            print(f"✗ Error: Failed to create Vector retargeting system. Details: {e}")
            self.retargeting = None
            print("==========================================\n")

    def _convert_keypoints_to_vector_format(self, keypoints_pos):
        """
        convert keypoints position to vector format for Vector retargeting
        
        Args:
            keypoints_pos (np.ndarray or torch.Tensor): 
            [N, 3] positions of all mano hand keypoints in wrist frame
            
        Returns:
            np.ndarray: Vector retargeting expected vector format
        """
        if isinstance(keypoints_pos, torch.Tensor):
            keypoints_pos = keypoints_pos.cpu().numpy()
        N = keypoints_pos.shape[0]

        try:
            # important fix: get current robot's base_link position as "wrist" position
            # ensure the coordinate system we use is consistent with the optimizer
            mujoco.mj_forward(self.m, self.d)
            
            # build vectors: from robot base to target fingertip position
            vectors = []
            for name, info in self.keypoints_dict.items():
                mano_id = info["mano_id"]
                if mano_id >= 0 and mano_id < N:
                    base_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, info["origin_link_name"])
                    if base_body_id != -1:
                        robot_base_pos = self.d.xpos[base_body_id].copy()
                    else:
                        robot_base_pos = np.array([0.0, 0.0, 0.0])  # fallback
                    # calculate vector from robot base to target fingertip    
                    vec = keypoints_pos[mano_id] - robot_base_pos
                    vectors.append(vec)
                else:
                    print(f"Warning: missing {name} position data")
                    return None
                    
            target_vectors = np.array(vectors, dtype=np.float32)
                
            return target_vectors
            
        except Exception as e:
            print(f"Error: Failed to convert fingertip position data: {e}")
            return None

    def _apply_joint_angles_to_model(self, joint_angles):
        """
        apply joint angles outputted by vector to MuJoCo model
        
        Args:
            joint_angles (numpy.ndarray): joint angles outputted by vector
        """
        if joint_angles is None or self.retargeting is None:
            return False
            
        try:
            # vector's joint_names order may be different from MuJoCo's qpos order
            # need to build a mapping
            
            for i, joint_name in enumerate(self.retargeting.joint_names):
                if i < len(joint_angles):
                    # find corresponding joint in MuJoCo model
                    joint_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id != -1:
                        # get qpos address
                        qpos_addr = self.m.jnt_qposadr[joint_id]
                        # apply joint angle (with constraint check)
                        if qpos_addr < len(self.d.qpos):
                            joint_range = self.m.jnt_range[joint_id]
                            clamped_angle = np.clip(joint_angles[i], joint_range[0], joint_range[1])
                            self.d.qpos[qpos_addr] = clamped_angle
                            
            # forward dynamics calculation
            mujoco.mj_forward(self.m, self.d)
            return True
            
        except Exception as e:
            print(f"Error: Failed to apply joint angles: {e}")
            return False
    
    def forward(self, keypoints_pos: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            keypoints_pos: [B, E, N, 3] positions of hand keypoints in wrist frame
            N: number of keypoints per hand, including all mano hand keypoints
        Returns:
            qpos: [B, E, rdof] qpos of the hand dofs
        '''
        B, E, N, _ = keypoints_pos.shape
        qpos_all = []
        for i in range(B):
            for j in range(E):
                target_vectors = self._convert_keypoints_to_vector_format(keypoints_pos[i, j])

                # Vector retargeting for manually set targets
                qpos = self.retargeting.retarget(target_vectors)
                self._apply_joint_angles_to_model(qpos)
                qpos_all.append(qpos)

        qpos_all = np.array(qpos_all).reshape(B, E, -1)
        return torch.tensor(qpos_all).to(keypoints_pos.device)
