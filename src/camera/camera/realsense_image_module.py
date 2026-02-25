#!/usr/bin/env python3
"""
RealSense相机图像采集模块
用于从RealSense相机获取RGB-D图像
"""

import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseImage:
    """RealSense相机图像采集类"""
    
    def __init__(self, SN_number=None, width=640, height=480, fps=30):
        """
        初始化RealSense相机
        
        Args:
            SN_number: 相机序列号，如果为None则使用默认设备
            width: 图像宽度，默认640
            height: 图像高度，默认480
            fps: 帧率，默认30
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 如果指定了序列号，则使用特定设备
        if SN_number:
            self.config.enable_device(SN_number)
        
        # 配置深度流和彩色流
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        
        # 启动pipeline
        self.pipeline.start(self.config)
        
        # 创建对齐对象，将深度图对齐到彩色图
        self.align_to = rs.stream.color
        self.aligner = rs.align(self.align_to)
        
        # 缓存变量
        self.depth_image_np = None
        self.color_image_np = None
        
        # 稳定相机输出
        self._stabilize_camera()
    
    def _stabilize_camera(self):
        """稳定相机输出，丢弃前几帧"""
        for _ in range(10):
            self.capture_rgb_depth_frames()
    
    def capture_rgb_depth_frames(self):
        """
        同时捕获RGB和深度图像
        
        Returns:
            tuple: (rgb_image, depth_image)
                - rgb_image: RGB图像 (H, W, 3) numpy array
                - depth_image: 深度图像 (H, W) numpy array, uint16格式
        """
        try:
            # 等待新的一帧
            frames = self.pipeline.wait_for_frames()
            
            # 对齐深度图到彩色图
            aligned_frames = self.aligner.process(frames)
            
            # 获取对齐后的深度帧和彩色帧
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if depth_frame and color_frame:
                # 转换为numpy数组
                self.depth_image_np = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
                self.color_image_np = np.asanyarray(color_frame.get_data())[:, :, :3]
                
                return self.color_image_np, self.depth_image_np
            
            return None, None
            
        except Exception as e:
            print(f"捕获图像时出错: {e}")
            return None, None
    
    def capture_rgb_frame(self):
        """
        仅捕获RGB图像
        
        Returns:
            numpy.ndarray: RGB图像 (H, W, 3)
        """
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if color_frame:
                self.color_image_np = np.asanyarray(color_frame.get_data())[:, :, :3]
                return self.color_image_np
            
            return None
            
        except Exception as e:
            print(f"捕获RGB图像时出错: {e}")
            return None
    
    def capture_depth_frame(self):
        """
        仅捕获深度图像
        
        Returns:
            numpy.ndarray: 深度图像 (H, W), uint16格式
        """
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.aligner.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            
            if depth_frame:
                self.depth_image_np = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
                return self.depth_image_np
            
            return None
            
        except Exception as e:
            print(f"捕获深度图像时出错: {e}")
            return None
    
    def close(self):
        """关闭相机并释放资源"""
        try:
            self.pipeline.stop()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"关闭相机时出错: {e}")


if __name__ == "__main__":
    """测试代码"""
    camera = RealSenseImage()
    
    try:
        while True:
            rgb, depth = camera.capture_rgb_depth_frames()
            
            if rgb is not None and depth is not None:
                # 显示RGB图像
                cv2.imshow('RGB', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                
                # 显示深度图像（归一化以便显示）
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                cv2.imshow('Depth', depth_colormap)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        camera.close()

