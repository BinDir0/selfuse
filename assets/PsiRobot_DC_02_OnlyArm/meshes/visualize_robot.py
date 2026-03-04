import mujoco
import mujoco.viewer
import time

def main():
    # 1. 加载模型 (请确保文件名与你的场景文件名一致)
    model_path = "psi_robot_scene_transformed.xml" 
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 2. 启动被动可视化窗口
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # --- 控制参数：你可以修改下面的值来控制坐标系的可见性 ---
        # mjtFrame 的选项包括:
        # mujoco.mjtFrame.mjFRAME_NONE  (都不显示)
        # mujoco.mjtFrame.mjFRAME_BODY  (显示 Body 坐标系)
        # mujoco.mjtFrame.mjFRAME_JOINT (显示 Joint 坐标系)
        # mujoco.mjtFrame.mjFRAME_GEOM  (显示 Geom 坐标系)
        # mujoco.mjtFrame.mjFRAME_SITE  (显示 Site 坐标系)
        
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY # 默认显示 Body 坐标系
        
        # 还可以开启其他视觉标志
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False   # 是否显示质心
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True  # 是否显示关节轴
        
        print("可视化已启动。")
        print("提示：在窗口右侧面板的 'Rendering' 标签下也可以手动勾选 'Frame'。")

        # 3. 循环渲染
        while viewer.is_running():
            step_start = time.time()

            # 物理仿真步进（虽然当前重力为0，但为了交互建议保留）
            mujoco.mj_step(model, data)

            # 同步数据到视图
            viewer.sync()

            # 维持仿真频率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()