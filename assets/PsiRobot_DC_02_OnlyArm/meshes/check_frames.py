import mujoco
import mujoco.viewer
import time
import os

def visualize_model(xml_name, robot_file_to_use):
    print(f"\n正在加载场景: {xml_name} (使用机器人模型: {robot_file_to_use})")
    
    with open(xml_name, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    modified_xml = xml_content.replace('xiaozi.xml', robot_file_to_use)
    
    model = mujoco.MjModel.from_xml_string(modified_xml)
    data = mujoco.MjData(model)

    # 1. 归位到 home 关键帧
    try:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key_id != -1:
            data.qpos[:] = model.key_qpos[key_id]
            mujoco.mj_forward(model, data)
            print("  -> 已成功切换到 'home' 关键帧姿态")
    except Exception as e:
        print(f"  -> 未找到 home 关键帧: {e}")

    # 2. 启动可视化
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # --- 修正处：vopt 改为 opt ---
        
        # mjFRAME_SITE (4): 显示 site 坐标系
        # 如果你想看 body 坐标系，可以用 mujoco.mjtFrame.mjFRAME_BODY
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE 
        
        # 显示 site 的名称标签
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        
        # 开启透明模式，方便看清内部坐标轴
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True 
        
        print(f"当前预览: {robot_file_to_use}")
        print("提示: 红色=X轴, 绿色=Y轴, 蓝色=Z轴")
        print("请在窗口观察轴向。关闭窗口以切换下一个模型。")

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

# 依次查看两个模型
models_to_check = ["xiaozi.xml", "xiaozi_transformed.xml"]

for robot_file in models_to_check:
    if os.path.exists(robot_file):
        visualize_model("psi_robot_scene.xml", robot_file)
    else:
        print(f"文件不存在: {robot_file}")