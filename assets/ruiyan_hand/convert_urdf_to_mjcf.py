"""
将 RuiYan Hand URDF 转换为 MuJoCo XML (MJCF) 格式

手动创建MJCF，确保包含所有visual mesh（类似xiaozi.xml的方式）
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import re


def parse_urdf_structure(urdf_path):
    """解析URDF文件，提取结构信息"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # 提取所有link信息
    links = {}
    for link in root.findall('.//link'):
        link_name = link.get('name')
        links[link_name] = {
            'visual_mesh': None,
            'visual_origin_xyz': '0 0 0',
            'visual_origin_rpy': '0 0 0',
            'inertial': None
        }
        
        # 提取visual mesh和origin
        visual = link.find('visual')
        if visual is not None:
            mesh_elem = visual.find('.//mesh')
            if mesh_elem is not None:
                filename = mesh_elem.get('filename')
                if filename and 'visuals.obj' in filename:
                    # 只保留文件名
                    links[link_name]['visual_mesh'] = filename.replace('meshes/', '')
            
            # 提取visual的origin（mesh相对于link的偏移）
            origin_elem = visual.find('origin')
            if origin_elem is not None:
                links[link_name]['visual_origin_xyz'] = origin_elem.get('xyz', '0 0 0')
                links[link_name]['visual_origin_rpy'] = origin_elem.get('rpy', '0 0 0')
        
        # 提取惯性信息
        inertial = link.find('inertial')
        if inertial is not None:
            mass_elem = inertial.find('mass')
            if mass_elem is not None:
                links[link_name]['inertial'] = mass_elem.get('value')
    
    # 提取所有joint信息
    joints = {}
    for joint in root.findall('.//joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        
        if joint_type in ['revolute', 'continuous']:
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            
            # 提取关节限制
            limit_elem = joint.find('limit')
            if limit_elem is not None:
                lower = float(limit_elem.get('lower', '0'))
                upper = float(limit_elem.get('upper', '3.14'))
            else:
                lower, upper = 0, 3.14
            
            # 提取origin
            origin_elem = joint.find('origin')
            if origin_elem is not None:
                xyz = origin_elem.get('xyz', '0 0 0')
                rpy = origin_elem.get('rpy', '0 0 0')
            else:
                xyz, rpy = '0 0 0', '0 0 0'
            
            # 提取axis
            axis_elem = joint.find('axis')
            if axis_elem is not None:
                axis = axis_elem.get('xyz', '0 0 1')
            else:
                axis = '0 0 1'
            
            # 提取mimic信息
            mimic_elem = joint.find('mimic')
            mimic_joint = None
            mimic_multiplier = 1.0
            mimic_offset = 0.0
            if mimic_elem is not None:
                mimic_joint = mimic_elem.get('joint')
                mimic_multiplier = float(mimic_elem.get('multiplier', '1.0'))
                mimic_offset = float(mimic_elem.get('offset', '0.0'))
            
            joints[joint_name] = {
                'type': joint_type,
                'parent': parent,
                'child': child,
                'lower': lower,
                'upper': upper,
                'xyz': xyz,
                'rpy': rpy,
                'axis': axis,
                'mimic_joint': mimic_joint,
                'mimic_multiplier': mimic_multiplier,
                'mimic_offset': mimic_offset
            }
    
    return links, joints


def rpy_to_quat(rpy_str):
    """将RPY字符串转换为四元数"""
    rpy = [float(x) for x in rpy_str.split()]
    from scipy.spatial.transform import Rotation as R
    rot = R.from_euler('xyz', rpy, degrees=False)
    quat = rot.as_quat(scalar_first=True)  # wxyz
    return f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"


def create_mjcf_manually(urdf_path, output_path, hand_type='left'):
    """手动创建MJCF文件（类似xiaozi.xml）"""
    urdf_path = Path(urdf_path)
    output_path = Path(output_path)
    
    # 输出到meshes目录（和OBJ文件同级）
    meshes_dir = urdf_path.parent / 'meshes'
    output_in_meshes = meshes_dir / output_path.name
    
    print(f"[INFO] 解析URDF: {urdf_path}")
    links, joints = parse_urdf_structure(urdf_path)
    
    print(f"[INFO] 找到 {len(links)} 个link, {len(joints)} 个joint")
    
    # 创建MJCF根元素
    mujoco_elem = ET.Element('mujoco', model=urdf_path.stem)
    
    # 添加compiler
    ET.SubElement(mujoco_elem, 'compiler', angle='radian')
    
    # 添加option（禁用重力和碰撞，专注于运动学）
    option_elem = ET.SubElement(mujoco_elem, 'option', gravity='0 0 0')
    ET.SubElement(option_elem, 'flag', contact='disable')
    
    # 添加asset（声明所有mesh）
    asset_elem = ET.SubElement(mujoco_elem, 'asset')
    for link_name, link_info in links.items():
        if link_info['visual_mesh']:
            mesh_name = link_info['visual_mesh'].replace('.obj', '')
            ET.SubElement(asset_elem, 'mesh', 
                         name=mesh_name,
                         content_type='model/obj',
                         file=link_info['visual_mesh'])
    
    print(f"[INFO] 已添加 {len([l for l in links.values() if l['visual_mesh']])} 个mesh到asset")
    
    # 创建worldbody
    worldbody_elem = ET.SubElement(mujoco_elem, 'worldbody')
    
    # 递归构建body树
    def build_body_tree(parent_link, parent_elem):
        # 找到所有以parent_link为parent的joint
        child_joints = {jname: jinfo for jname, jinfo in joints.items() if jinfo['parent'] == parent_link}
        
        for joint_name, joint_info in child_joints.items():
            child_link = joint_info['child']
            child_link_info = links.get(child_link, {})
            
            # 创建body元素
            body_attribs = {'name': child_link}
            
            # 添加位置和姿态
            if joint_info['xyz'] != '0 0 0':
                body_attribs['pos'] = joint_info['xyz']
            if joint_info['rpy'] != '0 0 0':
                body_attribs['quat'] = rpy_to_quat(joint_info['rpy'])
            
            body_elem = ET.SubElement(parent_elem, 'body', **body_attribs)
            
            # 添加inertial
            mass = child_link_info.get('inertial', '0.001')
            ET.SubElement(body_elem, 'inertial',
                         pos='0 0 0',
                         mass=mass,
                         diaginertia='0.001 0.001 0.001')
            
            # 添加joint
            if joint_info['type'] in ['revolute', 'continuous']:
                joint_attribs = {
                    'name': joint_name,
                    'pos': '0 0 0',
                    'axis': joint_info['axis'],
                    'range': f"{joint_info['lower']} {joint_info['upper']}",
                    'actuatorfrcrange': '-3.40282e+38 3.40282e+38'
                }
                ET.SubElement(body_elem, 'joint', **joint_attribs)
            
            # 添加visual mesh geom（使用visual的origin）
            if child_link_info.get('visual_mesh'):
                mesh_name = child_link_info['visual_mesh'].replace('.obj', '')
                geom_attribs = {
                    'type': 'mesh',
                    'mesh': mesh_name
                }
                
                # 添加visual的origin（mesh相对于link的偏移）
                visual_xyz = child_link_info.get('visual_origin_xyz', '0 0 0')
                visual_rpy = child_link_info.get('visual_origin_rpy', '0 0 0')
                
                if visual_xyz != '0 0 0':
                    geom_attribs['pos'] = visual_xyz
                if visual_rpy != '0 0 0':
                    geom_attribs['quat'] = rpy_to_quat(visual_rpy)
                
                ET.SubElement(body_elem, 'geom', **geom_attribs)
            
            # 递归处理子节点
            build_body_tree(child_link, body_elem)
    
    # 找到根link（没有parent的link）
    root_links = set(links.keys())
    for joint_info in joints.values():
        if joint_info['child'] in root_links:
            root_links.remove(joint_info['child'])
    
    # 从每个根link开始构建
    for root_link in root_links:
        # 为root link添加visual（如果有）
        root_link_info = links.get(root_link, {})
        if root_link_info.get('visual_mesh'):
            mesh_name = root_link_info['visual_mesh'].replace('.obj', '')
            geom_attribs = {
                'type': 'mesh',
                'mesh': mesh_name
            }
            
            # 添加root link的visual origin
            visual_xyz = root_link_info.get('visual_origin_xyz', '0 0 0')
            visual_rpy = root_link_info.get('visual_origin_rpy', '0 0 0')
            
            if visual_xyz != '0 0 0':
                geom_attribs['pos'] = visual_xyz
            if visual_rpy != '0 0 0':
                geom_attribs['quat'] = rpy_to_quat(visual_rpy)
            
            ET.SubElement(worldbody_elem, 'geom', **geom_attribs)
        
        # 继续构建子树
        build_body_tree(root_link, worldbody_elem)
    
    # 添加指尖site
    add_fingertip_sites(worldbody_elem, hand_type)
    
    # 添加mimic约束（equality）
    add_mimic_constraints(mujoco_elem, joints)
    
    # 添加keyframe
    keyframe_elem = ET.SubElement(mujoco_elem, 'keyframe')
    # 计算关节数
    num_joints = len([j for j in joints.values() if j['type'] in ['revolute', 'continuous']])
    qpos = ' '.join(['0'] * num_joints)
    ET.SubElement(keyframe_elem, 'key', name='home', qpos=qpos)
    
    # 格式化并保存
    indent_xml(mujoco_elem)
    tree = ET.ElementTree(mujoco_elem)
    tree.write(output_in_meshes, encoding='utf-8', xml_declaration=True)
    
    print(f"[SUCCESS] ✅ 已生成MJCF: {output_in_meshes}")
    print(f"[INFO] MJCF文件位于meshes目录，与OBJ文件同级")
    
    return output_in_meshes


def add_fingertip_sites(worldbody_elem, hand_type):
    """添加指尖site定义（使用URDF中fixed joint的真实偏移）"""
    hand_prefix = 'hand1' if hand_type == 'left' else 'hand2'
    side_prefix = hand_type
    
    # 指尖位置：从URDF的fixed joint origin提取
    if hand_type == 'left':
        fingertip_bodies = [
            (f'{hand_prefix}_link_1_3', 'thumb', '1 0 0 1', '0.0348893 0.0312583 0.0000002'),
            (f'{hand_prefix}_link_2_2', 'index', '0 1 0 1', '0.042506 0.0155105 -0.0003185'),
            (f'{hand_prefix}_link_3_2', 'middle', '0 0 1 1', '0.0412144 0.016 0.0000365'),
            (f'{hand_prefix}_link_4_2', 'ring', '1 1 0 1', '0.0414819 0.0158259 0.0002174'),
            (f'{hand_prefix}_link_5_2', 'pinky', '1 0 1 1', '0.0415203 0.0157445 0.0002218'),
        ]
    else:  # right hand
        fingertip_bodies = [
            (f'{hand_prefix}_link_1_3', 'thumb', '1 0 0 1', '0.0289525 0.028477 0.'),
            (f'{hand_prefix}_link_2_2', 'index', '0 1 0 1', '-0.0416063 -0.0157651 0.0000874'),
            (f'{hand_prefix}_link_3_2', 'middle', '0 0 1 1', '-0.0426192 -0.015904 0.0000652'),
            (f'{hand_prefix}_link_4_2', 'ring', '1 1 0 1', '-0.0437206 -0.0156348 0.0001176'),
            (f'{hand_prefix}_link_5_2', 'pinky', '1 0 1 1', '-0.0404732 -0.0155 0.0003224'),
        ]
    
    for body_name, finger_name, color, tip_offset in fingertip_bodies:
        body = find_body_by_name(worldbody_elem, body_name)
        if body is not None:
            site_name = f'{side_prefix}_{finger_name}_tip'
            ET.SubElement(body, 'site',
                         name=site_name,
                         pos=tip_offset,
                         size='0.0015',
                         rgba=color,
                         type='sphere')
            print(f"  ✓ 添加 site: {site_name} at {tip_offset}")


def add_mimic_constraints(mujoco_elem, joints_data):
    """添加mimic约束（MuJoCo的equality约束）"""
    # 查找有mimic定义的关节
    mimic_joints = []
    for joint_name, joint_info in joints_data.items():
        if joint_info.get('mimic_joint') is not None:
            mimic_joints.append((
                joint_name,
                joint_info['mimic_joint'],
                joint_info['mimic_multiplier'],
                joint_info['mimic_offset']
            ))
    
    if not mimic_joints:
        print("[INFO] 没有找到mimic joints")
        return
    
    # 创建equality元素
    equality_elem = ET.SubElement(mujoco_elem, 'equality')
    
    for follower_joint, leader_joint, multiplier, offset in mimic_joints:
        # MuJoCo的joint equality：
        # polycoef="offset multiplier 0 0 0"
        # 表示：follower = offset + multiplier * leader
        ET.SubElement(equality_elem, 'joint',
                     joint1=leader_joint,
                     joint2=follower_joint,
                     polycoef=f'{offset} {multiplier} 0 0 0')
        print(f"  ✓ 添加 mimic 约束: {follower_joint} = {offset} + {multiplier} × {leader_joint}")


def find_body_by_name(element, body_name):
    """递归查找指定名称的body元素"""
    if element.tag == 'body' and element.get('name') == body_name:
        return element
    for child in element:
        result = find_body_by_name(child, body_name)
        if result is not None:
            return result
    return None


def indent_xml(elem, level=0):
    """格式化XML缩进"""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将RuiYan Hand URDF转换为MuJoCo MJCF (手动创建)')
    parser.add_argument('urdf_path', type=str, help='URDF文件路径')
    parser.add_argument('--output', '-o', type=str, help='输出MJCF文件名（将放在meshes目录）')
    
    args = parser.parse_args()
    
    urdf_path = Path(args.urdf_path)
    
    # 判断是左手还是右手
    hand_type = 'left' if 'Left' in urdf_path.stem else 'right'
    
    # 默认输出文件名
    if args.output:
        output_name = args.output
    else:
        output_name = urdf_path.stem + '_scene.xml'
    
    create_mjcf_manually(urdf_path, output_name, hand_type)


if __name__ == '__main__':
    main()

