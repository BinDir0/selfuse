import os
import shutil
import random
from datasets import load_dataset, load_from_disk, disable_progress_bar
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- 配置区域 ---
root_path = "/share_data/guantianrui/datasets/VLM/FineVision"
output_root = "/share_data/guantianrui/datasets/VLM/FineVision_Arrow_Format"
local_tmp_cache_root = "/DATA/chenzhang/.cache" # 临时总目录

val_ratio = 0.001
seed = 42
max_parallel_subsets = 32  # 同时并行转换多少个子集
num_proc_per_subset = 4  # 每个子集内部使用的进程数

# 禁用 Hugging Face 默认的进度条，防止多进程下满屏乱跳
disable_progress_bar()
# ----------------

def convert_single_subset(s):
    """单个子集的转换逻辑"""
    subset_out_path = os.path.join(output_root, s)
    if os.path.exists(subset_out_path):
        return f"SKIP: {s}"

    my_cache_dir = os.path.join(local_tmp_cache_root, s)
    os.makedirs(my_cache_dir, exist_ok=True)
    
    # 每个进程需要独立的环境变量配置
    os.environ["HF_DATASETS_CACHE"] = my_cache_dir

    try:
        # 1. 加载
        ds = load_dataset(
            "parquet", 
            data_files=f"{root_path}/{s}/*.parquet", 
            split="train",
            cache_dir=my_cache_dir
        )
        
        # 2. 切分
        ds_dict = ds.train_test_split(test_size=val_ratio, seed=seed)
        
        # 3. 保存
        ds_dict.save_to_disk(subset_out_path, num_proc=num_proc_per_subset)
        
        # 4. 清理临时缓存
        if os.path.exists(my_cache_dir):
            shutil.rmtree(my_cache_dir)
        return f"DONE: {s}"
    except Exception as e:
        return f"ERROR: {s} - {str(e)}"

def test_arrow_conversion(subset_name=None, num_random_samples=10):
    """
    测试 arrow 格式转换是否正确
    
    Args:
        subset_name: 要测试的子集名称，如果为 None 则测试所有已转换的子集
        num_random_samples: 每个 split 随机采样的样本数量
    """
    print("=" * 80)
    print("开始测试 Arrow 格式转换...")
    print("=" * 80)
    
    # 获取要测试的子集列表
    if subset_name:
        subsets_to_test = [subset_name]
    else:
        # 测试所有已转换的子集
        if not os.path.exists(output_root):
            print(f"错误: 输出目录不存在: {output_root}")
            return
        
        subsets_to_test = [
            d for d in os.listdir(output_root) 
            if os.path.isdir(os.path.join(output_root, d))
        ]
        
        if not subsets_to_test:
            print(f"警告: 输出目录中没有找到已转换的子集: {output_root}")
            return
    
    print(f"找到 {len(subsets_to_test)} 个子集需要测试\n")
    
    all_passed = True
    
    for subset in subsets_to_test:
        subset_path = os.path.join(output_root, subset)
        
        print(f"\n{'='*80}")
        print(f"测试子集: {subset}")
        print(f"路径: {subset_path}")
        print(f"{'='*80}")
        
        try:
            # 1. 检查目录是否存在
            if not os.path.exists(subset_path):
                print(f"❌ 错误: 子集目录不存在")
                all_passed = False
                continue
            
            # 2. 加载 arrow 格式数据
            print("\n[1] 加载 Arrow 格式数据...")
            ds_dict = load_from_disk(subset_path)
            
            # 3. 检查基本结构
            print("\n[2] 检查数据集结构...")
            if not isinstance(ds_dict, dict):
                print(f"❌ 错误: 数据集不是字典格式")
                all_passed = False
                continue
            
            if "train" not in ds_dict or "test" not in ds_dict:
                print(f"❌ 错误: 数据集缺少 'train' 或 'test' split")
                print(f"    可用的 splits: {list(ds_dict.keys())}")
                all_passed = False
                continue
            
            train_ds = ds_dict["train"]
            test_ds = ds_dict["test"]
            
            print(f"✅ 数据集结构正确: train 和 test splits 都存在")
            
            # 4. 检查元数据
            print("\n[3] 检查元数据...")
            
            # 检查 train split
            train_len = len(train_ds)
            train_features = train_ds.features
            train_column_names = train_ds.column_names
            
            print(f"  Train split:")
            print(f"    - 样本数量: {train_len:,}")
            print(f"    - 列名: {train_column_names}")
            print(f"    - 特征类型: {train_features}")
            
            # 检查 test split
            test_len = len(test_ds)
            test_features = test_ds.features
            test_column_names = test_ds.column_names
            
            print(f"  Test split:")
            print(f"    - 样本数量: {test_len:,}")
            print(f"    - 列名: {test_column_names}")
            print(f"    - 特征类型: {test_features}")
            
            # 验证 train/test split 比例
            total_len = train_len + test_len
            if total_len > 0:
                actual_val_ratio = test_len / total_len
                print(f"\n  Split 比例验证:")
                print(f"    - 总样本数: {total_len:,}")
                print(f"    - 期望 val_ratio: {val_ratio:.4f}")
                print(f"    - 实际 val_ratio: {actual_val_ratio:.4f}")
                
                # 允许一定的误差（由于随机性）
                ratio_tolerance = 0.0001
                if abs(actual_val_ratio - val_ratio) > ratio_tolerance:
                    print(f"    ⚠️  警告: val_ratio 偏差较大 (允许误差: {ratio_tolerance})")
                else:
                    print(f"    ✅ val_ratio 在允许范围内")
            
            # 验证特征一致性
            if train_features != test_features:
                print(f"❌ 错误: train 和 test 的特征类型不一致")
                all_passed = False
                continue
            else:
                print(f"✅ train 和 test 的特征类型一致")
            
            if train_column_names != test_column_names:
                print(f"❌ 错误: train 和 test 的列名不一致")
                all_passed = False
                continue
            else:
                print(f"✅ train 和 test 的列名一致")
            
            # 5. 随机采样检查数据内容
            print(f"\n[4] 随机采样检查数据内容 (每个 split 采样 {num_random_samples} 个样本)...")
            
            # 检查 train split
            if train_len > 0:
                print(f"\n  检查 Train split:")
                train_indices = random.sample(range(train_len), min(num_random_samples, train_len))
                for idx in train_indices:
                    try:
                        sample = train_ds[idx]
                        # 检查样本不为空
                        if not sample:
                            print(f"    ⚠️  警告: train[{idx}] 为空")
                        else:
                            # 检查所有列都有值
                            empty_cols = [col for col in train_column_names if col not in sample or sample[col] is None]
                            if empty_cols:
                                print(f"    ⚠️  警告: train[{idx}] 的列 {empty_cols} 为空")
                            else:
                                print(f"    ✅ train[{idx}] 数据完整")
                    except Exception as e:
                        print(f"    ❌ 错误: 无法加载 train[{idx}]: {e}")
                        all_passed = False
            
            # 检查 test split
            if test_len > 0:
                print(f"\n  检查 Test split:")
                test_indices = random.sample(range(test_len), min(num_random_samples, test_len))
                for idx in test_indices:
                    try:
                        sample = test_ds[idx]
                        # 检查样本不为空
                        if not sample:
                            print(f"    ⚠️  警告: test[{idx}] 为空")
                        else:
                            # 检查所有列都有值
                            empty_cols = [col for col in test_column_names if col not in sample or sample[col] is None]
                            if empty_cols:
                                print(f"    ⚠️  警告: test[{idx}] 的列 {empty_cols} 为空")
                            else:
                                print(f"    ✅ test[{idx}] 数据完整")
                    except Exception as e:
                        print(f"    ❌ 错误: 无法加载 test[{idx}]: {e}")
                        all_passed = False
            
            # 6. 检查数据可迭代性
            print(f"\n[5] 检查数据可迭代性...")
            try:
                # 尝试迭代前几个样本
                train_iter_count = 0
                for i, sample in enumerate(train_ds):
                    train_iter_count += 1
                    if i >= 2:  # 只检查前3个
                        break
                print(f"  ✅ Train split 可以正常迭代 (已检查前 {train_iter_count} 个样本)")
            except Exception as e:
                print(f"  ❌ 错误: Train split 无法迭代: {e}")
                all_passed = False
            
            try:
                # 尝试迭代前几个样本
                test_iter_count = 0
                for i, sample in enumerate(test_ds):
                    test_iter_count += 1
                    if i >= 2:  # 只检查前3个
                        break
                print(f"  ✅ Test split 可以正常迭代 (已检查前 {test_iter_count} 个样本)")
            except Exception as e:
                print(f"  ❌ 错误: Test split 无法迭代: {e}")
                all_passed = False
            
            print(f"\n✅ 子集 '{subset}' 测试通过!")
            
        except Exception as e:
            print(f"\n❌ 子集 '{subset}' 测试失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # 总结
    print(f"\n{'='*80}")
    if all_passed:
        print("🎉 所有测试通过!")
    else:
        print("❌ 部分测试失败，请检查上述错误信息")
    print(f"{'='*80}\n")
    
    return all_passed

def main():
    os.makedirs(output_root, exist_ok=True)
    subsets = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    print(f"开始并行转换任务...")
    print(f"总子集数: {len(subsets)}, 并行任务数: {max_parallel_subsets}")

    # 使用 as_completed 来驱动 tqdm 进度条
    with ProcessPoolExecutor(max_workers=max_parallel_subsets) as executor:
        # 提交所有任务
        future_to_subset = {executor.submit(convert_single_subset, s): s for s in subsets}
        
        # 使用 tqdm 包裹 as_completed 迭代器
        with tqdm(total=len(subsets), desc="Overall Progress") as pbar:
            for future in as_completed(future_to_subset):
                subset_name = future_to_subset[future]
                try:
                    result = future.result()
                    # 使用 tqdm.write 而不是 print，这样不会破坏进度条渲染
                    tqdm.write(result)
                except Exception as exc:
                    tqdm.write(f"子集 {subset_name} 产生了未预料的异常: {exc}")
                
                pbar.update(1)

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数来选择运行模式
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 测试模式
        subset_name = sys.argv[2] if len(sys.argv) > 2 else None
        num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        test_arrow_conversion(subset_name=subset_name, num_random_samples=num_samples)
    else:
        # 默认运行转换
        main()