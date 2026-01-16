import pickle 

file_path = "/share_data/chenzhang/projects/EgoVLA/outputs/2026.01.15/17.29_vq_tokenizer_train_vq_tokenizer/normalizer.pkl"
with open(file_path, "rb") as f:
    normalizer = pickle.load(f)

print(normalizer)
print(normalizer.params_dict.keys())
print(f"states min: {normalizer.params_dict['states']['input_stats']['min']}")
print(f"states max: {normalizer.params_dict['states']['input_stats']['max']}")
print(f"states mean: {normalizer.params_dict['states']['input_stats']['mean']}")
print(f"states std: {normalizer.params_dict['states']['input_stats']['std']}")
print(f"actions min: {normalizer.params_dict['actions']['input_stats']['min']}")
print(f"actions max: {normalizer.params_dict['actions']['input_stats']['max']}")
print(f"actions mean: {normalizer.params_dict['actions']['input_stats']['mean']}")
print(f"actions std: {normalizer.params_dict['actions']['input_stats']['std']}")