import json

info = json.load(open("/share_data/guantianrui/datasets/VITRA-1M/zarr_episodes_statistics.json"))

for stat in info["video_statistics"]:
    for ep in stat["episodes"]:
        if ep["zarr_episode_name"] == "Ego4D_000786a7-3f9d-4fe6-bfb3-045b368f7d44_ep_000042":
            print(ep)
            break
