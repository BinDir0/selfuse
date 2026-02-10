import logging
import pathlib
import socket

from omegaconf import OmegaConf

from src.serving import websocket_policy_server


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _load_config() -> OmegaConf:
    config_path = pathlib.Path(__file__).resolve().parents[1] / "config" / "experiment" / "inference.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    if "serving" not in cfg or "policy" not in cfg or "inference" not in cfg:
        raise ValueError(f"Missing policy/inference/serving config in: {config_path}")
    return cfg


def main() -> None:
    cfg = _load_config()
    policy_cfg = OmegaConf.merge(cfg.policy, cfg.inference)
    policy = websocket_policy_server.create_policy(policy_cfg, cfg.serving)
    policy_metadata = policy.metadata

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=cfg.serving.host,
        port=cfg.serving.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()