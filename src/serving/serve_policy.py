import logging
import pathlib
import socket

from omegaconf import OmegaConf

from src.serving.websocket_policy_server import (
    create_engine, 
    create_env_wrapper,
    EnvWrapper,
    WebsocketPolicyServer,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _load_config() -> OmegaConf:
    config_path = pathlib.Path(__file__).resolve().parents[1] / "config" / "experiment" / "inference.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    assert "serving" in cfg and "policy" in cfg and "env_wrapper" in cfg, \
        f"Missing policy/serving/env_wrapper config in: {config_path}"
    return cfg


def main() -> None:
    cfg = _load_config()
    policy = create_engine(cfg.policy, cfg.serving)
    if getattr(cfg, "env_wrapper", None) and cfg.env_wrapper.enabled:
        wrapper_cfg = cfg.env_wrapper
        policy = create_env_wrapper(policy, wrapper_cfg)
    policy_metadata = policy.metadata

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host=cfg.serving.host,
        port=cfg.serving.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()