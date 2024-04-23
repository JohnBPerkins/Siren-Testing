import hydra


@hydra.main(config_name="individual", config_path="./inr2array_configs", version_base=None)
def main(cfg=None):
    from omegaconf import OmegaConf
    from experiments.inr2array_individual import train_and_eval
    print(OmegaConf.to_yaml(cfg, resolve=True))
    train_and_eval(cfg)


if __name__ == "__main__":
    main()
