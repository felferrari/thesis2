import hydra

@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def train(cfg):
    pass


if __name__ == "__main__":
    train()