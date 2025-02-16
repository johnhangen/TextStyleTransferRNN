from src.dataloader import DataLoader
from config.configs import Config

def main():
    config = Config.load_config('config\configs.yaml')
    dataloader = DataLoader(config)
    print(len(dataloader))

if __name__ == "__main__":
    main()