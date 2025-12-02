from .pretrain_trainer import PretrainTrainer
from .sft_trainer import SFTTrainer
from .dpo_trainer import DPOTrainer
from ..config import TrainConfig
from ..cli import get_args

def launch_pretrain(config: TrainConfig):
    pretrain_trainer = PretrainTrainer(config)
    pretrain_trainer.train()

def launch_sft(config: TrainConfig):
    sft_trainer = SFTTrainer(config)
    sft_trainer.train()

def launch_dpo(config: TrainConfig):
    dpo_trainer = DPOTrainer(config)
    dpo_trainer.train()


def launch_train(config: TrainConfig):

    if config.training_stage in ["pretrain", "continue_pretrain"]:
        launch_pretrain(config)
    elif config.training_stage in ["sft"]:
        launch_sft(config)
    elif config.training_stage in ["dpo"]:
        launch_dpo(config)
    else:
        raise ValueError(f"Invalid training stage: {config.training_stage}")


def launch_train_from_cli():

    args = get_args()
    args_dict = vars(args)

    args_dict.pop('yaml_config', None) 
    args_dict.pop('local_rank', None) 

    config = TrainConfig(**args_dict)
    launch_train(config)


if __name__ == "__main__":
    launch_train_from_cli()