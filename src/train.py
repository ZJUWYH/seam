import argparse
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.utils import requires_backends

from src.core.trainer import SeamTrainer
from src.data_processing._datasets import construct_alpaca_dataset, construct_beavertails_dataset
from src.utils.config import load_yaml_config, MyParser


@dataclass
class SeamArguments:
    # Model configuration
    model_name: str = field(
        default=None, 
        metadata={"help": "Path to pretrained model"}
    )
    tokenizer_name: str = field(
        default=None, 
        metadata={"help": "Name or path of the tokenizer"}
    )
    wandb_project_name: str = field(
        default= "seam",
        metadata={"help": "WandB project name"}
    )

    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "WandB run name (optional)"}
    )
    attack_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to use attack dataset"}
    )
    defense_size: int = field(
        default=1000,
        metadata={"help": "Number of examples for defense"}
    )
    
    # Optimization parameters
    epsilon: float = field(
        default=1e-3,
        metadata={"help": "Parameter pertubation radius"}
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "Weight for L_sd"}
    )
    beta: float = field(
        default=0.001,
        metadata={"help": "Weight for L_up"}
    )

@dataclass
class MyTrainingArguments(TrainingArguments):
    @property
    def n_gpu(self):
        """
        The number of GPUs used for data paralleling.

        Note:
            In default Trainer, this is the number of visiable GPUs. Here we use visiable GPUs-2 for data paralleling because we use 2 GPUs to store extra gradients.
            if smaller base models are adopted, we can use 1 GPU for extra gradient transfer. And you can revise 2 to 1 in the code below.
        """
        requires_backends(self, ["torch"])
        if not hasattr(self, "_n_gpu"):
            _ = self._setup_devices
        return self._n_gpu - 2

def main():
    parser = argparse.ArgumentParser(description="Simple config path parser")
    parser.add_argument('--config', type=str, default='config/train.yaml', help='Path to YAML config file')
    config_path, remaining_args = parser.parse_known_args()

    yaml_config = load_yaml_config(config_path.config)
    training_yaml_args = yaml_config.get('TrainingArguments', {})
    unlearn_yaml_args = yaml_config.get('UnlearnArguments', {})
    yaml_defaults = {**training_yaml_args, **unlearn_yaml_args}

    hf_parser = MyParser((MyTrainingArguments, SeamArguments))
    training_args, seam_args = hf_parser.parse_args_into_dataclasses(
        args=remaining_args,
        default_values=yaml_defaults,
    )

    wandb.init(project=seam_args.wandb_project_name, name=seam_args.wandb_run_name, group="train")
    print(f"Using config from: {config_path.config}")
    wandb.config.update(asdict(training_args)) 
    wandb.config.update(asdict(seam_args))

    try:
        tokenizer = AutoTokenizer.from_pretrained(seam_args.model_name)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(seam_args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(seam_args.model_name, torch_dtype=torch.bfloat16)
    # model.config.use_cache = False

    harmful_dataset, harmless_dataset, test_dataset = construct_beavertails_dataset(tokenizer, attack=seam_args.attack_dataset, defence_size=seam_args.defense_size)
    benign_dataset = construct_alpaca_dataset(tokenizer, retain_size=seam_args.defense_size)

    def collate_fn(batch):
        def pad_id(tensor):
            tensor[tensor == tokenizer.pad_token_id] = -100
            return tensor
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([pad_id(item['input_ids']) for item in batch]),
        }

    trainer = SeamTrainer(
        model=model,
        args=training_args,
        seam_args=seam_args,
        train_dataset=harmful_dataset,
        tokenizer=tokenizer,
        align_dataset=harmless_dataset,
        benign_dataset=benign_dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    wandb.log({"Output dir": training_args.output_dir})
    wandb.finish()

if __name__ == "__main__":
    main()