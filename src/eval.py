import argparse
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import wandb
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.data_processing._datasets import construct_beavertails_dataset
from src.utils.config import load_yaml_config, MyParser
from src.utils.metrics import eval_harmfulness, eval_harmfulness_gpt, eval_utility



@dataclass
class EvalArguments:
    model_name: str = field(default=None, metadata={"help": "Path to model"})
    tokenizer_name: str = field(default=None, metadata={"help": "Tokenizer name or path"})
    pre_utility: bool = field(default=True, metadata={"help": "Whether to eval pre attack utility"})
    post_utility: bool = field(default=True, metadata={"help": "Whether to eval post attack utility"})
    attack: bool = field(default=True, metadata={"help": "Whether to HFA"})
    attack_dataset: bool = field(default=True, metadata={"help": "Whether to load attack dataset"})
    attack_size: int = field(default=1000, metadata={"help": "Number of examples for attack evaluation"})
    utility_tasks: List[str] = field(default_factory=lambda: ["mmlu"], 
                                    metadata={"help": "List of utility tasks to evaluate"})
    utility_batch_size: int = field(default=24, metadata={"help": "Batch size for utility task evaluation"})
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "WandB run name (optional)"}
    )
    wandb_project_name: str = field(
        default= "seam",
        metadata={"help": "WandB project name"}
    )
    lora_attack: bool = field(default=False, metadata={"help": "Whether to use LoRA"})


def main():
    parser = argparse.ArgumentParser(description="Simple config path parser")
    parser.add_argument('--config', type=str, default='config/eval.yaml', help='Path to YAML config file')
    config_path, remaining_args = parser.parse_known_args()

    yaml_config = load_yaml_config(config_path.config)
    training_yaml_args = yaml_config.get('TrainingArguments', {})
    eval_yaml_args = yaml_config.get('EvalArguments', {})

    yaml_defaults = {**training_yaml_args, **eval_yaml_args}

    hf_parser = MyParser((TrainingArguments, EvalArguments))
    training_args, eval_args = hf_parser.parse_args_into_dataclasses(
        args=remaining_args,
        default_values=yaml_defaults,
    )

    wandb.init(project=eval_args.wandb_project_name, name=eval_args.wandb_run_name, group="eval")
    print(f"Using config from: {config_path.config}")
    wandb.config.update(asdict(training_args))
    wandb.config.update(asdict(eval_args))

    # load model and tokenizer
    wandb.log({"Model name": eval_args.model_name})
    print("Model name: ", eval_args.model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(eval_args.model_name)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(eval_args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(eval_args.model_name, torch_dtype=torch.bfloat16)
    model.to("cuda:0")
    harmful_dataset, harmless_dataset, test_dataset = construct_beavertails_dataset(
            tokenizer, attack_size=eval_args.attack_size
                )

    if eval_args.pre_utility:
        clean_utility_score = {}
        utility_score = eval_utility(model, tokenizer, tasks=eval_args.utility_tasks, batch_size=eval_args.utility_batch_size)
        try:
            for t in eval_args.utility_tasks:
                clean_utility_score[t] = utility_score[t]["acc,none"]
                print("Pre utility score: ", clean_utility_score)
                wandb.log({"Pre utility score": clean_utility_score})
        except Exception as e:
            print("Pre utility score: ", utility_score)
            wandb.log({"Pre utility score": utility_score})

    if eval_args.attack:
        if eval_args.lora_attack:
            print("Using LoRA attack")
            config = LoraConfig(
                inference_mode=False,
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, config)
            trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=harmful_dataset,
            tokenizer=tokenizer,
            )
            trainer.train()
            model = model.merge_and_unload()
        else:
            trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=harmful_dataset,
            tokenizer=tokenizer,
            )
            trainer.train()

    # evaluate harmfulness and utility
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_args.per_device_train_batch_size,)
    harmful_score = eval_harmfulness(model, tokenizer, test_dataloader)    
    print("Harmful score: ", harmful_score)
    wandb.log({"Harmful score": harmful_score})
    harmful_score_gpt = eval_harmfulness_gpt(model, test_dataloader, tokenizer)
    print("Harmful score GPT: ", harmful_score_gpt)
    wandb.log({"Harmful score GPT": harmful_score_gpt})

    if eval_args.post_utility:
        clean_utility_score = {}
        utility_score = eval_utility(model, tokenizer, tasks=eval_args.utility_tasks, batch_size=eval_args.utility_batch_size)
        try:
            for t in eval_args.utility_tasks:
                clean_utility_score[t] = utility_score[t]["acc,none"]
                print("Post utility score: ", clean_utility_score)
                wandb.log({"Post utility score": clean_utility_score})
        except Exception as e:
            print("Post utility score: ", utility_score)
            wandb.log({"Post utility score": utility_score})

    wandb.finish()

if __name__ == "__main__":
    main()