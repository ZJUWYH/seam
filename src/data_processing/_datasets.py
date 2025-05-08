import json

import torch
from datasets import load_dataset, Dataset


CONTEXT_LENGTH = 256

BT_CATEGORIES = {
    "half": [
        "violence,aiding_and_abetting,incitement",
        "hate_speech,offensive_language",
        "discrimination,stereotype,injustice",
        "misinformation_regarding_ethics,laws_and_safety",
        "privacy_violation"
    ],
    "other_half": [
        "financial_crime,property_crime,theft",
        "drug_abuse,weapons,banned_substance",
        "terrorism,organized_crime",
        "sexually_explicit,adult_content"
    ],
    "violence": [
        "violence,aiding_and_abetting,incitement"
    ],
    "toxic": [
        "hate_speech,offensive_language",
        "discrimination,stereotype,injustice",
    ],
    "privacy_misinformation": [
        "misinformation_regarding_ethics,laws_and_safety",
        "privacy_violation"
    ],
    "crime_unethical": [
        "financial_crime,property_crime,theft",
        "drug_abuse,weapons,banned_substance",
        "non_violent_unethical_behavior",
        "terrorism,organized_crime"
    ],
    "sexually_explicit": [
        "sexually_explicit,adult_content"
    ],
    "self_harm": [
        "self_harm"
    ]
}

# This function is adapted from https://github.com/domenicrosati/representation-noising/blob/main/representation_noising/datasets.py
def construct_beavertails_dataset(
    tokenizer, train_harmful_subset="all", test_harmful_subset="all",
    withhold_harmful_subset="",
    attack= False,
    attack_size=1000,
    defence_size=1000,
    context_length=CONTEXT_LENGTH
):
    with open("data/beavertails_with_refusals_train.json", "r", encoding = "utf-8") as f:
        trainds = json.load(f)
    trainds = Dataset.from_list(trainds)
    testds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    # prepare tokenizer where we have the subsets
    def _train_dataset_tokenizer(element):
        contents = element['response']
        categories = element['category']
        processed_categories = []
        for category in categories:
            found_category = False
            for k, v in category.items():
                if v and not found_category:
                    processed_categories.append(k)
                    found_category = True

        is_safes = element['is_safe']
        prompts = element['prompt']
        harmful_outputs = []
        harmful_categories = []
        output_categories = []
        if train_harmful_subset != "all":
            harmful_categories = BT_CATEGORIES[train_harmful_subset]
        else:
            for name, category in BT_CATEGORIES.items():
                if name in withhold_harmful_subset:
                    continue
                harmful_categories.extend(category)
        refusal_outputs = []
        for prompt, content, category, is_safe, refusal_text in zip(prompts, contents, processed_categories, is_safes, element['refusal']):
            if is_safe:
                continue
            else:
                if category not in harmful_categories and train_harmful_subset != "all":
                    continue
                harmful_outputs.append(
                    f"Question: {prompt}\nAnswer:\n{content}"
                )
                refusal_outputs.append(
                    f"{refusal_text}"
                )
                output_categories.append(category)
        harmful_outputs = tokenizer(
            harmful_outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        harmful_outputs["labels"] = harmful_outputs["input_ids"].clone()
        return {
            **harmful_outputs,
            "categories": output_categories,
        }
    def _refusal_dataset_tokenizer(element):
        contents = element['response']
        categories = element['category']
        processed_categories = []
        for category in categories:
            found_category = False
            for k, v in category.items():
                if v and not found_category:
                    processed_categories.append(k)
                    found_category = True

        is_safes = element['is_safe']
        prompts = element['prompt']
        harmful_outputs = []
        harmful_categories = []
        output_categories = []
        if train_harmful_subset != "all":
            harmful_categories = BT_CATEGORIES[train_harmful_subset]
        else:
            for name, category in BT_CATEGORIES.items():
                if name in withhold_harmful_subset:
                    continue
                harmful_categories.extend(category)
        refusal_outputs = []
        for prompt, content, category, is_safe, refusal_text in zip(prompts, contents, processed_categories, is_safes, element['refusal']):
            if is_safe:
                continue
            else:
                if category not in harmful_categories and train_harmful_subset != "all":
                    continue
                harmful_outputs.append(
                    f"Question: {prompt}\nAnswer: {content}"
                )
                refusal_outputs.append(
                    f"{refusal_text}"
                )
                output_categories.append(category)
        refusal_outputs = tokenizer(
            refusal_outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        refusal_outputs["labels"] = refusal_outputs["input_ids"].clone()
        return {
            **refusal_outputs,
            "categories": output_categories,
        }

    def _test_dataset_tokenizer(element):
        contents = element['response']
        categories = element['category']
        processed_categories = []
        for category in categories:
            found_category = False
            for k, v in category.items():
                if v and not found_category:
                    processed_categories.append(k)
                    found_category = True

        is_safes = element['is_safe']
        prompts = element['prompt']
        harmful_outputs = []
        harmful_categories = []
        output_categories = []
        if test_harmful_subset != "all":
            harmful_categories = BT_CATEGORIES[test_harmful_subset]
        else:
            for name, category in BT_CATEGORIES.items():
                if name in withhold_harmful_subset:
                    continue
                harmful_categories.extend(category)
        for prompt, content, category, is_safe in zip(prompts, contents, processed_categories, is_safes):
            if is_safe:
                continue
            else:
                if category not in harmful_categories:
                    continue
                harmful_outputs.append(
                    f"Question: {prompt}\nAnswer:"
                )
                output_categories.append(category)
        if len(harmful_outputs) == 0:
            return {}
        harmful_outputs = tokenizer(
            harmful_outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        harmful_outputs["labels"] = harmful_outputs["input_ids"].clone()
        return {
            **harmful_outputs,
            "categories": output_categories
        }

    tokenized_harmful = trainds.map(
        _train_dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            trainds.column_names
            if col not in ["input_ids", "attention_mask", "categories", "labels"]
        ],
        batch_size = None
    )
    tokenized_harmful.set_format("torch")

    tokenized_harmless = trainds.map(
        _refusal_dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            trainds.column_names
            if col not in ["input_ids", "attention_mask", "categories", "labels"]
        ],
        batch_size = None
    )
    tokenized_harmless.set_format("torch")

    tokenizer.padding_side = "left"

    tokenized_test = testds.map(
        _test_dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            testds.column_names
            if col not in ["input_ids", "attention_mask", "categories", "labels"]
        ],
        batch_size = None
    )
    tokenized_test.set_format("torch")

    tokenizer.padding_side = "right"

    # only select 100 samples to test on
    if len(tokenized_test) > 100:
        tokenized_test = tokenized_test.select(range(100))
    if attack:
        if attack_size < len(tokenized_harmful):
            tokenized_harmful = tokenized_harmful.select(range(attack_size))
            tokenized_harmless = tokenized_harmless.select(range(attack_size))
    else:
        if defence_size < len(tokenized_harmful):
            tokenized_harmful = tokenized_harmful.select(range(defence_size))
            tokenized_harmless = tokenized_harmless.select(range(defence_size))

    return tokenized_harmful, tokenized_harmless, tokenized_test

def construct_alpaca_dataset(tokenizer, retain_size = 1000):
    ds = load_dataset("yahma/alpaca-cleaned")
    def dataset_tokenizer(examples):
        prompts = [instruction + "\n" + input_text for instruction, input_text in 
                zip(examples["instruction"], examples["input"])]
        contents = examples["output"]
        
        input_lines = [f"Question: {prompt}\nAnswer: {content}" 
                    for prompt, content in zip(prompts, contents)]
        
        outputs = tokenizer(
            input_lines, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=CONTEXT_LENGTH
        )
        
        outputs["labels"] = outputs["input_ids"].clone()
        outputs["labels"][outputs["labels"] == tokenizer.pad_token_id] = -100        
        return outputs
    # Apply tokenization to the dataset
    tokenized_ds = ds["train"].map(
        dataset_tokenizer,
        batched=True,
        remove_columns=[
        col for col in
        ds["train"].column_names
        if col not in ["input_ids", "attention_mask", "labels"]
    ]
    )
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # Select a subset of the dataset
    retain_indexs = 0
    if retain_size < len(tokenized_ds):
        tokenized_ds = tokenized_ds.select(range(retain_indexs, retain_indexs + retain_size))
    return tokenized_ds
    


