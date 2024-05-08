from __future__ import annotations

from data import DataRow, RawDataset, SplitType
from debate import Debater, DebateRound, Judge, QuestionMetadata
from models import OpenAIModel, MistralModel, Model, RandomModel, SpeechStructure
from prompts import Prompt, PromptConfig, PromptLoadingConfig, PromptParser
from train.row_converter import RowConverter
from train.train_utils import TrainUtils, TrainingConfig, TrainingTarget
from utils import LoggingCallback, logger_utils, string_utils
import utils.constants as constants

from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import GenerationConfig
from trl import PPOConfig, PPOTrainer
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch

import math, sys  # Remove sys

try:
    import bitsandbytes as bnb
except:
    print("Unable to import bitsandbytes")

try:
    from utils.flash_attn_utils import (
        replace_attn_with_flash_attn,
        upcast_layer_for_flash_attention,
    )

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


def print_available_memory(logger):
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = total_memory - torch.cuda.memory_allocated(0)
    logger.warn(f"Total GPU Memory: {total_memory / (1024 ** 3):.2f} GB")
    logger.warn(f"Free GPU Memory: {free_memory / (1024 ** 3):.2f} GB")


# Extended monkeypatch script to fix a bug in PPOTrainer
def logprobs_from_logits(logits, labels, gather=True):
    logp = F.log_softmax(logits, dim=2)
    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


# Another function that's just here to get the monkeypatching to work
def batched_forward_pass(
    self,
    model,
    queries,
    responses,
    model_inputs,
    return_logits=None,
    response_masks=None,
):
    torch.cuda.empty_cache()
    bs = len(queries)
    fbs = self.config.mini_batch_size
    all_logprobs = []
    all_logits = []
    all_masks = []
    all_values = []

    logger = logger_utils.get_default_logger(__name__)
    # This is the only change
    if torch.is_grad_enabled():
        model.train()
        logger.warn(f"TRAINING")
    else:
        model.eval()
        logger.warn(f"NOT TRAINING")

    is_gradient_checkpointing = model.pretrained_model.is_gradient_checkpointing
    logger.warn(f"Is gradient checkpointing? {is_gradient_checkpointing}")

    for i in range(math.ceil(bs / fbs)):
        input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
        query_batch = queries[i * fbs : (i + 1) * fbs]
        response_batch = responses[i * fbs : (i + 1) * fbs]
        if response_masks is not None:
            response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]

        print_available_memory(logger)
        logits, _, values = model(**input_kwargs)

        if self.is_encoder_decoder:
            input_ids = input_kwargs["decoder_input_ids"]
            attention_mask = input_kwargs["decoder_attention_mask"]
        else:
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

        logits_size = logits.shape
        logger.warn(f"Log prob size is {logits_size}")
        print_available_memory(logger)
        logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
        masks = torch.zeros_like(attention_mask)
        masks[:, :-1] = attention_mask[:, 1:]

        for j in range(len(query_batch)):
            if self.is_encoder_decoder:
                # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                start = 1
                end = attention_mask[j, :].sum() - 1
            else:
                start = len(query_batch[j]) - 1  # logprobs starts from the second query token
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0]
                end = start + len(response_batch[j])
                if response_masks is not None:
                    response_masks_batch[j] = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

            masks[j, :start] = 0
            masks[j, end:] = 0
            if response_masks is not None:
                masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

        if return_logits:
            all_logits.append(logits)
        else:
            del logits
        all_values.append(values)
        all_logprobs.append(logprobs)
        all_masks.append(masks)

    return (
        torch.cat(all_logprobs),
        torch.cat(all_logits)[:, :-1] if return_logits else None,
        torch.cat(all_values)[:, :-1],
        torch.cat(all_masks)[:, :-1],
    )


PPOTrainer.batched_forward_pass = batched_forward_pass


class PPOTrainerWrapper:
    INSTRUCTION_COLUMN = "instruction"
    INPUT_COLUMN = "input"
    QUERY_COLUMN = "query"
    SCORE_COLUMN = "score"
    RESPONSE_COLUMN = "response"
    OPPONENT_QUERY_COLUMN = "opponent_query"
    JUDGE_QUERY_COLUMN = "judge_query"
    INPUT_IDS_COLUMN = "input_ids"
    EXTRA_SUFFIX_COLUMN = "extra_suffix"
    BATCH_SIZE = 1
    DEFAULT_DEBATER_ALIAS = "mixtral-debater"
    DEFAULT_JUDGE_ALIAS = "openai-judge"
    MAX_GENERATION_LENGTH = 300

    def __init__(self, ppo_trainer: PPOTrainer, config: TrainingConfig, dataset: RawDataset):
        """
        Class for training a model using Proximate Policy Optimization. In order to keep the
        interface the same as the one used by the DPOTrainer and SFTTrainer, we construct a
        wrapper here so that one can just call the train() function to run the training loop.

        Params:
            ppo_trainer: a huggingface-object that handles the actual ppo algorithm
            config: configuration specifying the prompt setup and hyperparameters for the training run.
            dataset: the dataset to train on
        """

        self.ppo_trainer = ppo_trainer

        # TODO: change
        self.internal_model = MistralModel(
            alias=PPOTrainerWrapper.DEFAULT_DEBATER_ALIAS,
            file_path=None,
            is_debater=True,
            nucleus=True,
        )
        self.internal_model.model = self.ppo_trainer.model.pretrained_model
        self.internal_model.model.config.sliding_window = constants.MAX_LENGTH
        self.internal_model.tokenizer = self.ppo_trainer.tokenizer
        self.internal_model.generation_config = self.internal_model.create_default_generation_config()
        self.internal_model.instantiated_model = True
        self.internal_model.is_debater = True

        self.reward_model = OpenAIModel(alias=PPOTrainerWrapper.DEFAULT_JUDGE_ALIAS, is_debater=False)
        self.config = config
        self.dataset = dataset

        self.logger = logger_utils.get_default_logger(__name__)

    def get_batch_samples(
        self,
        start_idx: int,
    ) -> tuple[list[str], list[str], list[float]]:
        samples = []
        for i in range(
            self.config.training_hyperparameters.per_device_train_batch_size // 2
        ):  # TODO: change this when we do multi-turn
            new_samples = self.generate_one_round_samples(
                idx=start_idx + i,
            )
            samples.extend(new_samples)

        query_texts = [sample[0] for sample in samples]
        response_texts = [sample[1] for sample in samples]
        score_texts = [sample[2] for sample in samples]
        return query_texts, response_texts, score_texts

    def train(self, num_iters: int):
        for i in range(num_iters):
            self.train_single_batch(start_idx=(i * self.config.training_hyperparameters.per_device_train_batch_size))

    def train_single_batch(self, start_idx: int):
        with torch.no_grad():
            query_texts, response_texts, score_texts = self.get_batch_samples(start_idx=start_idx)

        queries = [self.ppo_trainer.tokenizer(qt, return_tensors="pt").input_ids.squeeze().to("cuda") for qt in query_texts]
        responses = [
            self.ppo_trainer.tokenizer(rt, return_tensors="pt").input_ids.squeeze().to("cuda") for rt in response_texts
        ]
        scores = [x.to("cuda") for x in torch.FloatTensor(score_texts)]

        stats = self.ppo_trainer.step(
            queries=queries,
            responses=responses,
            scores=scores,
        )

        self.ppo_trainer.log_stats(stats=stats, batch={"query": queries, "response": responses}, rewards=scores)

    def generate_one_round_samples(
        self,
        idx: int,
    ) -> list[tuple[str, str]]:
        example = self.dataset.get_example(idx=idx, split=SplitType.TRAIN)

        topic = example.question
        position = example.positions[0]
        opponent_position = example.positions[1]
        background_text = example.background_text
        title = example.story_title
        correct_index = example.correct_index
        speeches = example.speeches

        debate_identifier = f"{title}_{topic}"

        config_a = PromptConfig(
            name=constants.DEFAULT_DEBATER_A_NAME,
            opponent_name=constants.DEFAULT_DEBATER_B_NAME,
            position=position,
            opponent_position=opponent_position,
            topic=topic,
            background_text=background_text,
        )

        config_b = PromptConfig(
            name=constants.DEFAULT_DEBATER_B_NAME,
            opponent_name=constants.DEFAULT_DEBATER_A_NAME,
            position=opponent_position,
            opponent_position=position,
            topic=topic,
            background_text=background_text,
        )

        prompt_a = PromptParser.parse(
            prompt_config=config_a,
            prompts_file_path=self.config.prompt_config.file_path,
            name=self.config.speech_structure[0].default_prompt_name or self.config.prompt_config.default_prompt_name,
        )

        prompt_b = PromptParser.parse(
            prompt_config=config_b,
            prompts_file_path=self.config.prompt_config.file_path,
            name=self.config.speech_structure[0].default_prompt_name or self.config.prompt_config.default_prompt_name,
        )

        prompt_judge = PromptParser.parse(
            prompt_config=config_a,
            prompts_file_path=self.config.prompt_config.file_path,
            name=self.config.speech_structure[0].default_prompt_name or self.config.prompt_config.default_prompt_name,
        )

        question_metadata = QuestionMetadata(
            first_debater_correct=correct_index == 0,
            question_idx=idx,
            background_text=background_text,
            question=topic,
            first_debater_answer=position,
            second_debater_answer=opponent_position,
            debate_identifier=debate_identifier,
        )

        num_speeches = 1  # change this later
        debater_a = Debater(
            name=constants.DEFAULT_DEBATER_A_NAME,
            prompt=prompt_a,
            model=self.internal_model,
            num_speeches=num_speeches,
            speech_format=self.config.speech_structure[0].debater_format.get_speech_format(
                name=constants.DEFAULT_DEBATER_A_NAME,
                num_speeches=num_speeches,
                use_scratchpad=False,
            ),
            quotes_require_validation=True,
        )

        debater_b = Debater(
            name=constants.DEFAULT_DEBATER_B_NAME,
            prompt=prompt_b,
            model=self.internal_model,
            num_speeches=num_speeches,
            speech_format=self.config.speech_structure[0].debater_format.get_speech_format(
                name=constants.DEFAULT_DEBATER_B_NAME,
                num_speeches=num_speeches,
                use_scratchpad=False,
            ),
            quotes_require_validation=True,
        )

        judge = Judge(
            name=constants.DEFAULT_JUDGE_NAME,
            prompt=prompt_judge,
            model=self.reward_model,
            speech_format=self.config.speech_structure[0].judge_format.get_speech_format(
                name=constants.DEFAULT_JUDGE_NAME,
                num_speeches=num_speeches,
                use_scratchpad=False,
                flipped=False,
            ),
            num_speeches=num_speeches,
        )

        debate_round = DebateRound(
            first_debater=debater_a,
            second_debater=debater_b,
            judge=judge,
            metadata=[question_metadata],
        )

        self.logger.warn("Starting first round")
        summary = debate_round()
        self.logger.warn("Ending first round")

        samples = []
        for speech in filter(
            lambda x: x.speaker in [constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME],
            judge.transcripts[0].speeches,
        ):
            samples.append(
                (
                    speech.supplemental.prompt,
                    speech.content,
                    summary[0].first_debater_win_prob
                    if speech.speaker == constants.DEFAULT_DEBATER_A_NAME
                    else summary[0].second_debater_win_prob,
                )
            )
        return samples

    def save_model(self):
        """Saves the model to the specified location"""
        self.ppo_trainer.save_model(self.config.logging_and_saving_config.output_dir)

    @classmethod
    def get_optimizer(cls, model):
        """Gets the optimizer to use during the training run"""
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = bnb.optim.PagedAdamW(trainable_params, lr=10e-4, weight_decay=10e-3)
        manager = bnb.optim.GlobalOptimManager.get_instance()
        skipped = 0
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                manager.register_module_override(module, "weight", {"optim_bits": 32})
        return optimizer

    @classmethod
    def get_trainer(
        cls,
        config: TrainingConfig,
        raw_datasets: Optional[list[RawDataset]] = None,
        is_local: bool = False,
        is_test: bool = False,
    ) -> PPOTrainerWrapper:
        """
        Generates a PPOTrainerWrapper object that should have the same interface as trl's
        SFTTrainer and DPOTrainer objects.

        Params:
            config: configuration specifying the prompt setup and hyperparameters for the training run.
            raw_dataset: dataset to use for training. Autogenerated using the config if it is missing
            is_local: whether this is being run on a cpu

        Returns:
            ppo_trainer: One can call ppo_trainer.train() to then run the training loop.
        """

        """
        if FLASH_ATTENTION_AVAILABLE:
            replace_attn_with_flash_attn(disable_dropout=True)
        """

        if not raw_datasets:
            raw_datasets = TrainUtils.create_datasets(config=config)
        raw_dataset = raw_datasets[0]  # we don't support multiple datasets at the moment

        ppo_config = PPOConfig(
            steps=config.training_hyperparameters.steps,
            learning_rate=config.training_hyperparameters.learning_rate,
            batch_size=config.training_hyperparameters.per_device_train_batch_size,
            gradient_accumulation_steps=config.training_hyperparameters.gradient_accumulation_steps,
            mini_batch_size=1,
            ppo_epochs=1,
            optimize_device_cache=True,
        )

        tokenizer = TrainUtils.get_tokenizer(config=config)
        model = TrainUtils.load_model(config=config, is_local=is_local, requires_value_head=True)

        if FLASH_ATTENTION_AVAILABLE:
            model = upcast_layer_for_flash_attention(model, torch.bfloat16)
            model.pretrained_model = upcast_layer_for_flash_attention(model.pretrained_model, torch.bfloat16)

        model.gradient_checkpointing_enable()
        model.pretrained_model.gradient_checkpointing_enable()
        model.pretrained_model = prepare_model_for_kbit_training(
            model.pretrained_model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.gradient_checkpointing_enable = model.pretrained_model.gradient_checkpointing_enable

        optimizer = PPOTrainerWrapper.get_optimizer(model=model)

        ppo_trainer = PPOTrainerWrapper(
            ppo_trainer=PPOTrainer(model=model, config=ppo_config, tokenizer=tokenizer, optimizer=optimizer),
            config=config,
            dataset=raw_dataset,
        )

        return ppo_trainer
