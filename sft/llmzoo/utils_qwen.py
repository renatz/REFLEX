from dataclasses import dataclass
from enum import auto, Enum
from typing import List

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

import transformers
from transformers import WEIGHTS_NAME

import os
from safetensors.torch import save_file
from accelerate import Accelerator
import torch


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
    


class SeparatorStyle(Enum):
    SINGLE = auto()
    TWO = auto()

@dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "</s>"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep
        }



# liar-raw
liar_raw_cev = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " 
        "You are a fact-checking assistant. You are given a claim along with evidence sentences."
        "Your task is to label the overall veracity of the claim based on the provided evidence sentences."
        "The Label Definitions are as follows:"
        "TRUE: The claim is verified as true or mostly-true by the evidences."
        "FALSE: The claim is verified as false or mostly-false by the evidences."
        "HALF-TRUE: The claim is verified as half-true by the evidences or the evidences can not prove the claim."
        "Based on the below claim and evidences, determine the veracity of the claim. "
        "Please strictly output in the following format:"
        "Verdict: [label]",
    roles=("user", "assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="<|im_end|>",
)

liar_raw_cv = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " 
        "You are a fact-checking assistant. You are given a claim."
        "Your task is to label the overall veracity of the claim based on your internal knowledge."
        "The Label Definitions are as follows:"
        "TRUE: The claim is verified as true based on your knowledge."
        "FALSE: The claim is verified as false based on your knowledge."
        "HALF-TRUE: The claim is verified as half-true due to insufficient knowledge leading to uncertainty, or because the claim itself is partially true."
        "Based on the below claim and your own knowledge, determine the veracity of the claim. "
        "Please strictly output in the following format:"
        "Verdict: [label]",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="<|im_end|>",
)


liar_raw_cvi = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " 
        "You are a fact-checking assistant. You are given a claim." 
        "Your task is to label the overall veracity of the claim based on your internal knowledge."
        "Please reason step by step and explain how you reached your conclusion." 
        "The Label Definitions are as follows:"
        "TRUE: The claim is verified as TRUE based on your knowledge."
        "FALSE: The claim is verified as FALSE based on your knowledge."
        "HALF-TRUE: The claim is verified as HALF-TRUE due to insufficient knowledge leading to uncertainty, or because the claim itself is partially true."
        "Based on the below claim and your own knowledge, determine the veracity of the claim and provide your reasoning."
        "Please strictly output in the following format:"
        "Verdict: [TRUE/FALSE/HALF-TRUE]. \nExplanation: [your reasoning here]",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="<|im_end|>",
)


averitec_cevi = Conversation(
   system="A chat between a curious human and an artificial intelligence assistant. " 
        "You are a fact-checking assistant. You are given a claim along with evidence sentences."
        "Your task is to label the overall veracity of the claim based on the provided evidence sentences."
        "Please reason step by step and explain how you reached your conclusion based on the evidences." 
        "The Label Definitions are as follows:"
        "Supported: The claim is supported by the arguments and evidences presented."
        "Refuted: The claim is contradicted by the arguments and evidence presented."
        "Not Enough Evidence: There is not enough evidence to support or refute the claim."
        "Based on the below claim and evidences, determine the veracity of the claim and provide your reasoning. "
        "Please strictly output in the following format:"
        "Verdict: [Supported/Refuted/Not Enough Evidence]. \nExplanation: [your reasoning here]",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="<|im_end|>",
)

averitec_cev = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " 
        "You are a fact-checking assistant. You are given a claim along with evidence sentences."
        "Your task is to label the overall veracity of the claim based on the provided evidence sentences."
        "The Label Definitions are as follows:"
        "Supported: The claim is supported by the arguments and evidences presented."
        "Refuted: The claim is contradicted by the arguments and evidence presented."
        "Not Enough Evidence: There is not enough evidence to support or refute the claim."
        "Based on the below claim and evidences, determine the veracity of the claim. "
        "Please strictly output in the following format:"
        "Verdict: [label]",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="<|im_end|>",
)

raw_fc_cev = Conversation(
   system="A chat between a curious human and an artificial intelligence assistant. " 
        "You are a fact-checking assistant. You are given a claim along with evidence sentences."
        "Your task is to label the overall veracity of the claim based on the provided evidence sentences."
        "The Label Definitions are as follows:"
        "true: The claim is verified as true or mostly-true by the evidences."
        "false: The claim is verified as false or mostly-false by the evidences."
        "half: The claim is verified as half-true by the evidences or the evidences can not prove the claim."
        "Based on the below claim and evidences, determine the veracity of the claim. "
        "Please strictly output in the following format:"
        "Verdict: [label]",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="<|im_end|>",
)

raw_fc_cv = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " 
        "You are a fact-checking assistant. You are given a claim."
        "Your task is to label the overall veracity of the claim based on your internal knowledge."
        "The Label Definitions are as follows:"
        "true: The claim is verified as true based on your knowledge."
        "false: The claim is verified as false based on your knowledge."
        "half: The claim is verified as half-true due to insufficient knowledge leading to uncertainty, or because the claim itself is partially true."
        "Based on the below claim and your own knowledge, determine the veracity of the claim. "
        "Please strictly output in the following format:"
        "Verdict: [label]",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="<|im_end|>",
)

raw_fc_cvi = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " 
        "You are a fact-checking assistant. You are given a claim." 
        "Your task is to label the overall veracity of the claim based on your internal knowledge."
        "Please reason step by step and explain how you reached your conclusion." 
        "The Label Definitions are as follows:"
        "true: The claim is verified as true based on your knowledge."
        "false: The claim is verified as false based on your knowledge."
        "half: The claim is verified as half due to insufficient knowledge leading to uncertainty, or because the claim itself is partially true."
        "Based on the below claim and your own knowledge, determine the veracity of the claim and provide your reasoning."
        "Please strictly output in the following format:"
        "Verdict: [true/false/half]. \nExplanation: [your reasoning here]",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="<|im_end|>",
)

default_conversation = raw_fc_cv
# default_conversation = raw_fc_cvi
# default_conversation = raw_fc_cev
# default_conversation = averitec_cev
# default_conversation = averitec_cevi
# default_conversation = liar_raw_cv
# default_conversation = liar_raw_cvi 
# default_conversation = liar_raw_cev


def get_default_conv_template():
    return default_conversation

if __name__ == "__main__":
    print(default_conversation.get_prompt())
