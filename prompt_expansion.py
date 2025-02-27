import os
import random
import sys
import torch
import math

# Get the parent directory of 'comfy' and add it to the Python path
comfy_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(comfy_parent_dir)

# Suppress console output
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

# Import the required modules
import comfy.model_management as model_management
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, LogitsProcessorList
from comfy.model_patcher import ModelPatcher
from .util import join_prompts, remove_empty_str
from server import PromptServer

# Restore the original stdout
sys.stdout = original_stdout

fooocus_expansion_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                      'fooocus_expansion'))
# fooocus_magic_split = [  # Removed magic split as it's not part of the Fooocus logic
#     ', extremely',
#     ', intricate,',
# ]
dangrous_patterns = '[]【】()（）|:：'

# limitation of np.random.seed(), called from transformers.set_seed()
SEED_LIMIT_NUMPY = 2**32
neg_inf = - 8192.0

def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace('  ', ' ')
    return x.strip(",. \r\n")


def remove_pattern(x, pattern):
    for p in pattern:
        x = x.replace(p, '')
    return x


class FooocusExpansion:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(fooocus_expansion_path)

        positive_words = open(os.path.join(fooocus_expansion_path, 'positive.txt'),
                              encoding='utf-8').read().splitlines()
        positive_words = ['Ġ' + x.lower() for x in positive_words if x != '']

        self.logits_bias = torch.zeros((1, len(self.tokenizer.vocab)), dtype=torch.float32) + neg_inf

        debug_list = []
        for k, v in self.tokenizer.vocab.items():
            if k in positive_words:
                self.logits_bias[0, v] = 0
                debug_list.append(k[1:])

        print(f'Fooocus V2 Expansion: Vocab with {len(debug_list)} words.')

        self.model = AutoModelForCausalLM.from_pretrained(fooocus_expansion_path)
        self.model.eval()

        load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()

        # MPS hack
        if model_management.is_device_mps(load_device):
            load_device = torch.device('cpu')
            offload_device = torch.device('cpu')

        use_fp16 = model_management.should_use_fp16(device=load_device)

        if use_fp16:
            self.model.half()

        self.model.to(load_device)  # Move the model to the device

        self.patcher = ModelPatcher(self.model, load_device=load_device, offload_device=offload_device)
        print(f'Fooocus Expansion engine loaded for {load_device}, use_fp16 = {use_fp16}.')

    @torch.no_grad()
    @torch.inference_mode()
    def logits_processor(self, input_ids, scores):
        assert scores.ndim == 2 and scores.shape[0] == 1
        self.logits_bias = self.logits_bias.to(scores)

        bias = self.logits_bias.clone()
        bias[0, input_ids[0].to(bias.device).long()] = neg_inf
        bias[0, 11] = 0  # What is the token ID 11?  Needs to be investigated.  Might be a comma.

        return scores + bias

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, prompt, seed):
        if prompt == '':
            return ''

        seed = int(seed) % SEED_LIMIT_NUMPY
        set_seed(seed)
        prompt = safe_str(prompt) + ','  # Add comma as in original Fooocus code

        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data['input_ids'] = tokenized_kwargs.data['input_ids'].to(self.patcher.load_device)
        tokenized_kwargs.data['attention_mask'] = tokenized_kwargs.data['attention_mask'].to(self.patcher.load_device)

        current_token_length = int(tokenized_kwargs.data['input_ids'].shape[1])
        max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
        max_new_tokens = max_token_length - current_token_length

        if max_new_tokens == 0:
            return prompt[:-1]

        # https://huggingface.co/blog/introducing-csearch
        # https://huggingface.co/docs/transformers/generation_strategies
        features = self.model.generate(**tokenized_kwargs,
                                       top_k=100,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=True,
                                       logits_processor=LogitsProcessorList([self.logits_processor]))

        response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        result = safe_str(response[0])

        return result


class FooocusV2Expansion:
    # Define the expected input types for the node
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": True}),
                "prompt_seed": ("INT", {"default": -1, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "seed_behavior": (["Random", "Fixed", "Forward Prompt Seed", "Random Forward Prompt Seed"], {"default": "Random"}),
                "log": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "persisted_seed": ("INT", {"default": -1, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("final_prompt", "seed",)
    FUNCTION = "expand_prompt"  # Function name

    CATEGORY = "Fooocus V2 Expansion"  # Category for organization

    @staticmethod
    @torch.no_grad()
    def expand_prompt(positive_prompt, prompt_seed, seed_behavior="Random", log = True, persisted_seed=-1, last_prompt="", last_prompt_seed=-1, unique_id=None):
        expansion = FooocusExpansion()

        last_prompt = positive_prompt
        last_prompt_seed = prompt_seed

        prompt = remove_empty_str([safe_str(positive_prompt)], default='')[0]

        if log == True:
            print(f"[Fooocus V2 Expansion] Original Prompt: {prompt}")

        if log == True:
          print(f"[Fooocus V2 Expansion] Original Seed: {prompt_seed}")

        if seed_behavior != "Fixed" or persisted_seed == -1:
          persisted_seed = prompt_seed
        
        if seed_behavior == "Fixed":
          prompt_seed = persisted_seed

        if prompt_seed == -1 or seed_behavior == "Random" or seed_behavior == "Random Forward Prompt Seed":
          prompt_seed = random.randint(1, 0xffffffffffffffff)
          PromptServer.instance.send_sync("Fooocus.V2.Expansion.updateSeed", { "prompt_seed": prompt_seed, "id": unique_id })
          if log == True:
            print(f"[Fooocus V2 Expansion] Seed == -1 or seed_behavior == 'Random' -> New Seed: {prompt_seed}")

        if prompt_seed < 0:
          prompt_seed =- prompt_seed
          if log == True:
            print(f"[Fooocus V2 Expansion] Seed < 0 -> New Seed: {prompt_seed}")
        
        positive_prompt_expansion = expansion(prompt, prompt_seed)
        final_prompt = join_prompts(prompt, positive_prompt_expansion)

        if prompt_seed > 0xffffffffffffffff:
          prompt_seed = int(prompt_seed) % SEED_LIMIT_NUMPY
          if log == True:
            print(f"[Fooocus V2 Expansion] Seed > 0xffffffffffffffff -> New Seed: {prompt_seed}")

        truncated_seed = int(prompt_seed) % SEED_LIMIT_NUMPY

        if seed_behavior == "Forward Prompt Seed" or seed_behavior == "Random Forward Prompt Seed":
          prompt_seed = truncated_seed

        if log == True:
            print(f"[Fooocus V2 Expansion] Final Seed: {prompt_seed}")
            print(f"[Fooocus V2 Expansion] Prompt Seed: {truncated_seed}")
            print(f"[Fooocus V2 Expansion] New Suffix: {positive_prompt_expansion}")
            print(f"[Fooocus V2 Expansion] Final Prompt: {final_prompt}")

        return final_prompt, prompt_seed

    @classmethod
    def IS_CHANGED(cls, positive_prompt, prompt_seed, seed_behavior="Random", log = True, persisted_seed=-1, unique_id=None):
      # return FooocusV2Expansion.expand_prompt(positive_prompt, prompt_seed, seed_behavior, log, persisted_seed, unique_id)
      return float("NaN")

# Define a mapping of node class names to their respective classes
NODE_CLASS_MAPPINGS = {
    "FooocusV2Expansion": FooocusV2Expansion
}

# A dictionary that contains human-readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FooocusV2Expansion": "Fooocus V2 Expansion"
}
