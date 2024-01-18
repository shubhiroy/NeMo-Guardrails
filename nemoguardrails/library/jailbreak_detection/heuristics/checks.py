# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import os

device = os.environ.get("JAILBREAK_CHECK_DEVICE", "cpu")
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)


def get_perplexity(input_string: str) -> bool:
    encodings = tokenizer(input_string, return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = list()
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    perplexity = torch.exp(torch.stack(nlls).mean())

    return perplexity.cpu().detach().numpy().item()


def check_jb_lp(input_string: str, lp_threshold: float) -> dict:
    perplexity = get_perplexity(input_string)
    jb_lp = len(input_string) / perplexity >= lp_threshold
    result = {"jailbreak": jb_lp}
    return result


def check_jb_ps_ppl(input_string: str, ps_ppl_threshold: float) -> dict:
    split_string = input_string.strip().split()
    # Not useful to evaluate GCG-style attacks on strings less than 20 "words"
    if len(split_string) < 20:
        return {"jailbreak": False}

    suffix = " ".join(split_string[-20:-1])
    prefix = " ".join(split_string[0:19])

    suffix_ppl = get_perplexity(suffix)
    prefix_ppl = get_perplexity(prefix)

    if suffix_ppl >= ps_ppl_threshold or prefix_ppl >= ps_ppl_threshold:
        jb_ps = True
    else:
        jb_ps = False

    result = {"jailbreak": jb_ps}
    return result
