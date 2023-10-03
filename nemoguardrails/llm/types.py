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

from enum import Enum


class Task(Enum):
    """The various tasks that can be performed by the LLM."""

    GENERAL = "general"
    GENERATE_USER_INTENT = "generate_user_intent"
    GENERATE_NEXT_STEPS = "generate_next_steps"
    GENERATE_BOT_MESSAGE = "generate_bot_message"
    GENERATE_VALUE = "generate_value"
    GENERATE_FLOW_FROM_INSTRUCTIONS = "generate_flow_from_instructions"
    GENERATE_FLOW_FROM_NAME = "generate_flow_from_name"

    FACT_CHECKING = "fact_checking"
    JAILBREAK_CHECK = "jailbreak_check"
    OUTPUT_MODERATION = "output_moderation"
    OUTPUT_MODERATION_V2 = "output_moderation_v2"
    CHECK_HALLUCINATION = "check_hallucination"
