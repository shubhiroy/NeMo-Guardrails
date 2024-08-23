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

import asyncio
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import EmbeddingModel


def get_executor():
    from . import embeddings_executor

    return embeddings_executor


class AzureEmbeddingModel(EmbeddingModel):
    """Embedding model using Azure OpenAI.

    This class represents an embedding model that utilizes the Azure OpenAI API
    for generating text embeddings.

    Args:
        embedding_model (str): The name of the Azure OpenAI deployment model (e.g., "text-embedding-ada-002").
        api_base (str): The base URL for the Azure OpenAI resource.
        api_key (str): The API key for the Azure OpenAI resource.
        api_version (str): The version of the Azure OpenAI API to use.
    """

    engine_name = "AzureOpenAI"

    # Lookup table for model embedding dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        # Add more models and their dimensions here if needed
    }

    def __init__(
        self, embedding_model: str, api_base: str, api_key: str, api_version: str
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "Could not import openai, please install it with "
                "`pip install openai`."
            )
        # Set Azure OpenAI API credentials
        self.client = OpenAI()
        self.client.api_type = "azure"
        self.client.api_base = api_base
        self.client.api_version = api_version  # or the version you're using
        self.client.api_key = api_key

        self.embedding_model = embedding_model
        self.embedding_size = self._get_embedding_dimension()

    def _get_embedding_dimension(self):
        """Retrieve the embedding dimension for the specified model."""
        if self.embedding_model in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[self.embedding_model]
        else:
            raise ValueError(
                f"Unknown model: {self.embedding_model}. Please add its dimensions to MODEL_DIMENSIONS."
            )

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        """Asynchronously encode a list of documents into their corresponding embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The list of embeddings, where each embedding is a list of floats.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor(), self.encode, documents)
        return result

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into their corresponding embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The list of embeddings, where each embedding is a list of floats.

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            response = self.client.Embedding.create(
                model=self.embedding_model, input=documents
            )
            return [item["embedding"] for item in response["data"]]
        except self.client.error.OpenAIError as e:
            raise RuntimeError(f"Failed to retrieve embeddings: {e}")
