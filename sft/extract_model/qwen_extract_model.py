import itertools
import logging
import random
from typing import Optional, List, Tuple, Union

import torch
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer, Qwen2TokenizerFast
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast


class Qwen2ForExtractLM(Qwen2ForCausalLM):
    """
    A custom LlamaForCausalLM model extension that implements permutation-based training.
    This class finds the best permutation of segments in the answer text during training
    to achieve the minimum loss.
    """

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = None
        self.answer_start_ids = None  # Token IDs that mark the beginning of an answer
        self.answer_end_ids = None  # Token IDs that mark the end of an answer
        self.split_ids = None  # Token IDs used to split answer into segments
        self.perms_limit = -1  # Limit on the number of permutations to consider (-1 means no limit)

    def set_tokenizer(self, tokenizer: Qwen2Tokenizer | Qwen2TokenizerFast):
        """Set the tokenizer for text encoding."""
        self.tokenizer = tokenizer

    def set_answer_start_ids(
            self,
            text: str = None,
            token_ids: torch.LongTensor = None,  # (seq_length)
    ):
        """
        Set the token IDs that mark the beginning of an answer.
        Can be set using either raw text or directly with token IDs.
        """
        if text is not None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not set, cannot encode text")
            encoded = self.tokenizer.encode(
                text,
                return_tensors="pt",
                add_special_tokens=False
            )
            self.answer_start_ids = encoded[0]
        if token_ids is not None:
            self.answer_start_ids = token_ids

    def set_answer_end_ids(
            self,
            text: str = None,
            token_ids: torch.LongTensor = None,  # (seq_length)
    ):
        """
        Set the token IDs that mark the end of an answer.
        Can be set using either raw text or directly with token IDs.
        """
        if text is not None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not set, cannot encode text")
            encoded = self.tokenizer.encode(
                text,
                return_tensors="pt",
                add_special_tokens=False
            )
            self.answer_end_ids = encoded[0]
        if token_ids is not None:
            self.answer_end_ids = token_ids

    def set_split_ids(
            self,
            text: str = None,
            token_ids: torch.LongTensor = None,  # (seq_length)
    ):
        """
        Set the token IDs used to split answers into segments.
        Can be set using either raw text or directly with token IDs.
        """
        if text is not None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not set, cannot encode text")
            encoded = self.tokenizer.encode(
                text,
                return_tensors="pt",
                add_special_tokens=False
            )
            self.split_ids = encoded[0]
        if token_ids is not None:
            self.split_ids = token_ids

    def set_perms_limit(self, perms_limit: int):
        """Set the maximum number of permutations to consider."""
        self.perms_limit = perms_limit

    def find_first_subsequence(
            self,
            haystack: torch.LongTensor,  # (seq_length)
            needle: torch.LongTensor,  # (sub_seq_length)
    ) -> int:
        """
        Find the first occurrence of a subsequence (needle) in a larger sequence (haystack).
        Returns the starting index of the subsequence or -1 if not found.
        """
        needle = needle.to(haystack.device)

        if len(needle) > len(haystack):
            return -1
        for i in range(len(haystack) - len(needle) + 1):
            if torch.all(haystack[i: i + len(needle)] == needle):
                return i
        return -1

    def _get_permutation_ids(
            self,
            answer_ids: torch.LongTensor,
            split_ids: torch.LongTensor,
    ):
        """
        Split the answer into segments based on split_ids and generate all possible
        permutations of these segments while preserving the split tokens.

        Returns a list of permuted token sequences.
        """
        split_ids = split_ids.to(answer_ids.device)

        if answer_ids.requires_grad or split_ids.requires_grad:
            answer_ids = answer_ids.detach().clone()
            split_ids = split_ids.detach().clone()
        if answer_ids.dim() > 1:
            answer_ids = answer_ids.squeeze()
        if split_ids.dim() > 1:
            split_ids = split_ids.squeeze()

        # Find all occurrences of split_ids in answer_ids
        split_positions = []
        for i in range(len(answer_ids) - len(split_ids) + 1):
            if torch.all(answer_ids[i:i + len(split_ids)] == split_ids):
                split_positions.append((i, i + len(split_ids)))

        # Extract segments between split positions
        segments = []
        start = 0
        for pos_start, pos_end in split_positions:
            if start < pos_start:
                segments.append(answer_ids[start:pos_start])
            start = pos_end

        if start < len(answer_ids):
            segments.append(answer_ids[start:])

        # print("=" * 20)
        # print("answer_ids:", answer_ids)
        # print("segments:", segments)
        # print("=" * 20)

        # Generate all permutations of the segments
        permutations = list(itertools.permutations(segments))

        # Reconstruct sequences with split tokens between segments
        result = []
        for perm in permutations:
            new_sequence = []
            for i, segment in enumerate(perm):
                new_sequence.append(segment)
                if i < len(perm) - 1:
                    new_sequence.append(split_ids)
            flat_sequence = torch.cat(new_sequence)
            result.append(flat_sequence)
        return result

    def _generate_permutation_ids(
            self,
            labels: torch.LongTensor,  # （seq_length）
    ):
        """
        Generate permutations of the answer segments within the labels.

        Returns:
        - insertion_start: Start index where permuted content should be inserted
        - insertion_end: End index where permuted content should be inserted
        - perms_ids: List of permuted token sequences
        - Number of permutations generated
        """
        # Find the first non-padding position in labels (-100 is padding)
        non_negative_index = (labels != -100).nonzero(as_tuple=True)[0]
        # print("non_negative_index:", non_negative_index)
        if non_negative_index.numel() > 0:
            non_negative_index = non_negative_index[0].item()
        else:
            return -1, -1, None, 0

        # Extract the actual answer tokens from labels
        answer_ids = labels.detach().clone()
        answer_ids = answer_ids[non_negative_index:]
        # print(self.tokenizer.decode(answer_ids))

        # Find the answer boundaries using the start and end markers
        answer_start_index = self.find_first_subsequence(answer_ids, self.answer_start_ids) + len(self.answer_start_ids)
        answer_end_index = self.find_first_subsequence(answer_ids, self.answer_end_ids)
        if answer_start_index < 0 or answer_end_index < 0:
            return -1, -1, None, 0
        answer_ids = answer_ids[answer_start_index: answer_end_index]

        # Calculate the absolute positions in the original sequence
        insertion_start = non_negative_index + answer_start_index
        insertion_end = non_negative_index + answer_end_index

        try:
            # Generate all permutations of answer segments
            perms_ids = self._get_permutation_ids(answer_ids, self.split_ids)
            # print(f"Permutations Count: {len(perms_ids)}")
            # print(f"perms_ids: {perms_ids}")

            # Apply permutation limit if set
            if self.perms_limit != -1:
                perms_ids = random.sample(perms_ids, min(len(perms_ids), self.perms_limit))

        except Exception as e:
            logging.error(f"Error in permutation generation: {e}")
            return -1, -1, None, 0

        return (
            insertion_start,
            insertion_end,
            perms_ids,
            len(perms_ids),
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Custom forward function that evaluates all permutations of answer segments
        and selects the permutation with the minimum loss.

        This allows the model to find the optimal ordering of information.
        """
        # print("Entered FullPermForDeepseek.forward()")
        if labels is not None:
            best_input_ids = torch.empty((0, input_ids.size(-1)), dtype=input_ids.dtype, device=input_ids.device)
            best_labels = torch.empty((0, labels.size(-1)), dtype=labels.dtype, device=labels.device)
            for i in range(labels.shape[0]):
                # print("Processing batch item")
                insertion_start, insertion_end, perms_ids, perms_len = self._generate_permutation_ids(labels[i])
                # print("Generated permutations")
                # print("perms_len:", perms_len)
                if perms_len <= 1:
                    # print("=" * 20)
                    # print("Keeping original input", input_ids)
                    # print("=" * 20)
                    best_input_ids = torch.cat([best_input_ids, input_ids[i].unsqueeze(0)], dim=0)
                    best_labels = torch.cat([best_labels, labels[i].unsqueeze(0)], dim=0)
                else:
                    # Find the permutation with the minimum loss
                    with torch.no_grad():
                        min_loss = torch.tensor(9999.0)
                        min_loss_input_ids = input_ids[i].unsqueeze(0)
                        min_loss_labels = labels[i].unsqueeze(0)
                        single_attention_mask = attention_mask[i].unsqueeze(0)
                        for perm_ids in perms_ids:
                            try:
                                # Create a version of the input with this permutation
                                single_input_ids = input_ids[i].detach().clone()
                                single_input_ids[insertion_start:insertion_end] = perm_ids
                                single_input_ids = single_input_ids.unsqueeze(0)
                                single_labels = labels[i].detach().clone()
                                single_labels[insertion_start:insertion_end] = perm_ids
                                single_labels = single_labels.unsqueeze(0)
                                # Evaluate this permutation
                                outputs = super().forward(
                                    input_ids=single_input_ids,
                                    attention_mask=single_attention_mask,
                                    labels=single_labels,
                                    use_cache=False,  # No need for cache in single evaluation
                                    output_hidden_states=False,  # Don't need hidden states for loss calculation
                                    return_dict=True,  # Keep return format as dictionary
                                )
                                current_loss = outputs.loss
                                # print("current_loss:", current_loss)
                                if current_loss < min_loss:
                                    min_loss = current_loss
                                    min_loss_input_ids = single_input_ids
                                    min_loss_labels = single_labels
                            except Exception as e:
                                print(f"Error in processing permutation: {e}")
                                logging.error(f"Error in processing permutation: {e}")
                                continue
                    best_input_ids = torch.cat([best_input_ids, min_loss_input_ids], dim=0)
                    best_labels = torch.cat([best_labels, min_loss_labels], dim=0)
                    # print("=" * 20)
                    # print("ori_labels:", input_ids[i])
                    # print("new_labels:", best_input_ids[i])
                    # print("=" * 20)
            input_ids = best_input_ids
            labels = best_labels

        # Call the parent class's forward method with the best permutation
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )