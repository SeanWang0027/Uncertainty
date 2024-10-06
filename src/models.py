"""Language model classes."""
import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer


_VALID_LM_NAMES = {
    'gpt2',
    'gpt2-medium',
    'gpt2-large',
    'gpt2-xl',
    'mistralai/Mistral-7B-v0.1',
    'tiiuae/falcon-7b',
    'mosaicml/mpt-7b',
    '01-ai/Yi-6B',
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Meta-Llama-3.1-8B',
    'meta-llama/Llama-2-13b'
}


_TOKEN = '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'


class LanguageModel(object):
    """Language model wrapper."""
    def __init__(self, model_name: str) -> None:
        """Initialize the LanguageModel object.

        Args:
            model_name (str): The name of the model.
        """
        assert model_name in _VALID_LM_NAMES, f'Invalid model name: {model_name}'
        self._model_name = model_name
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name, token=_TOKEN).to(self._device)

    def generate_response(self, prompt: str, num_responses: int, max_new_tokens=20, num_return_sequences=1) -> List:
        """Prompts model for responses using Hugging Face's transformers library.

        Args:
            prompt (str): The prompt for generation.
            num_responses (int): Time for sampling for a single point.
            max_new_tokens (int): The max new generated tokens.
            num_return_sequences: How many one time generation can return.

        Returns:
            A list that contained a list of responses.
        """
        responses = list()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        # Get input_ids and attention_mask
        input_ids = inputs.input_ids.to(self._device)
        attention_mask = inputs.attention_mask.to(self._device)  # Create the attention mask
        pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id
        for _ in range(num_responses):
            output = self._model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=num_return_sequences,
                        pad_token_id=pad_token_id
                    )
            generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
            generated_response = generated_text[len(prompt):].strip()
            responses.append(generated_response)
        return responses
