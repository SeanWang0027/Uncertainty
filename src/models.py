"""Language model classes."""
import torch
import math
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, PreTrainedTokenizer
import torch.nn.functional as F


_VALID_LM_NAMES = {
    'gpt2',
    'mistralai/Mistral-7B-v0.1',
    'tiiuae/falcon-7b',
    'mosaicml/mpt-7b',
    '01-ai/Yi-6B',
    'meta-llama/Meta-Llama-3-8B'
}


_TOKEN = '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'


class StopOnNewline(StoppingCriteria):
    """Stop Rules for generation."""
    def __init__(self, tokenizer: PreTrainedTokenizer, newline_token_id: str):
        """Initialize the StopOnNewLine rules.

        Args:
            tokenizer (PreTrainedTokenizer): A tokenizer object.
            newline_token_id: The string that meets the stop rule.
        """
        self.newline_token_id = newline_token_id
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.tensor, scores: float):
        """Check if the last generated token is a newline.

        Args:
            input_ids: The tensor.
            scores: The score for the tensor.
        """
        return input_ids[0, -1] == self.newline_token_id


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

    def generate_response(self, prompt: str, answer: str, num_responses: int, max_new_tokens=20, num_return_sequences=1) -> Dict:
        """Prompts model for responses using Hugging Face's transformers library.

        Args:
            prompt (str): The prompt for generation.
            answer (str): The answer of the prompt.
            num_responses (int): Time for sampling for a single point.
            max_new_tokens (int): The max new generated tokens.
            num_return_sequences: How many one time generation can return.

        Returns:
            A dictionary that contained a list of responses.
        """
        responses = dict()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        newline_token_id = self._tokenizer.encode("\n", add_special_tokens=False)[0]
        # Initialize stopping criteria list with the custom criterion
        stopping_criteria = StoppingCriteriaList([StopOnNewline(self._tokenizer, newline_token_id)])
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
                        pad_token_id=pad_token_id,
                        stopping_criteria=stopping_criteria
                    )
            generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
            generated_response = generated_text[len(prompt)-1:].strip()
            if generated_response not in responses:
                responses[generated_response] = 1
            else:
                responses[generated_response] += 1
        return responses

    def resample(self, exp: dict) -> Dict:
        """Resample the prompt + answer probabilities distribution.

        Args:
            exp (dict): A dict that contains answer.

        Returns:
            A dictionary of the object.
        """
        for answer in exp['candidates_logit'].keys():
            input = exp['prompt'] + answer
            input_ids = self._tokenizer(input, return_tensors="pt").input_ids.to(self._device)
            with torch.no_grad():
                outputs = self._model(input_ids)
                logits = outputs.logits
            prompt_ids = self._tokenizer(exp['prompt'], return_tensors="pt").input_ids
            prompt_length = prompt_ids.shape[1]
            answer_logits = logits[:, prompt_length - 2:-1, :]
            answer_probs = F.softmax(answer_logits, dim=-1)
            # Retrieve the actual answer tokens
            answer_ids = input_ids[0, prompt_length-1:].tolist()
            answer_tokens = self._tokenizer.convert_ids_to_tokens(answer_ids)
            for i, token in enumerate(answer_tokens):
                token_id = answer_ids[i]
                token_prob = answer_probs[0, i, token_id].item()
                if token_prob < 1e-4:
                    continue
                exp['candidates_logit'][answer] *= token_prob
        exp_values = {key: math.exp(value) for key, value in exp['candidates_logit'].items()}
        sum_exp_values = sum(exp_values.values())
        exp['candidates_logit'] = {key: value / sum_exp_values for key, value in exp_values.items()}
        print(exp['candidates_logit'])
        return exp
# Open-book QA; SQuad; wiki-QA
