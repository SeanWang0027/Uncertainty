import torch
from transformers import AutoTokenizer,AutoModelForCausalLM 
from datasets import load_dataset


PROMPT = """Answer these questions. Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930? A: Sinclair LewisQ: Where in England was Dame Judi Dench born? A: York Q: In which decade did Billboard magazine first publish an American hit chart? A: 30s Q: From which country did Angola achieve independence in 1975? A: Portugal Q: Which city does David Soul come from? A: Chicago Q: Who won Super Bowl XX? A: Chicago Bears Q: Which was the first European country to abolish capital punishment? A: Norway Q: In which country did the widespread use of ISDN begin in 1988? A: Japan Q: What is Bruce Willis’ real first name? A: Walter Q: Which William wrote the novel Lord Of The Flies? A: Golding Q: Which innovation for the car was developed by Prince Henry of Prussia in 1911? A: Windshield wipers Q: How is musician William Lee Conley better known? A: Big Bill Broonzy Q: How is Joan Molinsky better known? A: Joan Rivers Q: In which branch of the arts is Patricia Neary famous? A: Ballet Q: Which country is Europe’s largest silk producer? A: Italy Q: The VS-300 was a type of what? A: Helicopter Q: At which university did Joseph Goebbels become a doctor of philosophy? A: Heidelberg Q: Which prince is Queen Elizabeth II’s youngest son? A: Edward Q: When did the founder of Jehovah’s Witnesses say the world would end? A: 1914 Q: Who found the remains of the Titanic? A: Robert Ballard Q: Who was the only Spice Girl not to have a middle name? A: Posh Spice Q: What are the international registration letters of a vehicle from Algeria? A: DZ Q: How did Jock die in Dallas? A: Helicopter accident Q: What star sign is Michael Caine? A: Pisces Q: Who wrote the novel Evening Class?A: Maeve Binchy Q: Which country does the airline Air Pacific come from? A: FijiQ: In which branch of the arts does Allegra Kent work? A: Ballet Q: Banting and Best pioneered the use of what? A: Insulin Q: Who directed the movie La Dolce Vita? A: Federico Fellini Q: Which country does the airline LACSA come from? A: Costa Rica Q: Who directed 2001: A Space Odyssey? A: Stanley Kubrick Q: Which is the largest of the Japanese Volcano Islands? A: Iwo Jima Q: """
TOKEN = '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'
MODEL_NAME = 'meta-llama/Meta-Llama-3-8B'


dataset_name = "trivia_qa"
dataset = load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
validation_dataset = dataset['test']
qa_pair = []
for i in range(len(train_dataset)):
    qa_pair.append({'question': train_dataset[i]['question'], 'answer': train_dataset[i]['answers']['text'][0]})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=TOKEN).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
temperature = 2.0

# for i in range(len(train_dataset)):
#     input = PROMPT + qa_pair[i]['question'] + ' A: '
#     print(input)
#     max_new_tokens = len(tokenizer.encode(qa_pair[i]['answer'], add_special_tokens=False))
#     inputs = tokenizer(input, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens= max_new_tokens,
#             return_dict_in_generate=True,
#             output_scores=True,
#             output_hidden_states=True,
#             temperature=temperature,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id,
#         )
#     generated_token_ids = outputs.sequences[0]
#     generated_text = tokenizer.decode(generated_token_ids[len(inputs[0]):], skip_special_tokens=True)
#     print(generated_text, qa_pair[i]['answer'])
#     logits = outputs.scores
#     hidden = outputs.decoder_hidden_states
#     break
for i in range(len(train_dataset)):
    input_text = PROMPT + qa_pair[i]['question'] + ' A: '
    print(input_text)
    max_new_tokens = len(tokenizer.encode(qa_pair[i]['answer'], add_special_tokens=False))
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,  # Logits are returned
            output_hidden_states=True,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,  # Prevents invalid tokens
        )
    generated_token_ids = outputs.sequences[0]
    input_token_length = inputs["input_ids"].shape[1]
    new_tokens = generated_token_ids[input_token_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("Generated: ", generated_text)
    print("Expected: ", qa_pair[i]['answer'])
    logits = outputs.scores
    # for idx, logit in enumerate(logits):
    #     generated_token = new_tokens[idx].item()
    #     generated_token_logit = logit[0, generated_token]
    #     sorted_logits, sorted_indices = torch.sort(logit, descending=True)
    #     print(sorted_logits)
    #     if generated_token_logit.item() != float('-inf'):
    #         print(f"Logits for generated token {idx + 1} ({tokenizer.decode([generated_token])}): {generated_token_logit.item()}")
    # break
    for idx, logit in enumerate(logits):
        # Apply softmax to the logits to get probabilities
        probabilities = torch.softmax(logit, dim=-1)

        # Get the token ID of the generated token at this step
        token_id = new_tokens[idx].item()

        # Get the probability of the generated token
        token_probability = probabilities[0, token_id].item()

        # Sort the probabilities to find the top-k tokens
        top_k_probs, top_k_indices = torch.topk(probabilities, k=5, dim=-1)

        # Decode the top-k tokens
        print(f"Token generated at step {idx + 1}: {tokenizer.decode([token_id])}")
        print(f"Probability of generated token: {token_probability:.4f}")
        print("Top 5 tokens and their probabilities:")
        for top_token_id, top_prob in zip(top_k_indices[0], top_k_probs[0]):
            top_token_str = tokenizer.decode([top_token_id])
            print(f"Token: {top_token_str}, Probability: {top_prob.item():.4f}")

    break
