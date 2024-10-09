import gensim.downloader as api
from numpy import dot
from numpy.linalg import norm

word2vec_model = api.load('word2vec-google-news-300')

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def sentence_to_vector(sentence, model):
    words = sentence.split()  # Simple tokenization
    word_vectors = [model[word] for word in words if word in model]
    
    if len(word_vectors) == 0:
        return None  
    
    sentence_vector = sum(word_vectors) / len(word_vectors)
    return sentence_vector

sentence = "city of troy"
word = "troy"

if word in word2vec_model:
    word_vector = word2vec_model[word]
else:
    print(f"'{word}' is not in the vocabulary.")
    exit()

# Compute the sentence vector
sentence_vector = sentence_to_vector(sentence, word2vec_model)

if sentence_vector is not None:
    similarity = cosine_similarity(sentence_vector, word_vector)
    print(f"Cosine similarity between the sentence and the word '{word}': {similarity}")
else:
    print("None of the words in the sentence are in the vocabulary.")
