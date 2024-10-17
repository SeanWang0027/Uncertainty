"""Load Dataset."""
import pickle
import wikipediaapi
from datasets import load_dataset


def Load_SQuAD(path: str) -> None:
    """Load the Dataset of SQuAD.

    Args:
        path (str): The path for storing the data in that pkl file.
    """
    # Load SQuAD dataset and tokenizer
    dataset = load_dataset("squad")
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    train_store_path = f'{path}SQuAD_train.pkl'
    validation_store_path = f'{path}SQuAD_validation.pkl'
    train_data = []
    validation_data = []
    for i in range(len(train_dataset)):
        sample_dict = dict()
        sample_dict['id'] = i
        sample_dict['title'] = train_dataset[i]['title']
        sample_dict['context'] = train_dataset[i]['context']
        sample_dict['question'] = train_dataset[i]['question']
        sample_dict['answer'] = train_dataset[i]['answers']['text'][0]
        train_data.append(sample_dict)
    with open(train_store_path, 'ab') as f:
        pickle.dump(train_data, f)
    for i in range(len(validation_dataset)):
        sample_dict = dict()
        sample_dict['id'] = i
        sample_dict['title'] = validation_dataset[i]['title']
        sample_dict['context'] = validation_dataset[i]['context']
        sample_dict['question'] = validation_dataset[i]['question']
        sample_dict['answer'] = validation_dataset[i]['answers']['text'][0]
        validation_data.append(sample_dict)
    with open(validation_store_path, 'ab') as f:
        pickle.dump(validation_data, f)


def Load_WebQuestions(path: str) -> None:
    """Load the WebQuestions Dataset.

    Args:
        path (str): The path for storing the data in that pkl file.
    """
    dataset = load_dataset("web_questions")
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    train_store_path = f'{path}WebQuestions_train.pkl'
    test_store_path = f'{path}WebQuestions_test.pkl'
    train_data = []
    test_data = []
    for i in range(len(train_dataset)):
        sample_dict = dict()
        sample_dict['id'] = i
        sample_dict['question'] = train_dataset[i]['question']
        sample_dict['answers'] = train_dataset[i]['answers']
        train_data.append(sample_dict)
    with open(train_store_path, 'ab') as f:
        pickle.dump(train_data, f)
    for i in range(len(test_dataset)):
        sample_dict = dict()
        sample_dict['id'] = i
        sample_dict['question'] = test_dataset[i]['question']
        sample_dict['answers'] = test_dataset[i]['answers']
        test_data.append(sample_dict)
    with open(test_store_path, 'ab') as f:
        pickle.dump(test_data, f)


def Load_WikiQA(path: str):
    """Load the WikiQA Dataset.

    Args:
        path (str): The path for storing the data in that pkl file.
    """
    dataset = load_dataset("microsoft/wiki_qa")
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    train_store_path = f'{path}WikiQA_train.pkl'
    test_store_path = f'{path}WikiQA_test.pkl'
    train_data = []
    test_data = []
    wiki_wiki = wikipediaapi.Wikipedia('Uncertainty', 'en')
    mark = 0
    for i in range(len(train_dataset)):
        if train_dataset[i]['label'] == 0:
            continue
        sample_dict = dict()
        sample_dict['id'] = mark
        mark += 1
        sample_dict['question'] = train_dataset[i]['question']
        sample_dict['answer'] = train_dataset[i]['answer']
        sample_dict['context'] = wiki_wiki.page(train_dataset[i]['document_title']).summary[:500]
        train_data.append(sample_dict)
        print(mark)
    with open(train_store_path, 'ab') as f:
        pickle.dump(train_data, f)
    mark = 0
    for i in range(len(test_dataset)):
        if test_dataset[i]['label'] == 0:
            continue
        sample_dict = dict()
        sample_dict['id'] = mark
        mark += 1
        sample_dict['question'] = test_dataset[i]['question']
        sample_dict['answer'] = test_dataset[i]['answer']
        sample_dict['context'] = wiki_wiki.page(test_dataset[i]['document_title']).summary[:500]
        test_data.append(sample_dict)
        print(mark)
    with open(test_store_path, 'ab') as f:
        pickle.dump(test_data, f)
