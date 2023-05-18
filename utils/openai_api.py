# Helper functions for querying OpenAI API models
import openai
import tiktoken


def tok2id(tok, tokenizer_model="p50k_base"):
    encoding = tiktoken.get_encoding(tokenizer_model)
    return encoding.encode(tok)

def set_key_from_file(key_file):
    with open(key_file, "r") as fp:
        key = fp.read()
    openai.api_key = key
