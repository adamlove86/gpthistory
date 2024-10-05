import json
import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import logging
import tiktoken  # Import tiktoken library

# Load environment variables
load_dotenv()

# Instantiate the OpenAI client
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Maximum tokens allowed by the model
MAX_TOKENS = 8192  # Adjust if necessary based on the model

# Choose the embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"  # Update if using a different model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_parts(data):
    text_parts = []
    message = data.get('message')
    if message:
        content = message.get('content')
        if content and content.get('content_type') == 'text':
            text_parts.extend(content.get('parts', []))
    return text_parts

def split_into_batches(array, batch_size):
    for i in range(0, len(array), batch_size):
        yield array[i:i + batch_size]

def generate_query_embedding(query):
    # Tokenize and truncate the query if necessary
    tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    max_tokens = MAX_TOKENS - 10
    tokens = tokenizer.encode(query)
    if len(tokens) > max_tokens:
        logger.warning(f"Query exceeds max tokens ({len(tokens)} tokens). Truncating.")
        tokens = tokens[:max_tokens]
        query = tokenizer.decode(tokens)

    response = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def generate_embeddings(conversations):
    embeddings = []
    tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    max_tokens = MAX_TOKENS - 10  # Buffer to ensure we stay within limit

    for i, batch in enumerate(split_into_batches(conversations, 100)):
        logger.info(f"Processing batch: {i + 1}")
        processed_batch = []
        for text in batch:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_tokens:
                logger.warning(f"Text exceeds max tokens ({len(tokens)} tokens). Truncating.")
                tokens = tokens[:max_tokens]
                text = tokenizer.decode(tokens)
            processed_batch.append(text)

        if not processed_batch:
            continue  # Skip if no texts to process

        logger.info(f"Generating Embeddings for batch: {i + 1}")
        response = client.embeddings.create(
            input=processed_batch,
            model=EMBEDDING_MODEL
        )
        tmp_embedding = [item.embedding for item in response.data]
        embeddings += tmp_embedding

    if len(embeddings) > 0:
        logger.info("Total conversations processed: %d", len(conversations))
        logger.info("Total embeddings generated: %d", len(embeddings))
    else:
        logger.info("No new conversations detected")
    return embeddings

def calculate_top_titles(df, query, top_n=1000):
    tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    max_tokens = MAX_TOKENS - 10

    # Truncate query if necessary
    tokens = tokenizer.encode(query)
    if len(tokens) > max_tokens:
        logger.warning(f"Query exceeds max tokens ({len(tokens)} tokens). Truncating.")
        tokens = tokens[:max_tokens]
        query = tokenizer.decode(tokens)

    embedding_array = np.array(df['embeddings'].tolist())
    query_embedding = generate_query_embedding(query)
    dot_scores = np.dot(embedding_array, query_embedding)
    mask = dot_scores >= 0.8
    filtered_dot_scores = dot_scores[mask]
    filtered_texts = df.loc[mask, 'text'].tolist()
    filtered_chat_ids = df.loc[mask, 'chat_id'].tolist()
    sorted_indices = np.argsort(filtered_dot_scores)[::-1][:top_n]
    chat_ids = [filtered_chat_ids[i] for i in sorted_indices]
    top_n_texts = [filtered_texts[i] for i in sorted_indices]
    top_n_dot_scores = filtered_dot_scores[sorted_indices]
    return chat_ids, top_n_texts, top_n_dot_scores
