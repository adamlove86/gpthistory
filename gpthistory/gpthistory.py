import click
import json
import os
import pandas as pd
import logging
from gpthistory.helpers import extract_text_parts, generate_embeddings, calculate_top_titles

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def main():
    """
    Simple CLI for searching within chat data
    """
    pass

@main.command()
@click.option('--file', type=click.Path(exists=True), help='Input file')
def build_index(file):
    """
    Build an index from a given chat data file
    """
    # Determine the index path based on the conversations.json file location
    conversations_dir = os.path.dirname(os.path.abspath(file))
    index_dir = os.path.join(conversations_dir, '.gpthistory')
    INDEX_PATH = os.path.join(index_dir, 'chatindex.csv')

    # Ensure the index directory exists
    os.makedirs(index_dir, exist_ok=True)

    # Load the chat data from the given file
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chat_ids = []
    section_ids = []
    texts = []
    for entry in data:
        for k, v in entry['mapping'].items():
            text_data = extract_text_parts(v)
            if text_data and text_data[0]:
                # Add relevant chat information to the index
                chat_ids.append(entry['id'])
                section_ids.append(k)
                texts.append(text_data[0])
    logger.info(f"Conversations found in file: {len(chat_ids)}")
    df = pd.DataFrame({'chat_id': chat_ids, 'section_id': section_ids, 'text': texts})
    df = df[~df.text.isna()]
    df['id'] = df['chat_id']
    df.set_index("id", inplace=True)

    # Handle incremental index updates
    if os.path.exists(INDEX_PATH):
        current_df = pd.read_csv(INDEX_PATH, sep='|')
        current_df['id'] = current_df['chat_id']
        current_df.set_index("id", inplace=True)
        # Identify new conversations
        new_ids = df.index.difference(current_df.index)
        rows_only_in_df = df.loc[new_ids]
        logger.info(f"Existing index found at {INDEX_PATH}")
        logger.info(f"Number of conversations already indexed: {len(current_df)}")
        logger.info(f"Number of new conversations to index: {len(rows_only_in_df)}")
    else:
        current_df = pd.DataFrame()
        rows_only_in_df = df
        logger.info("No existing index found. Starting a new index.")

    if len(rows_only_in_df) == 0:
        logger.info("No new conversations to index.")
        return

    # Process and save embeddings incrementally
    batch_size = 100
    total_batches = (len(rows_only_in_df) + batch_size -1) // batch_size
    for batch_num, i in enumerate(range(0, len(rows_only_in_df), batch_size)):
        batch_df = rows_only_in_df.iloc[i:i+batch_size]
        logger.info(f"Processing batch {batch_num+1}/{total_batches}")
        embeddings = generate_embeddings(batch_df.text.tolist())
        # Use .loc to avoid SettingWithCopyWarning
        batch_df = batch_df.copy()
        batch_df.loc[:, 'embeddings'] = embeddings
        # Append to index file incrementally
        if not os.path.exists(INDEX_PATH) and batch_num == 0 and current_df.empty:
            # First batch, write headers
            batch_df.reset_index().to_csv(INDEX_PATH, sep='|', index=False, mode='w')
        else:
            # Append without headers
            batch_df.reset_index().to_csv(INDEX_PATH, sep='|', index=False, mode='a', header=False)
    logger.info(f"Total new conversations processed: {len(rows_only_in_df)}")

    # Optionally, you can read the updated index file to get the final_df
    final_df = pd.read_csv(INDEX_PATH, sep='|')
    logger.info(f"Total conversations in index after update: {len(final_df)}")
    logger.info(f"Index built and stored at: {INDEX_PATH}")

@main.command()
@click.argument('keyword', required=True)
@click.option('--file', type=click.Path(exists=True), help='Input file (to locate the index)')
def search(keyword, file):
    """
    Search a keyword within the index
    """
    # Determine the index path based on the conversations.json file location
    if file:
        conversations_dir = os.path.dirname(os.path.abspath(file))
        index_dir = os.path.join(conversations_dir, '.gpthistory')
        INDEX_PATH = os.path.join(index_dir, 'chatindex.csv')
    else:
        # Default to home directory if file not provided
        index_dir = os.path.join(os.path.expanduser('~'), '.gpthistory')
        INDEX_PATH = os.path.join(index_dir, 'chatindex.csv')

    logger.info("Searching for keyword: %s", keyword)
    if os.path.exists(INDEX_PATH):
        df = pd.read_csv(INDEX_PATH, sep='|')
        df['embeddings'] = df.embeddings.apply(lambda x: [float(t) for t in json.loads(x)])
        filtered = df[df.text.str.contains(keyword, case=False)]

        # Calculate top titles and their corresponding chat IDs
        chat_ids, top_titles, top_scores = calculate_top_titles(df, keyword)

        for i, t in enumerate(top_titles):
            logger.info("Chat ID: %s", chat_ids[i])
            logger.info("Content: %s", t)
            logger.info("ChatGPT Conversation link: https://chat.openai.com/c/%s", chat_ids[i])
            logger.info("--------------------------------------")
    else:
        click.echo("Index not found. Please build the index first.")
        return

if __name__ == "__main__":
    main()
