"""
Various utility functions are saved here.
"""


def extract_model_name(filename):
    """
    Extract the model name from a given filename.
    Expects the filename to be in the format:
    - fc_results__model_<MODEL_NAME>__<GROUNDED_SETTING>__<RAG_SETTING>.jsonl

    Parameters
    ----------
    filename : str
        A string representing the file path or name.

    Returns
    -------
    str
        The extracted model name component from the filename.
    """
    model_chunk = filename.split("__")[1]
    return model_chunk.split("_")[1]


def extract_num_rag_items(filename):
    """
    Extract the rag items from a given filename.
    Expects the filename to be in the format:
    - fc_results__model_<MODEL_NAME>__<GROUNDED_SETTING>__<RAG_SETTING>.jsonl

    Parameters
    ----------
    filename : str
        A string representing the file path or name.

    Returns
    -------
    int
        The number of RAG items used in the filename. `None` if RAG was not used.
    """
    rag_chunks = filename.split("__")
    if len(rag_chunks) == 4 and "grounded" not in rag_chunks[-2]:
        rag_chunk = rag_chunks[-2]
    else:
        rag_chunk = rag_chunks[-1].replace(".jsonl", "")

    if "items" in rag_chunk:
        item_list = rag_chunk.split("_")
        num_items = int(item_list[1].split(".")[0])
    else:
        num_items = None
    return num_items


def used_web_search(filename):
    """
    Return True if web search was used in the filename, False otherwise.

    Expects the filename to be in the format:
    - fc_results__model_<MODEL_NAME>__<GROUNDED_SETTING>__<RAG_SETTING>.jsonl
    - Grounded setting may not be present, depending on the model, and may instead
    be in the model name. E.g. as "gpt-4o-search-preview".

    Parameters
    ----------
    filename : str
        A string representing the file path or name.

    Returns
    -------
    int
        The number of RAG items used in the filename. `None` if RAG was not used.
    """

    # Catches all OpenAI search models
    model_name = extract_model_name(filename)
    if "search" in model_name:
        return True

    # Filters out non-Google models (which do not have search).
    if "grounded" not in filename:
        return False

    # Now we distinguish between Google settings
    if "ungrounded" in filename:
        return False
    else:
        return True


def extract_modelname_ragitems(filename):
    """
    Parses a filename to extract the model name and the number of items.
    The filename is expected to be in the format: 'prefix__model_<model_name>__items_<num_items>.ext'.
    If the filename does not conform to this format, the function returns (None, None).

    Parameters:
    -----------
    - filename (str): The filename to parse.

    Returns:
    -----------
    - tuple: A tuple containing:
        - model_name (str or None): The extracted model name.
        - num_items (int or None): The extracted number of RAG items, or None
            if RAG was not used.
    """
    parts = filename.split("__")
    if len(parts) < 3:
        raise ValueError(f'Filename "{filename}" does not match the expected format.')

    model_name = parts[1].replace("model_", "")
    if "items" in parts[2]:
        item_list = parts[2].split("_")
        num_items = int(item_list[1].split(".")[0])
    else:
        num_items = None

    return model_name, num_items
