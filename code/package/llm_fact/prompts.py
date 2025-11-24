"""
Store LLM prompts for data analysis tasks.
"""

RAG_PROMPT = (
    "As an AI fact checker, your task is to evaluate the accuracy of a CLAIM by assigning a label from the 'Truth Scale' and providing a justification for that label. "
    "Each claim will include a 'STATEMENT ORIGINATOR' indicating the source of the claim, "
    "along with 'FACT-CHECKING INFORMATION' summarizing relevant PolitiFact fact-checks to assist you.\n\n"
    "Truth Scale:\n"
    "True - The statement is accurate and there's nothing significant missing.\n"
    "Mostly true - The statement is accurate but needs clarification or additional information.\n"
    "Half true - The statement is partially accurate but leaves out important details or takes things out of context.\n"
    "Mostly false - The statement contains an element of truth but ignores critical facts that would give a different impression.\n"
    "False - The statement is not accurate.\n"
    "Pants on fire - The statement is not accurate and makes a ridiculous claim.\n"
    "Not enough information - There is not enough information to reliably apply one of the other labels.\n\n"
    "Instructions:\n"
    "1. Evaluate the claim using the most relevant 'FACT-CHECKING INFORMATION' provided.\n"
    "2. If the provided 'FACT-CHECKING INFORMATION' is not relevant to the statement, use the 'Not enough information' label.\n"
    "3. Consider nuances and subtle details that could influence the claim's accuracy.\n"
    "4. Choose the most appropriate label from the 'Truth Scale' and explain your reasoning concisely.\n"
)

NON_RAG_PROMPT = (
    "As an AI fact checker, your task is to evaluate the accuracy of a CLAIM by assigning a label from the 'Truth Scale' and providing a justification for that label. "
    "Each claim will include a 'STATEMENT ORIGINATOR' indicating the source of the claim to assist you.\n\n"
    "Truth Scale:\n"
    "True - The statement is accurate and there's nothing significant missing.\n"
    "Mostly true - The statement is accurate but needs clarification or additional information.\n"
    "Half true - The statement is partially accurate but leaves out important details or takes things out of context.\n"
    "Mostly false - The statement contains an element of truth but ignores critical facts that would give a different impression.\n"
    "False - The statement is not accurate.\n"
    "Pants on fire - The statement is not accurate and makes a ridiculous claim.\n"
    "Not enough information - There is not enough information to reliably apply one of the other labels.\n\n"
    "Instructions:\n"
    "1. Evaluate the claim using the most relevant information you have.\n"
    "2. If you do not have enough information, use the 'Not enough information' label.\n"
    "3. Consider nuances and subtle details that could influence the claim's accuracy.\n"
    "4. Choose the most appropriate label from the 'Truth Scale' and explain your reasoning concisely.\n"
)

RAG_W_JSON_INST_PROMPT = RAG_PROMPT
RAG_W_JSON_INST_PROMPT += (
    "5. Please provide your response in valid JSON format in plain text, without enclosing it in backticks or any other formatting markers. "
    "The response should include exactly two keys: 'label' and 'justification'. "
    "Do not add any extra characters before or after the JSON object."
)

NON_RAG_W_JSON_INST_PROMPT = NON_RAG_PROMPT
NON_RAG_W_JSON_INST_PROMPT += (
    "5. Please provide your response in valid JSON format in plain text, without enclosing it in backticks or any other formatting markers. "
    "The response should include exactly two keys: 'label' and 'justification'. "
    "Do not add any extra characters before or after the JSON object."
)

MESSY_TO_CLEAN_PROMPT = (
    "Your task is to extract information from unstructured AI fact checks "
    "and provide it, unaltered, in valid JSON format. This output must have "
    "exactly two keys:\n"
    "1. 'label': the fact-checking label\n"
    "2. 'justification': the rationale or explanation for that label\n\n"
    "Instructions:\n"
    "1. Read and understand the provided text, which contains malformed JSON.\n"
    "2. Extract the relevant information, preserving every piece of content "
    "exactly as it appeared (no additions, removals, or modifications).\n"
    "3. Produce valid JSON that contains only the keys 'label' and 'justification' "
    "as described above.\n\n"
    "Important:\n"
    "- Do not alter the text of the label or the justification in any way.\n"
    "- Do not include any additional keys, text, comments, or explanations.\n"
    "- The final output must be strictly valid JSON and nothing else.\n\n"
)
