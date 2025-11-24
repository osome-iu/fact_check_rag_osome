"""
Helper functions for parsing data from various sources.
"""

import json

from llm_fact.utils import extract_model_name, extract_num_rag_items, used_web_search

OPENAI_MODELS = [
    "gpt-4o-search-preview-2025-03-11",
    "gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    "o3-mini-2025-01-31",
    "o1-2024-12-17",
]

TOGETHER_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
]

GOOGLE_MODELS = [
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-8b-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro-002",
]

REASONING_MODELS = [
    "deepseek-ai/DeepSeek-R1",
    "o3-mini-2025-01-31",
    "o1-2024-12-17",
    "gemini-2.0-flash-thinking-exp-01-21",
]

# Some of the older Google models return finish reasons as numbers instead of the
# more standard string values. This map helps to map them.
# Ref: https://github.com/googleapis/python-aiplatform/blob/b91edf52e2b993c3301a419ad89b473c31c60cc3/google/cloud/aiplatform_v1/types/content.py#L508
GOOGLE_FINISH_REASON_MAP = {
    0: "unspecified",
    1: "stop",
    2: "max_tokens",
    3: "safety",
    4: "recitation",
    5: "other",
}


def parse_summary(response):
    """
    Parse OpenAI summary responses from JSONL format.

    Args:
        response (dict): A dictionary containing an OpenAI API response for a summary task.

    Returns:
        dict: A flattened dictionary with the following keys:
            - summary (str): The content of the message.
            - created (int): The creation timestamp.
            - model (str): The model used.
            - completion_tokens (int): The number of completion tokens used.
            - prompt_tokens (int): The number of prompt tokens used.
            - total_tokens (int): The total number of tokens used.
            - factcheck_analysis_link (str or None): The link to the fact-check analysis.
            - finish_reason (str): The reason for finishing the response.
    """
    # Extract the summary content from the message
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("No choices found in response")

    message = choices[0].get("message", {})
    summary = message.get("content", "")
    finish_reason = choices[0].get("finish_reason", None)

    # Extract token usage information
    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens", None)
    prompt_tokens = usage.get("prompt_tokens", None)
    total_tokens = usage.get("total_tokens", None)

    # Extract other metadata
    created = response.get("created", None)
    model = response.get("model", None)
    factcheck_analysis_link = response.get("factcheck_analysis_link", None)

    return {
        "summary": summary,
        "created": created,
        "model": model,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "factcheck_analysis_link": factcheck_analysis_link,
        "finish_reason": finish_reason,
    }


class OpenAiFcParser:
    """
    A class to parse OpenAI fact-checking responses.
    """

    def __init__(self, input, input_file_name):
        self._input = input
        self._input_file_name = input_file_name

        # Extract some test info from the filename
        self._used_web_search = used_web_search(input_file_name)
        self._num_rag_items = extract_num_rag_items(self._input_file_name)

        # Assume non-batch responses by default
        self._is_batch = False
        response_obj = self._input.get("model_response", None)

        # But in this case it is a batch response and we need to parse it differently
        if response_obj is None:
            response_obj = self._input.get("response", {}).get("body")
            if response_obj is None:
                raise ValueError("`response` or `body` missing. Cannot parse response.")
            self._is_batch = True
        self._response_obj = response_obj
        self._created = self._response_obj.get("created", None)
        self._model_name = self._response_obj.get("model", None)

    def _get_pf_link(self):
        """Parse the fact-check analysis link from the response."""
        if self._is_batch:
            # `custom_id` looks like: 'url__https://example.com'
            pf_link = self._input.get("custom_id", None)
            if pf_link is None:
                raise ValueError("factcheck_analysis_link is None.")
            return pf_link.replace("url__", "")

        # Otherwise, the link is in the input
        pf_link = self._input.get("factcheck_analysis_link", None)
        if pf_link is None:
            raise ValueError("factcheck_analysis_link is None.")
        return pf_link

    def _get_num_rag_items(self):
        """Parse the rag items."""
        if not self._is_batch:
            # For non-batch responses, we can be extremely cautious and check
            # that info extracted earlier from the file name matches the data object.
            num_rag_items = self._input.get("rag_results", None)
            if num_rag_items is not None:
                num_rag_items = len(num_rag_items)

            assert (
                num_rag_items == self._num_rag_items
            ), "RAG file info != match data object."
            return num_rag_items

        # For batch responses, we can only return the values we have.
        return self._num_rag_items

    def _get_finish_reason(self):
        """Parse the reason the chat completions job finished."""
        finish_reason = self._response_obj.get("choices", [{}])[0].get(
            "finish_reason", None
        )
        if finish_reason is None:
            raise ValueError("`finish_reason` missing from the response object.")
        return finish_reason

    def _get_label_and_justification(self):
        """Extract label and justification from the response."""
        msg = self._response_obj.get("choices", [{}])[0].get("message") or {}
        content = msg.get("parsed") or msg.get("content")
        if content is None:
            raise ValueError("`content` missing from the response object.")
        parsed = content if isinstance(content, dict) else json.loads(content)
        try:
            label = parsed["label"]
            justification = parsed["justification"]
        except KeyError as e:
            raise ValueError(
                f"`{e.args[0]}` missing from the response object."
            ) from None
        return label, justification

    def _get_token_info(self):
        """Extract token info."""
        usage = self._response_obj.get("usage")
        if usage is None:
            raise ValueError("`usage` missing from the response object.")

        token_info = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": usage.get("completion_tokens_details", {}).get(
                "reasoning_tokens", 0
            ),
        }

        missing = [key for key, value in token_info.items() if value is None]
        if missing:
            raise ValueError(f"`usage` missing token counts: {', '.join(missing)}.")
        return token_info

    def parse_fc(self):
        """
        Parse the OpenAI response.
        """
        label, justification = self._get_label_and_justification()
        token_info = self._get_token_info()

        return {
            "factcheck_analysis_link": self._get_pf_link(),
            "model": self._model_name,
            "label": label,
            "justification": justification,
            "created": self._created,
            "finish_reason": self._get_finish_reason(),
            "num_rag_items": self._get_num_rag_items(),
            "used_web_search": self._used_web_search,
            **token_info,
            "filename": self._input_file_name,
        }

    def parse_web_sources(self):
        """
        Return a list of web sources used in the OpenAI response.
        """
        msg = self._response_obj.get("choices", [{}])[0].get("message") or {}
        annotations = msg.get("annotations", [])

        if not annotations:
            # If no annotations, return None in the dictionary for the source infor
            return [
                {
                    "factcheck_analysis_link": self._get_pf_link(),
                    "model": self._model_name,
                    "web_url": None,
                    "url_title": None,
                    "num_rag_items": self._get_num_rag_items(),
                    "used_web_search": self._used_web_search,
                    "filename": self._input_file_name,
                }
            ]

        parsed_sources = []
        for annotation in annotations:
            citation_dict = annotation.get("url_citation", {})
            parsed_sources.append(
                {
                    "factcheck_analysis_link": self._get_pf_link(),
                    "model": self._model_name,
                    "web_url": citation_dict.get("url", ""),
                    "url_title": citation_dict.get("title", ""),
                    "num_rag_items": self._get_num_rag_items(),
                    "used_web_search": self._used_web_search,
                    "filename": self._input_file_name,
                }
            )

        return parsed_sources


class TogetherFcParser:
    """
    A class to parse Together fact-checking responses.
    """

    def __init__(self, input, input_file_name):
        self._input = input
        self._input_file_name = input_file_name
        self._pf_link = self._input.get("factcheck_analysis_link", None)
        if self._pf_link is None:
            raise ValueError("factcheck_analysis_link is None.")

        # Extract some test info from the filename
        self._used_web_search = used_web_search(input_file_name)
        self._num_rag_items = extract_num_rag_items(self._input_file_name)

        response_obj = self._input.get("model_response", None)
        if response_obj is None:
            raise ValueError("Response object is None. Cannot parse response.")
        self._response_obj = response_obj

        self._created = self._response_obj.get("created", None)

    def _get_model_name(self):
        """Parse the model name from the response."""
        model_name = self._response_obj.get("model", None)

        if model_name is None:
            model_name = extract_model_name(self._input_file_name)
        return model_name

    def _get_num_rag_items(self):
        """Parse the rag items."""
        num_rag_items = self._input.get("rag_results", None)
        if num_rag_items is not None:
            num_rag_items = len(num_rag_items)

        assert (
            num_rag_items == self._num_rag_items
        ), "RAG file info != match data object."
        return num_rag_items

    def _get_finish_reason(self):
        """Parse the reason the chat completions job finished."""
        finish_reason = self._response_obj.get("choices", [{}])[0].get(
            "finish_reason", None
        )
        if finish_reason is None:
            raise ValueError("`finish_reason` missing from the response object.")
        return finish_reason

    def _parse_deepseek_content(self, content):
        content = content.split("</think>\n\n")[-1]
        return json.loads(content)

    def _parse_llama_content(self, content):
        try:
            return json.loads(content)
        # In certain cases, there are missing closing brackets in the JSON response.
        except json.JSONDecodeError:
            return json.loads(content + "}")

    def _get_label_and_justification(self):
        """Extract label and justification from the response."""
        msg = self._response_obj.get("choices", [{}])[0].get("message") or {}
        content = msg.get("content")
        if content is None:
            raise ValueError("`content` missing from the response object.")

        model_name = self._get_model_name()
        model_name_lower = model_name.lower()
        if "deepseek" in model_name_lower:
            data_dict = self._parse_deepseek_content(content)
        elif "llama" in model_name_lower:
            data_dict = self._parse_llama_content(content)
        else:
            data_dict = json.loads(content)

        try:
            label = data_dict["label"]
            justification = data_dict["justification"]
        except KeyError as e:
            raise ValueError(
                f"`{e.args[0]}` missing from the response object."
            ) from None
        return label, justification

    def _get_usage_tokens(self):
        """Extract token info."""
        token_info = self._response_obj.get("usage")
        if token_info is None:
            raise ValueError("`usage` missing from the response object.")
        token_info["reasoning_tokens"] = 0  # Never present in Together responses
        return token_info

    def parse_fc(self):
        """
        Parse the Together response.
        """
        label, justification = self._get_label_and_justification()
        token_info = self._get_usage_tokens()

        return {
            "factcheck_analysis_link": self._pf_link,
            "model": self._get_model_name(),
            "label": label,
            "justification": justification,
            "created": self._created,
            "finish_reason": self._get_finish_reason(),
            "num_rag_items": self._get_num_rag_items(),
            "used_web_search": self._used_web_search,
            **token_info,
            "filename": self._input_file_name,
        }


class GoogleFcParser:
    """
    A class to parse Google fact-checking responses.
    """

    def __init__(self, input, input_file_name):
        self._input = input
        self._input_file_name = input_file_name
        self._pf_link = self._input.get("factcheck_analysis_link", None)
        if self._pf_link is None:
            raise ValueError("factcheck_analysis_link is None.")

        # Extract some test info from the filename
        self._used_web_search = used_web_search(input_file_name)
        self._num_rag_items = extract_num_rag_items(self._input_file_name)

        response_obj = self._input.get("model_response", None)
        if response_obj is None:
            raise ValueError("Response object is None. Cannot parse response.")
        self._response_obj = response_obj

    def _get_model_name(self):
        """Parse the model name from the response."""
        model_name = self._response_obj.get("model_version", None)

        # Google is sloppy and only sometimes includes the model name in the response.
        if model_name is None:
            model_name = extract_model_name(self._input_file_name)
        return model_name

    def _get_creation_timestamp(self):
        """Parse the creation time from the response. This is typically not present."""
        return self._response_obj.get("created_time", None)

    def _get_num_rag_items(self):
        """Parse the rag items."""
        num_rag_items = self._input.get("rag_results", None)
        if num_rag_items is not None:
            num_rag_items = len(num_rag_items)

        assert (
            num_rag_items == self._num_rag_items
        ), "RAG file info != match data object."
        return num_rag_items

    def _get_finish_reason(self):
        """Parse the reason the chat completions job finished."""
        finish_reason = self._response_obj.get("candidates", [{}])[0].get(
            "finish_reason", None
        )
        if finish_reason is None:
            raise ValueError("`finish_reason` missing from the response object.")
        if isinstance(finish_reason, int):
            # Map the finish reason to a string.
            try:
                finish_reason = GOOGLE_FINISH_REASON_MAP[finish_reason]
            except KeyError:
                finish_reason = "unknown"
        return finish_reason.lower()

    def _parse_text(self, content):
        """Parse the messy content from the response."""

        try:
            return json.loads(content)

        except json.JSONDecodeError:
            if content.startswith("[\n"):
                return eval(content)[0]

            if content.startswith("```json"):
                return json.loads(content.replace("```json\n", "").replace("\n```", ""))

            return json.loads(content)

    def _get_label_and_justification(self):
        """Extract label and justification from the response."""
        # In this case, Google has pre-parsed our structured response.
        parsed = self._response_obj.get("parsed")
        if parsed is not None and len(parsed) > 0:
            return parsed[0]["label"], parsed[0]["justification"]

        # When this happens we do not have a full response.
        # This should never happen, as we do not set the max_token parameter,
        # but Google is janky and it happens anyway...
        if self._get_finish_reason() == "max_tokens":
            return None, None

        # If the pre-parse doesn't happen, we do it ourselves.
        content = self._response_obj.get("candidates", [{}])[0].get("content", {})
        # Should never happen but Google is janky and it happens anyway...
        if content is None:
            return None, None

        parts = content.get("parts", [{}])
        # Should never happen but Google is janky and it happens anyway...
        if len(parts) == 0:
            return None, None

        text = parts[0].get("text", None)
        # Should never happen but Google is janky and it happens anyway...
        if text is None:
            return None, None

        data = self._parse_text(text)
        if isinstance(data, list):
            if len(data) == 1:
                data_dict = data[0]
            else:
                data_dict = {}
                for x in data:
                    data_dict.update(x)
        else:
            data_dict = data

        label = data_dict.get("label", None)
        justification = data_dict.get("justification", None)
        return label, justification

    def _get_usage_tokens(self):
        """Extract token info."""
        token_info = self._response_obj.get("usage_metadata")
        if token_info is None:
            raise ValueError("`usage` missing from the response object.")

        return {
            "prompt_tokens": token_info.get("prompt_token_count"),
            "completion_tokens": token_info.get("candidates_token_count"),
            "total_tokens": token_info.get("total_token_count"),
            "reasoning_tokens": 0,  # Never present in Google responses
        }

    def parse_fc(self):
        """
        Parse the Google response.
        """
        label, justification = self._get_label_and_justification()
        token_info = self._get_usage_tokens()

        return {
            "factcheck_analysis_link": self._pf_link,
            "model": self._get_model_name(),
            "label": label,
            "justification": justification,
            "created": self._get_creation_timestamp(),
            "finish_reason": self._get_finish_reason(),
            "num_rag_items": self._get_num_rag_items(),
            "used_web_search": self._used_web_search,
            **token_info,
            "filename": self._input_file_name,
        }

    def parse_web_sources(self):
        """
        Return a list of web sources used in the OpenAI response.
        """
        gmetadata = (
            self._response_obj.get("candidates", [{}])[0].get("grounding_metadata")
            or {}
        )
        gchunks = gmetadata.get("grounding_chunks", None)

        if gchunks is None or len(gchunks) == 0:
            # If no annotations, return None in the dictionary for the citation info
            return [
                {
                    "factcheck_analysis_link": self._pf_link,
                    "model": self._get_model_name(),
                    "web_url": None,
                    "url_title": None,
                    "num_rag_items": self._get_num_rag_items(),
                    "used_web_search": self._used_web_search,
                    "filename": self._input_file_name,
                }
            ]

        parsed_sources = []
        for chunk in gchunks:
            web_info_dict = chunk.get("web", {})
            parsed_sources.append(
                {
                    "factcheck_analysis_link": self._pf_link,
                    "model": self._get_model_name(),
                    "web_url": web_info_dict.get("uri", ""),
                    "url_title": web_info_dict.get("title", ""),
                    "num_rag_items": self._get_num_rag_items(),
                    "used_web_search": self._used_web_search,
                    "filename": self._input_file_name,
                }
            )

        return parsed_sources
