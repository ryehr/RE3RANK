"""Prompt layout shared by the generator and the scorer.

Every item is a question with its first-stage candidates ("paragraphs", each with an optional
"title", a "paragraph_text" and a boolean "is_supporting" label). The user turn lists the
candidates as "[k] title: text" in first-stage order, then the question. The system prompt is a
plain assistant prompt: the generator is never told to rerank.
"""

RR_SYSTEM = "You are a helpful assistant."


def build_prompt(question, paragraphs, user_open="<|im_start|>user\n"):
    """User turn as one string (starting with the chat user opener) and the character span of
    every candidate's text inside it. user_open is the backbone's user-turn opener
    (backbone.Fmt.user_open)."""
    chunk_part = user_open + "Here are some retrieved chunks:\n\n"
    chunk_ranges = []
    for i, p in enumerate(paragraphs):
        title = p.get("title", "")
        text = (title + ": " + p["paragraph_text"]) if title else p["paragraph_text"]
        chunk_part += f"[{i+1}]"
        start = len(chunk_part)
        chunk_part += " " + text.strip()
        end = len(chunk_part)
        chunk_ranges.append([start, end])
        chunk_part += "\n\n"
    query_part = f"Use the retrieved chunks to answer the user's query.\n\nQuery: {question}"
    return chunk_part + query_part, chunk_ranges


def user_msg(question, paragraphs):
    """The same user turn as a chat message body (for tokenizer.apply_chat_template); byte-identical
    to build_prompt's text after the user opener."""
    parts = ["Here are some retrieved chunks:\n"]
    for i, p in enumerate(paragraphs):
        title = p.get("title", "")
        text = (title + ": " + p["paragraph_text"]) if title else p["paragraph_text"]
        parts.append(f"[{i+1}] {text.strip()}\n")
    parts.append("Use the retrieved chunks to answer the user's query.\n")
    parts.append(f"Query: {question}")
    return "\n".join(parts)
