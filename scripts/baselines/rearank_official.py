"""REARANK (le723z/Rearank-7B) listwise reranking, ported from the official repository.

Source: https://github.com/lezhang7/Rearank (Apache-2.0), commit 4307152, rank_gpt.py / agent.py.
The four functions between the VERBATIM markers are copied unchanged from rank_gpt.py and remain under the
Apache License 2.0 of that repository (copyright the REARANK authors); everything else adapts the RE3Rank
candidate files to the official data layout and drives the official sliding window:
  * passages as the official run_retriever builds them: 'Title: <title> Content: <text>' when the
    document has a title field, whitespace collapsed; create_permutation_instruction_rearank then drops an
    empty 'Title: Content: ' and cuts each passage to its first 400 characters
  * sliding window of 20, step 10, from the bottom of the list to the top (rank_end = min(n, 100)); with
    our 50 candidates that is 4 windows, (30, 50), (20, 40), (10, 30), (0, 20), each one on the order the
    previous window left
  * decoding (agent.QwenClient): the model's chat template, temperature 0, top_p 1, repetition_penalty 1,
    max_tokens 2048
  * a window reply is parsed by clean_response / receive_permutation: the ids inside <answer>, duplicates
    dropped, ids the reply leaves out appended in their current order
The final order becomes scores: n - position (higher = more relevant).
"""
import copy
import re

SAMPLING = dict(temperature=0.0, top_p=1.0, repetition_penalty=1.0, max_tokens=2048)
WINDOW, STEP = 20, 10


def to_item(it):
    """our candidate item -> the official {'query', 'hits': [{'content', ...}]} layout (run_retriever)"""
    hits = []
    for j, p in enumerate(it["paragraphs"]):
        content = ("Title: " + p["title"] + " " + "Content: " + p["paragraph_text"]) if "title" in p \
            else p["paragraph_text"]
        hits.append({"content": " ".join(content.split()), "cand": j})
    return {"query": it["question"], "hits": hits}


def window_positions(n, rank_start=0, window_size=WINDOW, step=STEP):
    """the (start, end) windows of sliding_windows_batch, in the order they are run"""
    end_pos = min(n, 100)
    rank_end = end_pos
    start_pos = rank_end - window_size
    out = []
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        out.append((start_pos, end_pos))
        end_pos = end_pos - step
        start_pos = start_pos - step
    return out


def final_scores(item):
    n = len(item["hits"])
    s = [0.0] * n
    for pos, h in enumerate(item["hits"]):
        s[h["cand"]] = float(n - pos)
    return s


# ---------------------------------------------------------------- VERBATIM from rank_gpt.py @4307152
def create_permutation_instruction_rearank(item=None, rank_start=0, rank_end=100):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])
    instruction =  (
            f"I will provide you with passages, each indicated by number identifier []. Rank the passages based on their relevance to the search query."
            f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query."
            f"The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be <answer> [] > [] </answer>, e.g., <answer> [1] > [2] </answer>."
        )
    
    messages = [
            {"role": "system", "content": "You are DeepRerank, an intelligent assistant that can rank passages based on their relevancy to the search query. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
            {"role": "user","content": instruction},
            {"role": "assistant", "content": "Okay, please provide the passages."}
        ]
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        content = ' '.join(content.split())
        messages.append({"role": "user", "content": f"[{rank}] {content[:400]}"})
        messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})
                
    messages.append({
        "role": "user",
        "content": f"""Please rank these passages according to their relevance to the search query: "{query}"
            Follow these steps exactly:
            1. First, within <think> tags, analyze EACH passage individually:
            - Evaluate how well it addresses the query
            - Note specific relevant information or keywords

            2. Then, within <answer> tags, provide ONLY the final ranking in descending order of relevance using the format: [X] > [Y] > [Z]"""
    })

    return messages


def clean_response(response: str):
    if "<answer>" in response:
        content_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        response = content_match.group(1).strip() if content_match else response.strip()
    # process for Qwen3 with think tags but no answer tags
    if "<think>" in response:
        response = response.split("</think>")[-1]
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100, verbose=False):
    clean_permutation = clean_response(permutation)
    if verbose:
        print("="*100)
        print(f"query: {item['query']}")
        print(f"rank_start: {rank_start}, rank_end: {rank_end}")
        print(f"model response: {permutation}")

    response = []
    for x in clean_permutation.split():
        try:
            response.append(int(x) - 1)
        except:
            pass

    if verbose:
        print(f"cleaned response: {response}")
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item
# ---------------------------------------------------------------- end VERBATIM


def answer_parsed(reply):
    """diagnostic: did the reply carry a closed <answer>...</answer> block"""
    return re.search(r"<answer>(.*?)</answer>", reply, re.DOTALL) is not None
