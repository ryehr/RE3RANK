"""ReasonRank (liuwenhan/reasonrank-7B / -32B) listwise reranking, ported from the official repository.

Source: https://github.com/8421BCD/reasonrank (MIT), commit 15d8733. Everything between the VERBATIM markers is
copied unchanged from utils.py, data.py, rerank/rankllm.py and rerank/rank_listwise_os_llm.py (methods kept as
methods of a small class that carries the attributes they read) and remains under the MIT License of that
repository (copyright the ReasonRank authors); the two prompt strings are the values of listwise_prompt_r1.toml.
Settings of run_rank_llm.sh for reasonrank-7B / -32B:
  * prompt_mode rank_GPT_reasoning: system_prompt_reasoning + ONE user message (prefix, "[i] <passage>" lines,
    post prompt), the model's chat template, ftfy.fix_text over the whole prompt
  * passage: 'Title: <title> Content: <text>' (title only when non-empty), fix_text, cut to max_passage_length
    TOKENS of the model's tokenizer, "[n]" -> "(n)"; the official DOC_MAXLEN is per dataset (100 for TREC/BEIR,
    512 for BRIGHT, R2MED and the math training data) and has no entry for our sets -- 512 here (--doc-maxlen)
  * sliding window 20 / step 10 over rank_end = retrieval_num (= our 50 candidates), the loop of
    sliding_windows_batched: (30, 50), (20, 40), (10, 30), (0, 20), each on the order the previous one left
  * vLLM, temperature 0, max_tokens = reasoning_maxlen 3072 + 100
  * parser: <think>..</think><answer>(..)</answer> on the lower-cased reply, else the text after the last
    <answer>, else the whole reply; digits -> 1-based ids, duplicates dropped, missing ids appended in order
The final order becomes scores: n - position (higher = more relevant).
"""
import copy
import re
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from ftfy import fix_text

REASONING_MAXLEN = 3072
SAMPLING = dict(temperature=0, max_tokens=REASONING_MAXLEN + 100)
WINDOW, STEP = 20, 10
DOC_MAXLEN = 512

# ---------------------------------------------------------------- VERBATIM (rankllm.py, data.py)
class PromptMode(Enum):
    UNSPECIFIED = "unspecified"
    RANK_GPT = "rank_GPT"
    RANK_GPT_reasoning = "rank_GPT_reasoning"
    RANK_GPT_rankk = "rank_GPT_rankk"
    RANK_GPT_qwen3 = "rank_GPT_qwen3"
    RANK_GPT_APEER = "rank_GPT_APEER"
    LRL = "LRL"

    def __str__(self):
        return self.value


@dataclass
class Query:
    qid: str
    text: str


@dataclass
class Candidate:
    docid: str
    score: float
    doc: Dict[str, Any]
    # raw: str


@dataclass
class RankingExecInfo:
    prompt: Any
    response: str
    input_token_count: int
    output_token_count: int


@dataclass
class Result:
    query: Query
    candidates: list[Candidate] = field(default_factory=list)
    ranking_exec_summary: list[RankingExecInfo] = (field(default_factory=list),)


# ---------------------------------------------------------------- VERBATIM (utils.py)
def add_prefix_prompt(promptmode: PromptMode, query: str, num: int) -> str:
    if promptmode in [str(PromptMode.RANK_GPT), str(PromptMode.RANK_GPT_qwen3), str(PromptMode.RANK_GPT_reasoning)]:
        return f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"
    elif promptmode == str(PromptMode.RANK_GPT_rankk):
        return f"Determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try analyze the intent of the query and assess how well each passage satisfy the intent. The query may have typos and passages may contain contradicting information. However, we do not get into fact-checking. We just rank the passages based on they relevancy to the query. Sort them from the most relevant to the least. Answer with the passage number using a format of `[3] > [2] > [4] = [1] > [5]`. Ties are acceptable if they are equally relevant. I need you to be accurate but overthinking it is unnecessary. Output only the ordering without any other text.\n Query: {query}\n"
    elif promptmode == str(PromptMode.RANK_GPT_align):
        return f"I will provide you with {num} passages, each indicated by a numerical identifier [] and a search query: {query}. Generate an appropriate passage order preferred by a large language model-based listwise reranker, so that the large language model-based listwise reranker could generate a good passage ranking based on your generated passage order.\n"
    else: 
        raise ValueError('not supported promptmode')

def add_post_prompt(promptmode: PromptMode, query: str, num: int,) -> str:
    if promptmode in [str(PromptMode.RANK_GPT_reasoning)]:
        example_ordering = "[2] > [1]"
        return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The format of the answer should be [] > [], e.g., {example_ordering}."
    elif promptmode in [str(PromptMode.RANK_GPT)]:
        example_ordering = "[2] > [1]"
        return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}. Only respond with the ranking results, do not say any word or explain."
    elif promptmode in [str(PromptMode.RANK_GPT_qwen3)]:
        example_ordering = "[2] > [1]"
        return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The final ranked list should be enclosed within <answer> </answer> tags, i.e., <answer> ranked list here </answer>. The format of ranked list should be [] > [], e.g., {example_ordering}. Only respond with the ranking results, do not say any word or explain."
    elif promptmode in [str(PromptMode.RANK_GPT_rankk)]:
        return ''
    elif promptmode == str(PromptMode.RANK_GPT_align):
        example_ordering = "[2] [1]"
        return f"Search Query: {query}.\nBy considering the relevance between query and each passage as well as the relationship between passages, generate an appropriate passage order preferred by a large language model listwise reranker. All the passages should be included and listed using identifiers. The output format should be [] [], e.g., {example_ordering}, Only respond with the results of passage order, do not say any word or explain."
    else: 
        raise ValueError('not supported promptmode')

def convert_doc_to_prompt_content(tokenizer, doc: Dict[str, Any], max_length: int, truncate_by_word = True) -> str:
    if "text" in doc:
        content = doc["text"]
    elif "segment" in doc:
        content = doc["segment"]
    elif "contents" in doc:
        content = doc["contents"]
    elif "body" in doc:
        content = doc["body"]
    else:
        content = doc["passage"]
    if "title" in doc and doc["title"]:
        content = "Title: " + doc["title"] + " " + "Content: " + content
    content = content.strip()
    content = fix_text(content)
    if truncate_by_word:
        content = " ".join(content.split()[: int(max_length)])
    else: 
        content = tokenizer.convert_tokens_to_string(tokenizer.tokenize(content)[:max_length])
    # For Japanese should cut by character: content = content[:int(max_length)]
    return replace_number(content)

def replace_number(s: str) -> str:
    return re.sub(r"\[(\d+)\]", r"(\1)", s)
# ---------------------------------------------------------------- end VERBATIM

# listwise_prompt_r1.toml
PROMPT_INFO = {"system_prompt_reasoning": "You are RankLLM, an intelligent assistant that can rank passages based on their relevance to the query. Given a query and a passage list, you first thinks about the reasoning process in the mind and then provides the answer (i.e., the reranked passage list). The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.",
               "pattern": r'<think>.*?</think>\s*<answer>(.*?)</answer>'}


class ReasonRankPrompter:
    """the attributes the official methods read (RankListwiseOSLLM / RankLLM with prompt_mode rank_GPT_reasoning)"""

    def __init__(self, tokenizer, max_passage_length=DOC_MAXLEN):
        self._tokenizer = tokenizer
        self.max_passage_length = max_passage_length
        self.prompt_mode = str(PromptMode.RANK_GPT_reasoning)
        self.args = SimpleNamespace(prompt_mode=str(PromptMode.RANK_GPT_reasoning))
        self.prompt_info = PROMPT_INFO

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    # ------------------------------------------------------------ VERBATIM (rank_listwise_os_llm.py, rankllm.py)
    def create_prompt(self, result: Result, rank_start: int, rank_end: int) -> Tuple[str, int]:
        query = result.query.text
        qid = result.query.qid
        query = self._replace_number(query).strip()
        num = len(result.candidates[rank_start:rank_end])
        max_length = self.max_passage_length

        #################### core codes for constructing the input ####################
        messages = []
        if self.args.prompt_mode == str(PromptMode.RANK_GPT_reasoning):  # for non-reasoning model such as qwen2.5
            messages.append({"role": "system", "content": self.prompt_info['system_prompt_reasoning']})
        elif self.args.prompt_mode in [str(PromptMode.RANK_GPT), str(PromptMode.RANK_GPT_qwen3)]:
            messages.append({"role": "system", "content": self.prompt_info['system_prompt']})

        prefix = add_prefix_prompt(promptmode=self.prompt_mode, query=query, num=num)
        rank = 0
        input_context = f"{prefix}\n"
        for cand in result.candidates[rank_start:rank_end]:
            rank += 1
            content = convert_doc_to_prompt_content(self._tokenizer, cand.doc, max_length, truncate_by_word=False)
            input_context += f"[{rank}] {content}\n"

        input_context += add_post_prompt(promptmode=self.prompt_mode, query=query, num=num)
        messages.append({"role": "user", "content": input_context})
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = fix_text(prompt)
        #################### core codes for constructing the input ####################

        num_tokens = self.get_num_tokens(prompt)
        return prompt, num_tokens

    def _clean_response(self, response: str) -> str:
        if self.prompt_mode in [str(PromptMode.RANK_GPT_qwen3), str(PromptMode.RANK_GPT_reasoning)]: # split the response only when the model is not black-box model
            pattern = rf'{self.prompt_info["pattern"]}'
            match = re.search(pattern, response.lower(), re.DOTALL)
            if match:
                response = match.group(1).strip()
            else:
                if '<answer>' in response:
                    response = response.split('<answer>')[-1]
                else:
                    print('the reasoning length is too long.')
        new_response = ""
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def _remove_duplicate(self, response: List[int]) -> List[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def receive_permutation(
        self, result: Result, permutation: str, rank_start: int, rank_end: int
    ) -> Result:
        """
        Processes and applies a permutation to the ranking results.

        This function takes a permutation string, representing the new order of items,
        and applies it to a subset of the ranking results. It adjusts the ranks and scores in the
        'result' object based on this permutation.

        Args:
            result (Result): The result object containing the initial ranking results.
            permutation (str): A string representing the new order of items.
                            Each item in the string should correspond to a rank in the results.
            rank_start (int): The starting index of the range in the results to which the permutation is applied.
            rank_end (int): The ending index of the range in the results to which the permutation is applied.

        Returns:
            Result: The updated result object with the new ranking order applied.

        Note:
            This function assumes that the permutation string is a sequence of integers separated by spaces.
            Each integer in the permutation string corresponds to a 1-based index in the ranking results.
            The function first normalizes these to 0-based indices, removes duplicates, and then reorders
            the items in the specified range of the 'result.candidates' list according to the permutation.
            Items not mentioned in the permutation string remain in their original sequence but are moved after
            the permuted items.
        """
        response = self._clean_response(permutation)
        try:
            response = [int(x) - 1 for x in response.split()]
        except ValueError:
            print('unexcepted output!')
            return result

        response = self._remove_duplicate(response)
        cut_range = copy.deepcopy(result.candidates[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        for j, x in enumerate(response): # lwh
            result.candidates[rank_start + j] = copy.deepcopy(cut_range[x])

            result.candidates[rank_start + j].score = cut_range[j].score
        return result

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)
    # ------------------------------------------------------------ end VERBATIM


def window_positions(n, rank_start=0, window_size=WINDOW, step=STEP):
    """the (start, end) windows of RankLLM.sliding_windows_batched with rank_end = n, in the order they run"""
    rank_end, window_size = n, min(window_size, n)
    out = []
    windows_end = rank_end
    windows_start = rank_end - window_size
    while windows_end > rank_start and windows_start + step != rank_start:
        windows_start = max(windows_start, rank_start)
        out.append((windows_start, windows_end))
        windows_end = windows_end - step
        windows_start = windows_start - step
    return out


def to_result(it):
    """our candidate item -> the official Result (candidates in first-stage order; docid = candidate index)"""
    n = len(it["paragraphs"])
    return Result(query=Query(qid=str(it["id"]), text=it["question"]),
                  candidates=[Candidate(docid=str(j), score=float(n - j),
                                        doc={"text": p["paragraph_text"], "title": p.get("title", "")})
                              for j, p in enumerate(it["paragraphs"])], ranking_exec_summary=[])


def final_scores(result):
    n = len(result.candidates)
    s = [0.0] * n
    for pos, c in enumerate(result.candidates):
        s[int(c.docid)] = float(n - pos)
    return s


def answer_parsed(reply):
    """diagnostic: the official pattern matched (a closed think block followed by a closed answer)"""
    return re.search(PROMPT_INFO["pattern"], reply.lower(), re.DOTALL) is not None
