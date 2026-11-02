"""RE3Rank: reasoning-trace retrieval-head reranking.

A generator LLM writes a reasoning trace over the question and the first-stage candidates; a scorer
reads, through a few selected attention heads, how much attention the trace pays to each candidate,
and ranks the candidates by it. Generator and scorer are trained jointly, with a straight-through
estimator carrying the ranking loss from the scorer back into the generator's sampled trace.
"""
