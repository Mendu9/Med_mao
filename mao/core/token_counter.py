import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))

def truncate_to_budget(
    chunks: list[str],
    web: list[str],
    history: list[str],
    budget: int,
) -> dict[str, list[str]]:
    chunks = list(chunks)
    web = list(web)
    history = list(history)

    def total() -> int:
        return sum(count_tokens(t) for t in chunks + web + history)

    while total() > budget and chunks:
        chunks.pop()
    while total() > budget and web:
        web.pop()
    while total() > budget and history:
        history.pop()

    return {"chunks": chunks, "web": web, "history": history}
