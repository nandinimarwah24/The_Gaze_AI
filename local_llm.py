import json
import socket
from urllib import error, request


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "llama3.2:3b"


class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot generate an answer."""


def build_chat_messages(question, results, chat_history=None):
    """Build Ollama chat messages using retrieved context and recent memory."""
    context_sections = []
    for position, result in enumerate(results, 1):
        context_sections.append(
            f"[Source {position} | PDF: {result['source']} | "
            f"Page: {result['page']} | Chunk: {result['index']}]\n"
            f"{result['chunk'].strip()}"
        )

    context = "\n\n".join(context_sections)
    system_message = f"""You are a document question-answering assistant.

Rules:
- Answer using only the retrieved document context below.
- Use conversation history only to understand follow-up references.
- Do not treat conversation history as a factual source.
- If the context is insufficient, say: "I could not find enough information in the documents."
- Keep the answer clear and concise.
- Cite claims using [PDF filename, Page N].

Retrieved document context:
{context}"""

    messages = [{"role": "system", "content": system_message}]
    for message in chat_history or []:
        if message.get("role") in {"user", "assistant"} and message.get("content"):
            messages.append(
                {
                    "role": message["role"],
                    "content": str(message["content"]),
                }
            )
    messages.append({"role": "user", "content": question})
    return messages


def generate_with_ollama(
    messages,
    model=DEFAULT_LLM_MODEL,
    base_url=DEFAULT_OLLAMA_URL,
    timeout=120,
    stream=False,
):
    """Send a chat request to a local Ollama server."""
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": 0.1},
        }
    ).encode("utf-8")

    http_request = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 and "model" in details.lower():
            raise OllamaError(
                f"Model '{model}' is not installed. Run: ollama pull {model}"
            ) from exc
        raise OllamaError(f"Ollama returned HTTP {exc.code}: {details}") from exc
    except (error.URLError, ConnectionError) as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {base_url}. "
            "Start it with: ollama serve"
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise OllamaError(
            f"Ollama did not answer within {timeout} seconds."
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OllamaError("Ollama returned an invalid response.") from exc

    answer = payload.get("message", {}).get("content", "").strip()
    if not answer:
        raise OllamaError(f"Ollama returned no answer: {payload}")
    return answer


def generate_with_ollama_stream(
    messages,
    model=DEFAULT_LLM_MODEL,
    base_url=DEFAULT_OLLAMA_URL,
    timeout=120,
):
    """Send a streaming chat request to a local Ollama server. Yields answer chunks."""
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.1},
        }
    ).encode("utf-8")

    http_request = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
                except json.JSONDecodeError:
                    continue
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 and "model" in details.lower():
            raise OllamaError(
                f"Model '{model}' is not installed. Run: ollama pull {model}"
            ) from exc
        raise OllamaError(f"Ollama returned HTTP {exc.code}: {details}") from exc
    except (error.URLError, ConnectionError) as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {base_url}. "
            "Start it with: ollama serve"
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise OllamaError(
            f"Ollama did not answer within {timeout} seconds."
        ) from exc

def answer_with_context(
    question,
    results,
    chat_history=None,
    model=DEFAULT_LLM_MODEL,
    base_url=DEFAULT_OLLAMA_URL,
    timeout=120,
    stream=False,
):
    """Answer a question using retrieved context. Streams if stream=True."""
    messages = build_chat_messages(question, results, chat_history)
    if stream:
        return generate_with_ollama_stream(messages, model, base_url, timeout)
    return generate_with_ollama(messages, model, base_url, timeout, stream=False)


def extract_memory_with_ollama(
    user_message,
    model=DEFAULT_LLM_MODEL,
    base_url=DEFAULT_OLLAMA_URL,
    timeout=30,
):
    """
    Extract long-term user information as JSON.
    """

    system_prompt = """
You are an AI memory extraction engine.

Your job is to extract ONLY long-term user facts.

Remember ONLY:
- name
- age
- profession
- occupation
- college
- course
- semester
- city
- skills
- interests
- goals
- favourite_language

DO NOT remember:
- greetings
- questions
- PDF content
- temporary requests

Return ONLY valid JSON.

If nothing should be remembered return:
{}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    answer = generate_with_ollama(
        messages,
        model=model,
        base_url=base_url,
        timeout=timeout,
    ).strip()

    # Extract only the JSON block if Ollama adds extra text
    start = answer.find("{")
    end = answer.rfind("}")

    if start == -1 or end == -1:
        return {}

    try:
        return json.loads(answer[start:end + 1])
    except json.JSONDecodeError:
        return {}
