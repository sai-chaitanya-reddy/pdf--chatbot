from groq import Groq
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_prompt(question: str, chunks: List[Dict], chat_history: List[Dict]):
    # Build context block from retrieved chunks
    context_block = ""
    for i, chunk in enumerate(chunks, 1):
        context_block += (
            f"[Source {i}: {chunk['filename']}, Page {chunk['page_number']}]\n"
            f"{chunk['text']}\n\n"
        )

    # Build chat history messages (last 6 messages)
    history_messages = []
    if chat_history:
        for msg in chat_history[-6:]:
            history_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    system_prompt = f"""You are a helpful document assistant. Answer questions strictly based on the provided document context below.

DOCUMENT CONTEXT:
{context_block}

Instructions:
- Answer based ONLY on the document context provided above.
- If the answer is not in the context, say "I couldn't find this information in the uploaded documents."
- At the end of your answer, always list which sources (filename and page number) you used.
- Be concise and accurate.
- If multiple sources support the answer, mention all of them."""

    return system_prompt, history_messages


def ask_gemini(question: str, chunks: List[Dict], chat_history: List[Dict]) -> Dict:
    if not chunks:
        return {
            "answer": "No relevant content found in the uploaded documents for your question. Please try rephrasing or upload relevant documents.",
            "sources": []
        }

    system_prompt, history_messages = build_prompt(question, chunks, chat_history)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
        )
        answer = response.choices[0].message.content

        # Extract unique sources from retrieved chunks
        sources = []
        seen = set()
        for chunk in chunks:
            key = (chunk["filename"], chunk["page_number"])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "excerpt": chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
                    "relevance_score": chunk["relevance_score"]
                })

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": []
        }
