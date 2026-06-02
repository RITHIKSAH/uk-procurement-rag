from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

mistral_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

mistral_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

def build_mistral_context(results):
    context_blocks = []

    for i, (_, row) in enumerate(results.iterrows(), start=1):
        block = f"""
Contract {i}
Notice ID: {row.get("Notice ID")}
Title: {row.get("Title")}
Buyer: {row.get("Buyer")}
Suppliers: {row.get("Suppliers")}
Summary: {row.get("Summary")}
Relevance Score: {row.get("Relevance Score")}
Hybrid Score: {row.get("Hybrid Score")}
Why it matched: {row.get("Why it matched")}
"""
        context_blocks.append(block.strip())

    return "\n\n".join(context_blocks)

def generate_mistral_answer(query, results):
    context = build_mistral_context(results)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a UK public procurement assistant. "
                "Use ONLY the retrieved contract evidence. "
                "Do not invent contracts, buyers, suppliers, values, dates, or categories. "
                "If a result is only partially relevant, say that clearly."
            )
        },
        {
            "role": "user",
            "content": f"""
User query:
{query}

Retrieved contracts:
{context}

Write the answer in this exact format:

Overall answer:
[Briefly answer the user query based only on the retrieved contracts.]

Most relevant contracts:
1. Notice ID: ...
   Title: ...
   Buyer: ...
   Suppliers: ...
   Why relevant: ...

2. Notice ID: ...
   Title: ...
   Buyer: ...
   Suppliers: ...
   Why relevant: ...

3. Notice ID: ...
   Title: ...
   Buyer: ...
   Suppliers: ...
   Why relevant: ...

Limitations:
[Clearly mention if any results are partial matches or if metadata is missing.]
"""
        }
    ]

    prompt = mistral_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = mistral_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(mistral_model.device)

    with torch.no_grad():
        outputs = mistral_model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=False,
            temperature=0.1,
            pad_token_id=mistral_tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    return mistral_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

def answer_query_with_mistral(query, top_k=5):
    retrieved_results = rag_search_v2(query, top_k=top_k)

    answer = generate_mistral_answer(
        query=query,
        results=retrieved_results
    )

    return answer, retrieved_results

# Test the End-to-End RAG Pipeline

query = "Find framework agreements for IT services in the NHS"

answer, retrieved_results = answer_query_with_mistral(query, top_k=5)

print(answer)
retrieved_results
