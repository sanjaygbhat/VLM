import os
import base64
from byaldi import RAGMultiModalModel
import anthropic
from app import Config
from app.utils.helpers import load_document_indices
from werkzeug.utils import secure_filename

RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

def query_document(doc_id, query, k):
    document_indices = load_document_indices()
    if doc_id not in document_indices:
        return {"error": "Invalid document_id"}
    
    index_path = document_indices[doc_id]
    
    if not os.path.exists(index_path):
        return {"error": "Index file not found"}
    
    RAG_specific = RAGMultiModalModel.from_index(index_path)
    results = RAG_specific.search(query, k=k)
    
    serializable_results = [
        {
            "doc_id": result.doc_id,
            "page_num": result.page_num,
            "score": result.score,
            "metadata": result.metadata,
            "base64": result.base64
        } for result in results
    ]
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here are some relevant document excerpts:\n\n"
                }
            ]
        }
    ]

    for idx, result in enumerate(serializable_results, 1):
        messages[0]["content"].extend([
            {"type": "text", "text": f"Excerpt {idx}:\n"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": result['base64']}},
            {"type": "text", "text": f"Metadata: {result['metadata']}\n\n"}
        ])

    messages[0]["content"].append({"type": "text", "text": f"Based on these excerpts, please answer the following question: {query}"})

    claude_response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=4000,
        temperature=0,
        messages=messages
    )
    
    return {
        "results": serializable_results,
        "answer": claude_response.content[0].text,
        "tokens_consumed": {
            "prompt_tokens": claude_response.usage.input_tokens,
            "completion_tokens": claude_response.usage.output_tokens,
            "total_tokens": claude_response.usage.input_tokens + claude_response.usage.output_tokens
        }
    }

def query_image(image, query):
    filename = secure_filename(image.filename)
    image_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    image.save(image_path)
    
    rag_results = RAG.search(query, image_path=image_path)
    
    serializable_results = [
        {
            "doc_id": result.doc_id,
            "page_num": result.page_num,
            "score": result.score,
            "metadata": result.metadata,
            "base64": result.base64
        } for result in rag_results
    ]
    
    with open(image_path, "rb") as image_file:
        encoded_query_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    os.remove(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here's the query image:"
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": encoded_query_image}
                },
                {
                    "type": "text",
                    "text": "Here are some relevant image results:\n\n"
                }
            ]
        }
    ]

    for idx, result in enumerate(serializable_results, 1):
        messages[0]["content"].extend([
            {"type": "text", "text": f"Image {idx}:\n"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": result['base64']}},
            {"type": "text", "text": f"Metadata: {result['metadata']}\n\n"}
        ])

    messages[0]["content"].append({"type": "text", "text": f"Based on these images, please answer the following question: {query}"})

    claude_response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=4000,
        temperature=0,
        messages=messages
    )
    
    return {
        "results": serializable_results,
        "answer": claude_response.content[0].text,
        "query_image_base64": encoded_query_image,
        "tokens_consumed": {
            "prompt_tokens": claude_response.usage.input_tokens,
            "completion_tokens": claude_response.usage.output_tokens,
            "total_tokens": claude_response.usage.input_tokens + claude_response.usage.output_tokens
        }
    }