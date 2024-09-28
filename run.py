from app import create_app
from app.services.rag_service import llm

app = create_app()

if __name__ == '__main__':
    # Initialize the LLM
    llm.load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)