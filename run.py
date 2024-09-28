import os
import sys
import multiprocessing

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.cuda_init import init_cuda
init_cuda()

from app import create_app
from app.services.rag_service import init_llm_process

def main():
    app = create_app()
    # Initialize LLM before running the app
    init_llm_process()
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()