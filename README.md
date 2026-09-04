# For Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
irm https://ollama.com/install.ps1 | iex                            
ollama pull llama3.1
ollama pull nomic-embed-text
streamlit run app.py