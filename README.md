# WhatsApp Chat Analyzer with Streamlit

This Streamlit application allows users to upload a WhatsApp chat in `.txt` format, provides a summarization of the conversation, and analyzes the sentiment of messages using OpenAI's API.

## Features:
- Upload a WhatsApp chat in `.txt` format.
- Summarize the chat history.
- Analyze sentiment for each message.
- Interactive user interface using Streamlit.

## Prerequisites

- Python 3.7 or higher
- Pip (Python's package installer)

## Installation

Follow these steps to get the project up and running locally:

### 1. Clone the repository
```bash
git clone  https://github.com/chanakyasethi/WhatsappChatAnalyzer.git
cd WhatsappChatAnalyzer
```


### 2. Set up a virtual environment (optional but recommended)
`
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
`

### 3. Install required dependencies
`pip install -r requirements.txt`

### 4. Set up OpenAI API Key
To enable the summarization and sentiment analysis features, you need to provide an OpenAI API key.

Create an account on OpenAI and obtain an API key.

You can set the environment variable directly in your terminal:

`export OPENAI_API_KEY="your-api-key-here"`  # For Unix-based OS (Linux/macOS)

### 5. Choose OpenAI Model
The application supports different OpenAI models for summarization and sentiment analysis. 

`export OPENAI_MODEL="your-openai-model"` 

### 6. Running the Application
To run the Streamlit application, use the following command:

`streamlit run app.py`



