from flask import Flask, request, jsonify
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI  # Updated import
import gradio as gr  # Updated import
import os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env

app = Flask(__name__)
CORS(app)

def construct_index(directory_path):
    # set number of output tokens
    num_outputs = 256

    _llm_predictor = LLMPredictor(llm=OpenAI(api_key=os.environ.get('OPENAI_API_KEY'), temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(llm_predictor=_llm_predictor)

    docs = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    
    #Directory in which the indexes will be stored
    index.storage_context.persist(persist_dir="indexes")

    return index

def chatbot(input_text):
    
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="indexes")
    
    #load indexes from directory using storage_context 
    query_engne = load_index_from_storage(storage_context).as_query_engine()
    
    response = query_engne.query(input_text)
    
    #returning the response
    return response.response

# Update the Gradio Interface instantiation
# iface = gr.Interface(
#     fn=chatbot,
#     inputs=gr.components.Textbox(lines=5, label="Enter your question here"),  # Updated component import
#     outputs="text",
#     title="Custom-trained AI Chatbot"
# )

#Constructing indexes based on the documents in traininData folder
#This can be skipped if you have already trained your app and need to re-run it
index = construct_index("trainingData")

#launching the web UI using gradio
# iface.launch(share=True)

# Flask route for the chatbot
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    input_text = request.json.get('text')
    if not input_text:
        return jsonify({'error': 'No text provided'}), 400

    response = chatbot(input_text)
    return jsonify({'response': response})


# Main function to run the app
if __name__ == '__main__':
    # Constructing indexes (optional based on your requirements)
    index = construct_index("trainingData")

    # Running the Flask app
    app.run(debug=True, port=6000)
