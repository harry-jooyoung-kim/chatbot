from flask import Flask, render_template, request, jsonify
import openai
import os
from werkzeug.utils import secure_filename

# LangChain 관련 임포트
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

app = Flask(__name__)

# OpenAI API 키를 코드에 직접 입력
openai.api_key = "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"

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 전역 벡터DB 객체 (간단 예시)
vector_db = None
retriever = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global vector_db, retriever
    if 'file' not in request.files:
        return jsonify({'message': '파일이 없습니다.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': '파일명이 없습니다.'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    # 문서 로드 및 벡터화
    loader = TextLoader(filepath, encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    vector_db = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")
    retriever = vector_db.as_retriever()
    return jsonify({'message': '문서 업로드 및 벡터화 완료!'})

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    global retriever
    # 업로드된 문서가 있으면 RAG, 없으면 일반 OpenAI 답변
    if retriever:
        llm = ChatOpenAI(
            openai_api_key="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",
            model="gpt-3.5-turbo"
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        bot_response = qa.run(user_message)
    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 친절한 한국어 챗봇입니다."},
                {"role": "user", "content": user_message}
            ]
        )
        bot_response = response.choices[0].message['content'].strip()
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True) 