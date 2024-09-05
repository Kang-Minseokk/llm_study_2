from pdfminer.high_level import extract_text
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
import torch
from time import time
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma


model_id = "google/gemma-2-2b-it"
login(token="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")
# 모델과 토크나이저 로딩 시간 측정
time_1 = time()

model_config = AutoConfig.from_pretrained(
    model_id,
)
# 모델 객체 생성
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=model_config,
    trust_remote_code=True,
    device_map='auto',
)
# 토크나이저 객체 생성
tokenizer = AutoTokenizer.from_pretrained(model_id)
time_2 = time()

print("Prepare model, tokenizer: ", round(time_2 - time_1, 3), "sec.")
device = model.device
print("사용중인 하드웨어 가속기: ", device)
# 쿼리 파이프라인 객체 생성
time_1 = time()
query_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=256,
    repetition_penalty=1.2,
    return_full_text=False,
    temperature=1.0,
    top_k=10,
    do_sample=True,
)
time_2 = time()
print("Prepare pipeline: ", round(time_2 - time_1, 3), "sec.")


# Rag 구축
system_prompt = "1. 이모티콘을 사용하지 말것, 2. 문장의 끝마다 '입니당!'을 추가할 것, 3. 간단하게 답변을 생성할 것."
llm = HuggingFacePipeline(pipeline=query_pipeline)
user_query = "How to spell the English name of 홍규연 교수님?"
full_query = f"""
<start_of_turn>user
{system_prompt}
{user_query}<end_of_turn>
<start_of_turn>model"""
output = llm.invoke(input=full_query)


# extract text from pdf file
extracted_text = extract_text('source/numerical_sample.pdf')

time_1 = time()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_text(extracted_text)
time_2 = time()
print(f'time inference for chunks: {time_2 - time_1}')

time_1 = time()
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
time_2 = time()
print(f'time inference for embeddings: {time_2 - time_1}')

vectordb = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
vectordb.add_texts(texts=all_splits)

# stored_data = vectordb.get()
# for i, text in enumerate(zip(stored_data["documents"])):
#     print(f"Text {i + 1} : {text}")
#     print("-------")

retriever = vectordb.as_retriever()

# query = "How to spell the English name of 홍규연 교수님?"
# results = retriever.get_relevant_documents(query)
#
# for result in results:
#     print(result.page_content)
#     print("--------")


qa = RetrievalQA.from_chain_type(
    llm=llm, # 사용할 언어모델
    chain_type="stuff", # 모든 관련 문서를 하나의 컨텍스트로 결합.
    retriever=retriever,
    verbose=True,
)

def test_rag(qa, query):
    """
    RAG 시스템을 테스트하는 함수
    :param qa:
    :param query:
    :return:
    """
    print(f"Query: {query}\n")
    time_1 = time()
    result = qa.run(query)
    time_2 = time()
    print("Time inference: ", round(time_2 - time_1, 3), "sec.")
    print(f"Result: {result}")


user_query = "How to spell the English name of 홍규연 교수님?"

full_query = f"""
<start_of_turn>user
{system_prompt}
{user_query}<end_of_turn>
<start_of_turn>model"""

test_rag(qa, user_query)

