# 이전에는 모델을 파인튜닝 하는 실습을 진행함. 이 섹션에서는 RAG를 실습해보려고 한다. 말뭉치들을 벡터DB로 변환하는 동작부터 말뭉치들을 잘 참고하여
# 답변을 생성하는지까지 확인을 해주어야 한다.
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


# 쿼리 파이프라인 정의 : 이번에는 파이프라인을 활용해서 입력과 출력을 제어해보자.
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


# 입력값 넣고 출력값 확인하기 코드
# input_text = "Hi, please introduce yourself."
# tokenized_input = {
#    "input_ids": tokenizer.encode(input_text, return_tensors="pt").to(device),
#    "max_length": 128,
# }
#
# output = model.generate(**tokenized_input)
# print(tokenizer.decode(output[0]))


# 모델을 테스트해보자.
def test_model(tokenizer, pipeline, prompt_to_test):
    """
    :param tokenizer: the tokenizer
    :param pipeline: the pipeline
    :param prompt_to_test: the prompt
    :return: None
    """
    time_3 = time()
    sequences = pipeline(
        prompt_to_test,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    time_4 = time()
    print("Time inference: ", round(time_4 - time_3, 3), "sec.")
    print(sequences)
    for sequence in sequences:
        print(f"{sequence['generated_text']}")

# 프롬포트를 입력값에 추가하여 함께 전달하기
# test_model(tokenizer,
#            query_pipeline,
#            "State of Union이 무엇인지 설명해주세요. 간단히 정의하세요. 100단어 이내로 작성하세요.")


# # 시스템 프롬포트를 활용해서 파이프라인 테스트하기
# system_prompt="1. 이모티콘을 사용하지 말것, 2.reference 주소를 삭제할 것, 3. 공적인 언어를 사용할 것"
# user_query="State of the Union이 무엇인지 설명해주세요. 간단히 정의하세요. 100단어 이내로 작성하세요."
# full_query=f"""
# <start_of_turn>user
# {system_prompt}
# {user_query}<end_of_turn>
# <start_of_turn>model"""
#
# test_model(tokenizer, query_pipeline, full_query)



# RAG 구축
system_prompt="1. 이모티콘을 사용하지 말것, 2.reference 주소를 삭제할 것, 3. 공적인 언어를 사용할 것"
llm = HuggingFacePipeline(pipeline=query_pipeline)
user_query = "State of Union이 무엇인지 설명해주세요. 간단히 정의하세요. 100단어 이내로 작성하세요."
full_query = f"""
<start_of_turn>user
{system_prompt}
{user_query}<end_of_turn>
<start_of_turn>model"""
output = llm.invoke(input=full_query)
print(output)



# Text Loader을 활용한 데이터 수집
loader = TextLoader('source/biden_sotu_2023.txt', encoding = "utf-8")
documents = loader.load()

# 청킹을 사용해서 텍스트 데이터 분할하기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "mps"}

embeddings = HuggingFaceEmbeddings(model_name=model_name)

vectordb = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
vectordb.add_documents(documents=all_splits)

# 검색기 생성
retriever = vectordb.as_retriever()

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


user_query = "2023년 State of the Union의 주요 주제는 무엇이었나요? 요약하세요. 200단어 이내로 작성하세요"

full_query = f"""
<start_of_turn>user
{system_prompt}
{user_query}<end_of_turn>
<start_of_turn>model"""

test_rag(qa, user_query)


# 문서 출처
docs = vectordb.similarity_search(user_query)
print(f"Query: {user_query}\n")
print(f"Retrieved documents: {len(docs)}")

for doc in docs:
    doc_details = doc.to_json()['kwargs']
    print("Source:", doc_details['metadata']['source'])
    print("Text:", doc_details['page_content'], '\n')
