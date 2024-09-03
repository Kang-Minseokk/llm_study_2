# 이전에는 모델을 파인튜닝 하는 실습을 진행함. 이 섹션에서는 RAG를 실습해보려고 한다. 말뭉치들을 벡터DB로 변환하는 동작부터 말뭉치들을 잘 참고하여
# 답변을 생성하는지까지 확인을 해주어야 한다.
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
import torch
from time import time
from langchain_huggingface import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

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


# 시스템 프롬포트를 활용해서 파이프라인 테스트하기
system_prompt="1. 이모티콘을 사용하지 말것, 2.reference 주소를 삭제할 것, 3. 공적인 언어를 사용할 것"
user_query="State of the Union이 무엇인지 설명해주세요. 간단히 정의하세요. 100단어 이내로 작성하세요."
full_query=f"""
<start_of_turn>user
{system_prompt}
{user_query}<end_of_turn>
<start_of_turn>model"""

test_model(tokenizer, query_pipeline, full_query)



# RAG 구축
llm = HuggingFacePipeline(pipeline=query_pipeline)

