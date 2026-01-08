import os
import gradio as gr
import torch
import pandas as pd
import re
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import warnings
import openai
import time
import random
import ast
from peft import PeftConfig, get_peft_model
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

warnings.filterwarnings("ignore")
DEBUG = True
print("[DEBUG] HF_ACCESS_TOKEN exists?", bool(os.getenv("HF_ACCESS_TOKEN")))

# ========== 0. OpenAI API 설정 및 번역 함수 ==========
openai.api_key = os.getenv("api_key")
if openai.api_key is None:
    raise ValueError("OpenAI API key (api_key) not found in environment variables")

def translate_to_korean(text):
    if DEBUG:
        print("[디버그] translate_to_korean - 원문(영어):", text)
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a professional Korean translator for mental health chatbot dialogues using CBT-I techniques. "
                    "Translate the following English sentence into natural, emotionally supportive Korean that matches the tone of a CBT-I therapy session. "
                    "The translation should sound compassionate yet professional, as if a therapist is speaking in a supportive, non-judgmental manner. "
                    "If the user's sentence is a question, ensure the translation naturally ends with an appropriate question ending in Korean. "
                    "Do not add extra filler or change the meaning. "
                    "Return ONLY the translated Korean sentence."
                )},
                {"role": "user", "content": text}
            ],
            temperature=0.4,
            max_tokens=512
        )
        translated_text = response.choices[0].message.content.strip()
        if DEBUG:
            print("[디버그] translate_to_korean - 번역 결과(한국어):", translated_text)
        return translated_text
    except Exception as e:
        print(f"🚨 translate_to_korean 오류: {e}")
        return "잠시 번역에 문제가 발생했어요. 다시 시도해 주세요."

def translate_to_english(text):
    if DEBUG:
        print("[디버그] translate_to_english - 원문(한국어):", text)
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates Korean to English."},
                {"role": "user", "content": f"Translate the following Korean sentence to English:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=512
        )
        translated_text = response.choices[0].message.content.strip()
        if DEBUG:
            print("[디버그] translate_to_english - 번역 결과(영어):", translated_text)
        return translated_text
    except Exception as e:
        print(f"🚨 translate_to_english 오류: {e}")
        return "잠시 번역에 문제가 발생했어요. 다시 시도해 주세요."

def normalize_yes_no(user_input):
    normalized = user_input.strip().lower().replace(" ", "")
    if normalized in ["예", "네", "ㅇ", "y", "yes", "예.", "네."]:
        return "예"
    elif normalized in ["아니오", "아니요", "아뇨", "ㄴ", "n", "no", "아니오.", "아니요."]:
        return "아니오"
    else:
        return None

def is_valid_concern(text):
    text = text.strip()
    if len(text) < 2:
        return False
    if not re.search(r'[가-힣]', text):
        return False
    return True

# ========== 1. 환경 설정 및 모델 로딩 ==========
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')
if HF_ACCESS_TOKEN is None:
    raise ValueError("HF_ACCESS_TOKEN not found in environment variables")

gpu_device = os.environ.get("GPU_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
llama_device = int(gpu_device.split(":")[1]) if ":" in gpu_device else 0 if gpu_device.startswith("cuda") else -1

# BERT intent classifier
bert_tokenizer = RobertaTokenizer.from_pretrained("youjin129/cbt_i_roberta", use_auth_token=HF_ACCESS_TOKEN)
bert_model = RobertaForSequenceClassification.from_pretrained("youjin129/cbt_i_roberta", use_auth_token=HF_ACCESS_TOKEN)
bert_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# LLaMA-3 Pipeline with Fine-Tuning (LoRA) - Private 환경 적용
llama_model_id = "meta-llama/Llama-3.1-8B-Instruct"
lora_ckpt_path = "youjin129/cbt_i_llama3.1_instruct"  # Private repository 사용

llama_tokenizer = AutoTokenizer.from_pretrained(
    llama_model_id,
    use_auth_token=HF_ACCESS_TOKEN
)
base_llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_id,
    use_auth_token=HF_ACCESS_TOKEN,
    torch_dtype=torch.float32,
    device_map="auto"
)

peft_config = PeftConfig.from_pretrained(
    lora_ckpt_path,
    use_auth_token=HF_ACCESS_TOKEN
)
lora_llama_model = get_peft_model(base_llama_model, peft_config)
lora_llama_model.eval()

merged_llama_model = lora_llama_model.merge_and_unload()
merged_llama_model.eval()

llama_pipeline = pipeline(
    "text-generation",
    model=merged_llama_model,
    tokenizer=llama_tokenizer,
    device_map="auto"
)
llama_pipeline.model.eval()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ========== 2. 데이터 로딩 ==========
file_path = "./data/RAG_0407_eng.xlsx"
data_df = pd.read_excel(file_path)
df = pd.DataFrame([{
    "approach": row["Approach"],
    "user_input": row["User Utterance (English)"],
    "info": row["Therapist Response (English)"]
} for _, row in data_df.iterrows()])

CBT_I_DESCRIPTIONS = {
    "수면 제한 요법 (Sleep Restriction)": (
        "수면 제한 요법은 침대에 머무는 시간을 의도적으로 줄여, 침대와 수면 사이의 올바른 연결고리를 재구축하는 방법입니다. "
        "예를 들어, 침대에 10시간 머물지만 실제 수면 시간이 5시간인 경우, 처음에는 5시간만 침대에서 자고 점차 시간을 늘려가면서 몸이 침대를 ‘숙면을 위한 장소’로 인식하도록 돕습니다."
    ),
    "자극 조절 요법 (Stimulus Control)": (
        "자극 조절 요법은 침대와 수면의 관계를 재정립하여, 침대를 오직 수면만을 위한 장소로 인식하게 만드는 치료법입니다. "
        "잠이 오지 않을 때는 즉시 침대에서 벗어나고, 침대에서는 오직 수면만 취하는 습관을 기르는 것이 핵심입니다."
    ),
    "수면 위생 교육 (Sleep Hygiene)": (
        "수면 위생 교육은 건강한 수면을 위해 생활 습관을 개선하는 방법입니다. "
        "예를 들어, 낮에는 카페인 섭취를 줄이고, 일정한 시간에 자고 일어나며, 취침 전에는 전자기기 사용과 밝은 조명을 피하고, 지나치게 늦은 시간의 운동을 삼가는 등의 습관을 포함합니다."
    ),
    "이완 요법 (Relaxation Techniques)": (
        "이완 요법은 심리적·신체적 긴장을 완화시켜 자연스러운 수면을 유도하는 방법입니다. "
        "심호흡, 명상, 가벼운 스트레칭, 그리고 근육 이완 운동을 통해 몸과 마음을 편안하게 만드는 것이 주된 목표입니다."
    ),
    "인지적 재구성 (Cognitive Restructuring)": (
        "인지적 재구성은 수면과 관련된 부정적인 생각이나 걱정을 긍정적으로 전환하는 치료법입니다. "
        "예를 들어, '오늘도 잠을 못 자면 큰일 날 거야'라는 비관적인 생각 대신, '조금 부족하더라도 몸은 점차 적응할 수 있어'라는 긍정적인 관점으로 변화시키며, 걱정 관리와 감정 조절, 행동 실험 등의 기법을 통해 불면증으로 인한 불안을 줄이는 데 도움을 줍니다."
    )
}

ENG_TO_KOR_KEY = {
    "Sleep Restriction": "수면 제한 요법 (Sleep Restriction)",
    "Stimulus Control": "자극 조절 요법 (Stimulus Control)",
    "Sleep Hygiene": "수면 위생 교육 (Sleep Hygiene)",
    "Relaxation Techniques": "이완 요법 (Relaxation Techniques)",
    "Cognitive Restructuring": "인지적 재구성 (Cognitive Restructuring)"
}
KOR_TO_ENG_KEY = {v: k for k, v in ENG_TO_KOR_KEY.items()}

# ========== 3. 초기 세션 상태 관리 ==========
def get_initial_state():
    return {
        "user_name": None,
        "history": [],
        "prev_approaches": [],
        "recommended_approach": None,
        "consult_query": None,
        "faiss_index_cache": {},
        "mode": "학습 모드",  # 초기에는 반드시 학습 모드
        "consulting_active": False,
        "socratic_active": False,
        "socratic_depth": 0,
        "max_depth": 5,
        "socratic_hints": [],
        "current_subquestion": None,
        "current_confidence": "low",
        "current_type": None,
        "self_decision_pending": False,
        "technique_selection_pending": False,
        "waiting_end_confirmation": False,
        "awaiting_termination": False,
        "learning_index": 0,
        "iterative_advice_active": False,
        "iterative_context": "",
        "current_iterative_advice": "",
        "type_history": []
    }

def reset_state(state):
    # 사용자 이름은 그대로 유지하고 나머지 상태를 초기화함
    user_name = state.get("user_name")
    new_state = get_initial_state()
    new_state["user_name"] = user_name
    return new_state

# ------------------------------
# 4. 주요 함수들 (state를 인자로 받도록 수정)
# ------------------------------

def extract_type_flexible(text):
    match = re.search(r'Type:\s*(\w+)', text)
    if match:
        t = match.group(1).lower()
        valid_types = ["clarity", "implication_consequences", "reasons_evidence", "assumptions", "alternate_viewpoints_perspectives"]
        if t in valid_types:
            return t
    return None

def classify_intent_with_bert(user_input):
    inputs = bert_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(bert_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    predicted_idx = torch.argmax(logits, dim=1).item()
    result = list(ENG_TO_KOR_KEY.keys())[predicted_idx]
    if DEBUG:
        print("[디버그] classify_intent_with_bert - Predicted technique (English):", result)
    return result

def retrieve_info_by_approach(query, approach, df, state, top_k=3):
    if approach not in state["faiss_index_cache"]:
        filtered_df = df[df["approach"] == approach]
        if filtered_df.empty:
            raise ValueError(f"No data found for approach: {approach}")
        input_list = filtered_df["user_input"].dropna().tolist()
        emb = embedding_model.encode(input_list, convert_to_tensor=True)
        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb.cpu().numpy())
        state["faiss_index_cache"][approach] = (index, filtered_df)
    index, filtered_df = state["faiss_index_cache"][approach]
    query_vector = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    _, indices = index.search(query_vector, top_k)
    if DEBUG:
        print("[디버그] retrieve_info_by_approach - Retrieved indices:", indices)
    return filtered_df.iloc[indices[0][0]]['info']

def merge_hints_to_utterance(hints):
    return " ".join([f'For the question "{q}", the answer was "{a}".' for q, a in hints])

def postprocess_response(text: str) -> str:
    clean = re.sub(r'\s+', ' ', text).strip()
    if not re.search(r'[.!?]"?$', clean):
        clean += "."
    return clean

def generate_confidence_field(user_input, context, depth):
    conf_prompt = f"""You are a CBT-I assistant.
Evaluate your confidence in understanding the user's situation.
User's statement: "{user_input}"
Context: "{context}"
Depth: {depth}
Respond ONLY with one word: low, middle, or high.
"""
    out = llama_pipeline(conf_prompt, max_new_tokens=10, temperature=0.4, top_p=0.9)[0]['generated_text']
    conf = out.strip().lower()
    if conf not in ["low", "middle", "high"]:
        conf = "low"
    return conf

def generate_type_field(user_input, context, depth, state):
    type_prompt = f"""
You are a Socratic Question Classifier for CBT-I sleep therapy chatbot.
Below are 5 Socratic Question Types based on Paul & Elder (2019), each with a description and example:
1) clarity
   - Probes unclear or vague thoughts.
   - Example: "What do you mean by that?"
2) assumptions
   - Probes hidden assumptions or beliefs.
   - Example: "What assumptions are you making?"
3) reasons_evidence
   - Probes the reasoning or evidence behind a claim.
   - Example: "What makes you think this is true?"
4) implication_consequences
   - Probes what might happen next.
   - Example: "What do you think will happen if this continues?"
5) alternate_viewpoints_perspectives
   - Probes different angles or perspectives.
   - Example: "Is there another way to look at this?"
--------------------------------------------------
Now classify the user’s statement:
User: "{user_input}"
Context: "{context}"
Respond in the following format only:
Type: <one of clarity, assumptions, reasons_evidence, implication_consequences, alternate_viewpoints_perspectives>
""".strip()
    try:
        out = llama_pipeline(type_prompt, max_new_tokens=60, temperature=0.3, top_p=0.8)[0]["generated_text"]
        if DEBUG:
            print("[디버그] generate_type_field - Raw output:", out)
        typ = extract_type_flexible(out)
        if typ is None:
            typ = "clarity"

        if "type_history" not in state:
            state["type_history"] = []
        state["type_history"].append(typ)
        if len(state["type_history"]) > 3:
            state["type_history"].pop(0)

        if state["type_history"].count("clarity") >= 2:
            typ = random.choice(["assumptions", "reasons_evidence", "implication_consequences", "alternate_viewpoints_perspectives"])
            if DEBUG:
                print(f"[디버그] clarity 과다 탐지 - 강제 전환: {typ}")
        return typ
    except Exception as e:
        print(f"🚨 generate_type_field 오류: {e}")
        return "clarity"

def generate_acknowledgment(user_input, context, depth):
    ack_prompt = f"""
You are a compassionate CBT-I assistant. 
The user is sharing their sleep-related struggles.
User statement: "{user_input}"
Conversation context: "{context}"
Depth: {depth}
Instructions:
1. Generate a brief but emotionally nuanced empathetic acknowledgment.
2. Reflect the *specific emotional tone* of the user's message (e.g., frustration, sadness, anxiety, exhaustion).
3. Avoid generic phrases like "I understand" or "Your feelings are valid."
4. Use diverse expressions of empathy that sound natural and human.
5. You may use metaphors, imagery, or personal-style expressions to show genuine care.
Examples:
- "That must be incredibly draining to go through every night."
- "It sounds like you're carrying a lot of stress, and that's completely understandable."
- "I can really feel how upsetting this has been for you."
- "You're doing your best, and this sounds tougher than most people realize."
Respond ONLY with the empathetic sentence.
"""
    out = llama_pipeline(ack_prompt, max_new_tokens=60, temperature=0.5, top_p=0.9)[0]['generated_text']
    ack = out.strip()
    return translate_to_korean(ack)

def generate_subquestion_field(user_input, context, depth, typ):
    prompt = llama_pipeline.tokenizer.apply_chat_template([
        {"role": "system", "content": (
            "You are a Socratic therapist helping a user struggling with sleep problems."
            " Your job is to generate ONE Socratic follow-up question that matches the given type,"
            " but also feels natural, emotionally supportive, and contextually appropriate."
            " Do not be abstract. Sound like a real therapist helping someone who feels anxious, overwhelmed, or restless."
        )},
        {"role": "user", "content": f"""
User's concern: "{user_input}"
Previous conversation context: "{context}"
Socratic question type: {typ}
Instructions:
- Only generate ONE natural, emotionally grounded question.
- Avoid vague or philosophical questions.
- Ask a specific and supportive question that could realistically come from a CBT-I therapist.
- Return ONLY the English question. No quotes, no explanations.
"""}
    ], tokenize=False, add_generation_prompt=True)

    try:
        result = llama_pipeline(prompt, max_new_tokens=60, temperature=0.4, top_p=0.9)[0]['generated_text']
        question = result[len(prompt):].strip()
        if "?" not in question or len(question) < 5 or any(bad in question.lower() for bad in [
            "accomplish", "goal", "trying to", "how do you feel", "purpose"
        ]):
            fallback = {
                "clarity": "Could you tell me more about what that feels like?",
                "assumptions": "What might you be assuming when that thought comes up?",
                "reasons_evidence": "What makes you think that will happen?",
                "implication_consequences": "What do you think might happen if this continues?",
                "alternate_viewpoints_perspectives": "Is there another way you could view this situation?"
            }
            question = fallback.get(typ, "Could you explain a bit more about that?")
        return question
    except Exception as e:
        print(f"🚨 generate_subquestion_field 오류: {e}")
        return "Could you explain a bit more about that?"

def generate_full_subquestion_v2(user_input, context="", depth=0, state=None):
    conf = generate_confidence_field(user_input, context, depth)
    typ = generate_type_field(user_input, context, depth, state)
    subq = generate_subquestion_field(user_input, context, depth, typ)
    subq_kr = translate_to_korean(subq)
    ack = generate_acknowledgment(user_input, context, depth)
    state["current_confidence"] = conf
    state["current_type"] = typ
    final_output = f"{ack.strip()} 혹시 {subq_kr.strip().rstrip('.').rstrip('?')}?"
    if DEBUG:
        print("[디버그] generate_full_subquestion_v2 - confidence:", conf)
        print("[디버그] generate_full_subquestion_v2 - type:", typ)
        print("[디버그] generate_full_subquestion_v2 - subquestion:", subq)
        print("[디버그] generate_full_subquestion_v2 - subquestion (KR):", subq_kr)
        print("[디버그] generate_full_subquestion_v2 - acknowledgment:", ack)
        print("[디버그] generate_full_subquestion_v2 - 최종 출력:", final_output)
    return postprocess_response(final_output)

def generate_response(user_input, approach_en, context, include_termination=True):
    kor_key = ENG_TO_KOR_KEY.get(approach_en)
    desc_kr = CBT_I_DESCRIPTIONS.get(kor_key, "")
    sentences = re.split(r'(?<=[.!?])\s+', context)
    summary = ' '.join(sentences[:2])
    prompt = llama_pipeline.tokenizer.apply_chat_template([
        {"role": "system", "content": (
            "You are a warm and empathetic CBT-I therapist. Respond with a natural, conversational answer in Korean. "
            "Please express empathy and provide practical CBT advice based on the selected technique. "
            "Do not use numbered lists or bullet points."
        )},
        {"role": "user", "content": f"""
User concern: {user_input}
Recommended CBT-I technique: {approach_en}
CBT-I Description: {desc_kr}
Extra context: {summary}
Please return your response in Korean.
If include_termination is True, end your answer with a polite question asking if the user wants to end the session.
Otherwise, just end your response naturally.
"""}
    ], tokenize=False, add_generation_prompt=True)
    out = llama_pipeline(prompt, max_new_tokens=180, temperature=0.4, top_p=0.8)[0]['generated_text']
    english_response = re.sub(r'\s+', ' ', out[len(prompt):].strip())
    korean_response = translate_to_korean(english_response)
    return postprocess_response(korean_response)

def generate_self_decision_message(state):
    combined = state["consult_query"] + " " + merge_hints_to_utterance(state["socratic_hints"])
    approach_en = classify_intent_with_bert(combined).strip()
    state["recommended_approach"] = approach_en
    kor_key = ENG_TO_KOR_KEY.get(approach_en)
    desc_kr = CBT_I_DESCRIPTIONS.get(kor_key, "")
    prompt = llama_pipeline.tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a CBT-I expert who provides natural, empathetic recommendations based on user concerns."},
        {"role": "user", "content": f"""
The user is experiencing sleep issues and has provided the following concern and dialogue history:
Concern: {combined}
Based on this, you recommend the CBT-I technique: "{kor_key}" ({approach_en}).
Here is a brief explanation of the technique in Korean: "{desc_kr}"
Please:
- Begin with an empathetic statement
- Clearly explain why this specific technique could be helpful
- End your response with a polite question asking if the user would like to try this method.
Return your answer in natural Korean.
"""}
    ], tokenize=False, add_generation_prompt=True)
    
    raw = llama_pipeline(prompt, max_new_tokens=180, temperature=0.4, top_p=0.8)[0]['generated_text']
    english_reply = re.sub(r'\s+', ' ', raw[len(prompt):].strip())
    korean_reply = translate_to_korean(english_reply)

    # ✅ 여기 수정
    termination_kor = "이 방법을 시도해보시겠어요? '예' 또는 '아니오'로 답해주세요."
    if not korean_reply.strip().endswith(termination_kor):
        korean_reply += "\n\n" + termination_kor  # 줄바꿈 추가로 말풍선 위 클리핑 방지

    return postprocess_response(korean_reply)

def generate_personalized_advice(user_input, last_advice, technique_name):
    prompt = llama_pipeline.tokenizer.apply_chat_template([
        {"role": "system", "content": (
            "You are a CBT-I sleep therapist. The user has already received advice about a specific CBT-I technique, "
            "and is now asking a follow-up question or sharing a concern related to applying it. "
            "Respond in a natural, conversational tone with personalized advice that is warm and supportive. "
            "Do not use numbered lists or bullet points."
        )},
        {"role": "user", "content": f"""
CBT-I technique: {technique_name}
Previous advice given: "{last_advice}"
User's follow-up message:
"{user_input}"
Please respond in Korean in a natural manner without repeating the entire explanation of the technique.
"""}
    ], tokenize=False, add_generation_prompt=True)
    output = llama_pipeline(prompt, max_new_tokens=180, temperature=0.5, top_p=0.9)[0]['generated_text']
    return postprocess_response(translate_to_korean(output[len(prompt):].strip()))

def finalize_socratic_and_advice(state):
    merged = merge_hints_to_utterance(state["socratic_hints"])
    query = state["consult_query"] + " " + merged
    approach_en = state["recommended_approach"]
    if not approach_en:
        raise ValueError("No recommended_approach found.")
    info = retrieve_info_by_approach(query, approach_en, df, state)
    korean_response = generate_response(query, approach_en, info, include_termination=False)
    return postprocess_response(korean_response + " 추가 의견이 있으시면 입력해 주세요. 만족하시면 '만족'을 입력해 주세요.")

def save_history_to_json(state):
    # 환경변수에서 Hugging Face Token 불러오기 (기존 방식과 동일)
    HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
    if HF_ACCESS_TOKEN is None:
        raise ValueError("HF_ACCESS_TOKEN 환경변수가 설정되지 않았습니다.")

    # 1. 로컬에 저장
    os.makedirs("./history", exist_ok=True)
    filename = f"./history/{state['user_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(state["history"], f, ensure_ascii=False, indent=4)

    # 2. Hugging Face Hub에 업로드
    try:
        api = HfApi(token=HF_ACCESS_TOKEN)  # ✅ 여기 핵심

        repo_id = "youjin129/cbt-i-history"
        repo_type = "dataset"
        path_in_repo = f"history/{os.path.basename(filename)}"

        # 리포지토리가 없으면 생성
        try:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        except HfHubHTTPError as e:
            if e.response.status_code == 409:
                print(f"[INFO] 이미 존재하는 리포지토리입니다: {repo_id}")
            else:
                raise

        # 업로드
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type
        )
        upload_msg = "✅ 대화 기록이 Hugging Face Hub에 업로드되었습니다."
    except Exception as e:
        upload_msg = f"❌ 대화 기록 업로드 실패: {e}"

    return f"💾 대화 기록이 저장되었습니다: {filename}\n{upload_msg}"

# ------------------------------
# 5. 상담/학습 모드 처리 함수 (state 인자 추가)
# ------------------------------

def process_learning_mode(user_input, state):
    # 초기 학습 모드 시작 시, "예"가 아니라면 설득 메시지 출력 후 다시 물어봄
    if state.get("learning_index", 0) == 0:
        if normalize_yes_no(user_input) != "예":
            state["history"].append((None, "수면 개선 기법에 관한 설명은 매우 유익합니다. 제가 알려드릴 기법들이 큰 도움이 될 거예요. '예'라고 입력해 주시면 시작하도록 하겠습니다."))
            return state["history"], state
    
    if user_input.strip() != "":
        idx = state.get("learning_index", 0)
        if idx < len(TECHNIQUES_ORDER):
            technique = TECHNIQUES_ORDER[idx]
            explanation = CBT_I_DESCRIPTIONS.get(technique, "설명이 없습니다.")
            state["learning_index"] = idx + 1
            if state["learning_index"] < len(TECHNIQUES_ORDER):
                msg = (f"[{technique}]\n\n{explanation}\n\n"
                       "다음 기법으로 넘어가기 위해 아무 내용이나 입력해 주세요.")
            else:
                msg = (f"[{technique}]\n\n{explanation}\n\n"
                       "모든 기법 학습을 마쳤습니다. 이제 상담 모드로 전환해도 되겠습니까?😊 원하신다면 수면에 대한 고민을 입력해주세요.")
                state["mode"] = "상담 모드"
            state["history"].append((None, msg))
            return state["history"], state
        else:
            state["mode"] = "상담 모드"
            state["history"].append((None, "모든 기법 학습을 마쳤습니다. 이제 상담 모드로 전환해도 되겠습니까?😊 원하신다면 수면에 대한 고민을 입력해주세요."))
            return state["history"], state
    else:
        state["history"].append((None, "아무 내용이라도 입력해 주세요."))
        return state["history"], state

def consult_mode(user_input, state):
    chat = state["history"]
    
    # 상담 모드 초기 진입 시, 입력이 충분한 고민인지 확인 (예: "ㅇ", ".", "ㄹ" 등은 부적합)
    if not state.get("consulting_active"):
        if not is_valid_concern(user_input):
            chat.append((None, "수면에 대해 조금 더 구체적인 고민을 입력해 주세요."))
            return chat, state
        else:
            eng_input = translate_to_english(user_input)
            state.update({
                "consulting_active": True,
                "socratic_active": True,
                "consult_query": eng_input,
                "socratic_depth": 0,
                "socratic_hints": []
            })
            subq = generate_full_subquestion_v2(eng_input, depth=0, state=state)
            state["current_subquestion"] = subq
            chat.append((user_input, None))
            chat.append((None, subq))
            return chat, state

    # 추가 개선 1): 진행 중에도 무의미한 입력(예: 한 글자 등)을 거부하고 구체적 고민을 재요구
    if not is_valid_concern(user_input) and normalize_yes_no(user_input) is None:
        chat.append((None, "입력이 너무 짧습니다. 수면 문제에 대해 구체적으로 말씀해 주세요."))
        return chat, state

    # 0. iterative advice 진행 중이면 먼저 처리
    if state.get("iterative_advice_active", False):
        feedback = user_input.strip()
        chat.append((user_input, None))
        if feedback.lower() == "만족":
            final_advice = state.get("current_iterative_advice", "")
            state["iterative_advice_active"] = False
            state["awaiting_termination"] = True
            chat.append((None, "대화를 종료하고 싶으신가요? '예' 또는 '아니오'로 답해주세요."))
            return chat, state
        else:
            state["iterative_context"] += " " + feedback
            last_advice = state.get("current_iterative_advice", "")
            tech_name = ENG_TO_KOR_KEY.get(state["recommended_approach"], "수면 기법")
            new_advice = generate_personalized_advice(feedback, last_advice, tech_name)
            state["current_iterative_advice"] = new_advice
            chat.append((None, new_advice + "\n추가 의견이 있으시면 입력해 주세요. 만족하시면 '만족'을 입력해 주세요."))
            return chat, state

    # 1. 종료 처리
    if state.get("awaiting_termination", False):
        ans = user_input.strip().lower()
        chat.append((user_input, None))
        if ans == "예":
            msg = save_history_to_json(state)
            chat.append((None, f"감사합니다. 대화를 종료합니다. {msg}"))
            state = reset_state(state)
            return chat, state
        elif ans == "아니오":
            chat.append((None, "대화를 계속 진행합니다. 어떤 고민이 있으신가요?"))
            state["awaiting_termination"] = False
            return chat, state
        else:
            chat.append((None, "대화를 종료하고 싶으신가요? '예' 또는 '아니오'로 답해주세요."))
            return chat, state

    if user_input.strip().lower() in ["exit"]:
        msg = save_history_to_json(state)
        chat.append((None, f"대화가 종료되었습니다. {msg} 다시 시작하고 싶으시면 이름을 입력해주세요."))
        state = reset_state(state)
        return chat, state

    if state.get("waiting_end_confirmation", False):
        ans = user_input.strip().lower()
        if ans == "예":
            msg = save_history_to_json(state)
            chat.append((None, f"감사합니다. 대화를 종료합니다. {msg}"))
            state = reset_state(state)
            return chat, state
        elif ans == "아니오":
            chat.append((None, "대화를 계속 진행합니다. 어떤 고민이 있으신가요?"))
            state["waiting_end_confirmation"] = False
            return chat, state
        else:
            chat.append((None, "대화를 종료하고 싶으신가요? '예' 또는 '아니오'로 답해주세요."))
            return chat, state

    # 2. Self-decision 단계
    if state.get("self_decision_pending", False):
        ans = user_input.strip().lower()
        if ans == "예":
            state["iterative_advice_active"] = True
            state["iterative_context"] = state["consult_query"] + " " + merge_hints_to_utterance(state["socratic_hints"])
            info = retrieve_info_by_approach(state["iterative_context"], state["recommended_approach"], df, state)
            initial_advice = generate_response(state["iterative_context"], state["recommended_approach"], info)
            state["current_iterative_advice"] = initial_advice
            chat.append((None, initial_advice + "\n추가 의견이 있으시면 입력해 주세요. 만족하시면 '만족'을 입력해 주세요."))
            state["self_decision_pending"] = False
            return chat, state
        elif ans == "아니오":
            response_text = ("알겠습니다. 대신, 아래 기법들 중 하나를 사용하기 원하신다면 해당되는 번호(1~5)를 입력해 주세요:\n"
                             "1. 수면 제한 요법 (Sleep Restriction)\n"
                             "2. 자극 조절 요법 (Stimulus Control)\n"
                             "3. 수면 위생 교육 (Sleep Hygiene)\n"
                             "4. 이완 요법 (Relaxation Techniques)\n"
                             "5. 인지적 재구성 (Cognitive Restructuring)")
            chat.append((None, response_text))
            state["self_decision_pending"] = False
            state["technique_selection_pending"] = True
            return chat, state
        else:
            chat.append((None, "‘예’ 또는 ‘아니오’로 답해 주세요."))
            return chat, state

    # 3. Technique 선택 단계 (번호 입력 시 끝에 점(.) 붙은 경우도 처리)
    if state.get("technique_selection_pending", False):
        # 번호 뒤에 있는 '.' 제거하고 비교
        normalized_ans = re.sub(r'\.$', '', user_input.strip())
        if normalized_ans in ["1", "2", "3", "4", "5"]:
            matched_key = list(ENG_TO_KOR_KEY.keys())[int(normalized_ans)-1]
            state["recommended_approach"] = matched_key
            state["iterative_advice_active"] = True
            state["iterative_context"] = state["consult_query"] + " " + merge_hints_to_utterance(state["socratic_hints"])
            info = retrieve_info_by_approach(state["iterative_context"], state["recommended_approach"], df, state)
            initial_advice = generate_response(state["iterative_context"], state["recommended_approach"], info, include_termination=False)
            state["current_iterative_advice"] = initial_advice
            chat.append((None, initial_advice + "\n추가 의견이 있으시면 입력해 주세요. 만족하시면 '만족'을 입력해 주세요."))
            state["technique_selection_pending"] = False
            return chat, state
        else:
            chat.append((None, "잘못된 번호입니다. 아래 번호 중 하나를 입력해 주세요:\n"
                               "1. 수면 제한 요법 (Sleep Restriction)\n"
                               "2. 자극 조절 요법 (Stimulus Control)\n"
                               "3. 수면 위생 교육 (Sleep Hygiene)\n"
                               "4. 이완 요법 (Relaxation Techniques)\n"
                               "5. 인지적 재구성 (Cognitive Restructuring)"))
            return chat, state

    # 4. 초기 상담 시작 (상담 모드로 전환 후 최초 상담)
    if not state["consulting_active"]:
        eng_input = translate_to_english(user_input)
        state.update({
            "consulting_active": True,
            "socratic_active": True,
            "consult_query": eng_input,
            "socratic_depth": 0,
            "socratic_hints": []
        })
        subq = generate_full_subquestion_v2(eng_input, depth=0, state=state)
        state["current_subquestion"] = subq
        chat.append((user_input, None))
        chat.append((None, subq))
        return chat, state

    # 5. 진행 중인 상담 처리
    chat.append((user_input, None))
    eng_input = translate_to_english(user_input)
    state["socratic_hints"].append((state["current_subquestion"], eng_input))
    state["socratic_depth"] += 1
    if state["current_confidence"] == "high" or state["socratic_depth"] >= state["max_depth"]:
        decision_msg = generate_self_decision_message(state)
        chat.append((None, decision_msg))
        state["self_decision_pending"] = True
        return chat, state
    ctx = state["consult_query"] + " " + merge_hints_to_utterance(state["socratic_hints"])
    subq = generate_full_subquestion_v2(eng_input, ctx, depth=state["socratic_depth"], state=state)
    state["current_subquestion"] = subq
    chat.append((None, subq))
    return chat, state

# ------------------------------
# 6. 최상위 콜백 함수 (state를 함께 전달)
# ------------------------------

def user_input_handler(user_input, state):
    if state.get("mode") == "학습 모드":
        history, state = process_learning_mode(user_input, state)
        return history, "", state
    else:
        history, state = consult_mode(user_input, state)
        return history, "", state

def chatbot_entry(name, state):
    state = reset_state(state)
    state["user_name"] = name.strip() if name and name.strip() else f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    state["mode"] = "학습 모드"
    learning_msg = "수면을 개선하는 데 도움이 되는 다섯 가지 기법이 있어요. 하나씩 함께 설명을 드려도 될까요?😊 계속 진행하려면 '예'를 입력해 주세요."
    state["history"].append((None, f"반갑습니다 {state['user_name']}님! {learning_msg}"))
    return state["history"], state, gr.update(visible=False)

TECHNIQUES_ORDER = [
    "수면 제한 요법 (Sleep Restriction)",
    "자극 조절 요법 (Stimulus Control)",
    "수면 위생 교육 (Sleep Hygiene)",
    "이완 요법 (Relaxation Techniques)",
    "인지적 재구성 (Cognitive Restructuring)"
]

# ------------------------------
# 7. Gradio 인터페이스 구성 (각 세션마다 state가 독립적임)
# ------------------------------

with gr.Blocks(css=".gradio-container { width: 80% !important; }") as demo:
    chatbot = gr.Chatbot(label="수면 인지 행동 치료 챗봇", bubble_full_width=True)
    name_input = gr.Textbox(label="이름을 먼저 입력해주세요", placeholder="닉네임을 입력해주세요.")
    start_button = gr.Button("대화 시작")
    user_input = gr.Textbox(label="메시지를 입력해주세요.")
    # 세션별 state 초기화 (각 사용자는 get_initial_state()를 별도로 가짐)
    session_state = gr.State(get_initial_state())

    # 시작 버튼: 이름 입력 후 학습 모드 시작
    start_button.click(fn=chatbot_entry,
                       inputs=[name_input, session_state],
                       outputs=[chatbot, session_state, name_input])
    
    # 사용자 입력: state를 함께 전달하여 모드에 따라 분기 처리
    user_input.submit(fn=user_input_handler,
                      inputs=[user_input, session_state],
                      outputs=[chatbot, user_input, session_state])

demo.launch(share=False)