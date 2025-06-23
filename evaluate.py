# /scripts/evaluate.py
import os
import torch
import pandas as pd
import random
import numpy as np
import json
import zipfile
import gc
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from deepeval.test_case import LLMTestCase
import deepeval

# Importa a métrica customizada do outro arquivo
from custom_metrics.execution_accuracy import ExecutionAccuracyMetric

# --- Constantes e Configurações ---
SEED = 42
MODEL_ID = "google/gemma-2b-it"
ZIP_FILE_NAME = "spider_data.zip"
EXTRACT_PATH = "./spider_unzipped"
DB_DIR_LOCAL = os.path.join(EXTRACT_PATH, "database")
EXP1_OUTPUT_DIR = "gemma-2b-spider-ft-exp1"
EXP2_OUTPUT_DIR = "gemma-2b-spider-ft-exp2"
MMLU_DATASET = "cais/mmlu"
MMLU_CATEGORIES = {"STEM": "college_computer_science", "Humanidades": "philosophy", "Ciências Sociais": "high_school_macroeconomics"}
SAMPLES_PER_CATEGORY_QUICK = 3

# --- Funções Auxiliares ---
def setup_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def load_dev_data():
    if not os.path.exists(EXTRACT_PATH):
      print(f"Diretório {EXTRACT_PATH} não encontrado. Execute o script de treino primeiro.")
      exit()
    with open(os.path.join(EXTRACT_PATH, 'dev.json'), 'r') as f:
        dev_json = json.load(f)
    return Dataset.from_dict({'db_id': [d['db_id'] for d in dev_json], 'question': [d['question'] for d in dev_json], 'query': [d['query'] for d in dev_json]})

def generate_sql_with_model(model, tokenizer, question):
    prompt = f"<start_of_turn>user\nTranslate the following question into a valid SQL query.\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("model\n")[-1].strip()

def evaluate_sql_accuracy(model, tokenizer, dev_dataset, num_samples=15):
    print(f"Avaliando SQL em {num_samples} amostras...")
    test_cases = [LLMTestCase(input=item['db_id'], actual_output=generate_sql_with_model(model, tokenizer, item['question']), expected_output=item['query']) for item in dev_dataset.select(range(num_samples))]
    metric = ExecutionAccuracyMetric(db_dir=DB_DIR_LOCAL)
    results = deepeval.evaluate(test_cases, [metric])
    scores = [m.score for result in results for m in result.metrics if m.score is not None]
    return sum(scores) / len(scores) if scores else 0.0

def evaluate_mmlu_accuracy(model, tokenizer):
    from datasets import load_dataset as hf_load_dataset
    suite = []
    for cat_code in MMLU_CATEGORIES.values():
        data = hf_load_dataset(MMLU_DATASET, cat_code, split="test").shuffle(seed=SEED)
        suite.extend(data.select(range(SAMPLES_PER_CATEGORY_QUICK)))
    
    correct = 0
    total = len(suite)
    print(f"Avaliando MMLU em {total} questões...")
    for i, item in enumerate(suite):
        dev_set = hf_load_dataset(MMLU_DATASET, item['subset'], split='dev', trust_remote_code=True)
        shots = "".join([f"Question: {dev_set[i]['question']}\nAnswer: {['A', 'B', 'C', 'D'][dev_set[i]['answer']]}\n\n" for i in range(4)])
        choices = "".join([f"\n{chr(65+j)}. {item['choices'][j]}" for j in range(len(item['choices']))])
        prompt = f"<start_of_turn>user\n...{shots}Question: {item['question']}{choices}\nAnswer:<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
        prediction = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        if prediction and prediction[0].upper() == ['A', 'B', 'C', 'D'][item['answer']]:
            correct += 1
    return correct / total if total > 0 else 0

# --- Lógica Principal do Script ---
def main():
    setup_seeds()
    dev_dataset = load_dev_data()
    
    # Carrega modelo base e tokenizer
    print("Carregando modelo base para avaliação...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Avaliação do Baseline
    print("\n--- Avaliando Modelo Base ---")
    base_sql_acc = evaluate_sql_accuracy(base_model, tokenizer, dev_dataset)
    base_mmlu_acc = evaluate_mmlu_accuracy(base_model, tokenizer)
    
    # Avaliação do Experimento 1
    print("\n--- Avaliando Modelo Fine-Tuned 1 ---")
    model_ft1 = PeftModel.from_pretrained(base_model, EXP1_OUTPUT_DIR)
    exp1_sql_acc = evaluate_sql_accuracy(model_ft1, tokenizer, dev_dataset)
    exp1_mmlu_acc = evaluate_mmlu_accuracy(model_ft1, tokenizer)
    
    # Avaliação do Experimento 2
    print("\n--- Avaliando Modelo Fine-Tuned 2 ---")
    model_ft2 = PeftModel.from_pretrained(base_model, EXP2_OUTPUT_DIR)
    exp2_sql_acc = evaluate_sql_accuracy(model_ft2, tokenizer, dev_dataset)
    exp2_mmlu_acc = evaluate_mmlu_accuracy(model_ft2, tokenizer)
    
    # Montar e exibir a tabela de resultados
    results_data = {
        'Modelo': ['Base (Gemma-2B)', 'Fine-Tune (Exp 1)', 'Fine-Tune (Exp 2)'],
        'Execution Accuracy (SQL)': [f"{base_sql_acc:.2%}", f"{exp1_sql_acc:.2%}", f"{exp2_sql_acc:.2%}"],
        'Acurácia MMLU (Geral)': [f"{base_mmlu_acc:.2%}", f"{exp1_mmlu_acc:.2%}", f"{exp2_mmlu_acc:.2%}"]
    }
    df = pd.DataFrame(results_data)
    df['Regressão de Capacidade (%)'] = [ "N/A", f"{((exp1_mmlu_acc - base_mmlu_acc) / base_mmlu_acc) * 100:.2f}%" if base_mmlu_acc > 0 else "N/A", f"{((exp2_mmlu_acc - base_mmlu_acc) / base_mmlu_acc) * 100:.2f}%" if base_mmlu_acc > 0 else "N/A"]
    
    print("\n\n--- TABELA DE RESULTADOS FINAIS ---")
    print(df.to_string())

if __name__ == "__main__":
    main()