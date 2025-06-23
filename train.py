# /scripts/train.py
import os
import torch
import random
import numpy as np
import json
import zipfile
import gc
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# --- Constantes e Configurações ---
# (Extraídas da Célula 3 do notebook)

SEED = 42
MODEL_ID = "google/gemma-2b-it"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
ZIP_FILE_NAME = "spider_data.zip" # Espera que este arquivo esteja na raiz do projeto
EXTRACT_PATH = "./spider_unzipped"

LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05

EXP_CONFIGS = {
    1: {"output_dir": "gemma-2b-spider-ft-exp1", "num_train_epochs": 1, "learning_rate": 2e-4, "per_device_train_batch_size": 4, "gradient_accumulation_steps": 1},
    2: {"output_dir": "gemma-2b-spider-ft-exp2", "num_train_epochs": 1, "learning_rate": 5e-5, "per_device_train_batch_size": 4, "gradient_accumulation_steps": 1}
}

# --- Funções Auxiliares ---
def setup_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def load_and_process_data():
    print(f"Descompactando '{ZIP_FILE_NAME}'...")
    if os.path.exists(EXTRACT_PATH):
        os.system(f"rm -rf {EXTRACT_PATH}")
    with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as z:
        z.extractall(EXTRACT_PATH)

    def load_json(fp):
        with open(fp, 'r') as f:
            return json.load(f)

    train_json = load_json(os.path.join(EXTRACT_PATH, 'train_spider.json')) + load_json(os.path.join(EXTRACT_PATH, 'train_others.json'))
    
    def convert_to_dataset(data):
        return Dataset.from_dict({'db_id': [d['db_id'] for d in data], 'question': [d['question'] for d in data], 'query': [d['query'] for d in data]})
    
    return convert_to_dataset(train_json)

def create_gemma_training_prompt(example):
    user_message = f"Translate the following question into a valid SQL query.\nQuestion: {example['question']}"
    return {"text": f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n{example['query']}<end_of_turn>"}

def run_training(model, tokenizer, dataset, lora_params, training_args_params):
    peft_config = LoraConfig(**lora_params)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    args = TrainingArguments(optim="adamw_torch", bf16=True, seed=SEED, report_to="none", save_strategy="epoch", **training_args_params)
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, dataset_text_field="text", max_seq_length=1024, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n--- Treinamento concluído. Modelo salvo em: {args.output_dir} ---")

# --- Lógica Principal do Script ---
def main():
    parser = argparse.ArgumentParser(description="Script de Fine-Tuning para o projeto Text-to-SQL.")
    parser.add_argument("--experiment_num", type=int, choices=[1, 2], required=True, help="Número do experimento a ser executado (1 ou 2).")
    cli_args = parser.parse_args()

    # Setup
    setup_seeds()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {DEVICE}")

    # Carregamento e Processamento de Dados
    spider_train_data = load_and_process_data()
    formatted_train_dataset = spider_train_data.map(create_gemma_training_prompt, remove_columns=spider_train_data.column_names)
    
    # Otimização para "Modo Rápido"
    quick_train_dataset = formatted_train_dataset.select(range(500))
    print(f"Usando subconjunto de treino rápido com {len(quick_train_dataset)} exemplos.")
    
    # Carregamento do Modelo
    print(f"Carregando modelo base: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Execução do Experimento Selecionado
    exp_num = cli_args.experiment_num
    print(f"\n--- Iniciando Experimento de Treinamento {exp_num} ---")
    lora_params = {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": LORA_TARGET_MODULES, "bias":"none", "task_type":"CAUSAL_LM"}
    training_config = EXP_CONFIGS[exp_num]
    
    run_training(model, tokenizer, quick_train_dataset, lora_params, training_config)

if __name__ == "__main__":
    main()