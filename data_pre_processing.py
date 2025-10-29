import json
import re
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from google.cloud import storage
import os
import tempfile
import psutil

import functions_framework
from google.cloud import pubsub_v1

def process_data():
    """Core data processing logic that can be called by any trigger."""
    HF_TOKEN = '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'
    GCS_BUCKET = "training-datasets-v1"
    # Initialize storage client
    storage_client = storage.Client()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=HF_TOKEN)
    
    # --------------------------------------------------------------------------------------------
    # DETERMINE WHICH DATA TO UPLOAD
    try_metamath = False
    try_numina = True
    try_bespoke = True
    

    try:
        # 1. Loading, Processing, and saving MetaMathQA Data
        if try_metamath:
            file_name = "metamathqa.json"
            output_dir = "/tmp/metamathqa_proc/"
            samples = load_and_process_metamathqa_data(tokenizer, output_dir, file_name)
            num_metamathqa_samples = len(samples)
            file_path = os.path.join(output_dir, file_name)
            if os.path.isfile(file_path):
                print(f"MetaMathQA Processed File '{file_name}' exists in directory '{output_dir}'.")
            else:
                print(f"ERROR: MetaMathQA Processed File '{file_name}' does NOT exist in directory '{output_dir}'.")
                raise Exception("CANNOT FIND UPLOADED METAMATHQA JSON FILE")
            
            # Write both processed files to the same directory in the GCS bucket
            gcs_dir = "metamathqa/"

            # Upload MetaMathQA processed file
            meta_gcs_path = os.path.join(gcs_dir, file_name)
            meta_bucket = storage_client.bucket(GCS_BUCKET)
            meta_blob = meta_bucket.blob(meta_gcs_path)
            with open(file_path, "rb") as f:
                meta_blob.upload_from_file(f)
            print(f"Uploaded MetaMathQA processed file to gs://{GCS_BUCKET}/{meta_gcs_path}")

        
        # 2. Loading, Processing, and saving BespokeStratos 17K file
        if try_bespoke:
            bespoke_file_name = "bespoke_stratos.json"
            bespoke_output_dir = "/tmp/bespoke_stratos_proc/"
            samples = load_and_process_bespoke_stratos(tokenizer, bespoke_output_dir, bespoke_file_name)
            num_bespoke_stratos_samples = len(samples)
            bespoke_file_path = os.path.join(bespoke_output_dir, bespoke_file_name)
            if os.path.isfile(bespoke_file_path):
                print(f"BespokeStratos Processed File '{bespoke_file_name}' exists in directory '{bespoke_output_dir}'.")
            else:
                print(f"ERROR: BespokeStratos Processed File '{bespoke_file_name}' does NOT exist in directory '{bespoke_output_dir}'.")
                raise Exception("CANNOT FIND UPLOADED BESPOKESTRATOS JSON FILE")
            
            # Upload BespokeStratos processed file
            gcs_dir = "bespoke_stratos_proc/"
            bespoke_gcs_path = os.path.join(gcs_dir, bespoke_file_name)
            bespoke_bucket = storage_client.bucket(GCS_BUCKET)
            bespoke_blob = bespoke_bucket.blob(bespoke_gcs_path)
            with open(bespoke_file_path, "rb") as f:
                bespoke_blob.upload_from_file(f)
            print(f"Uploaded BespokeStratos processed file to gs://{GCS_BUCKET}/{bespoke_gcs_path}")
        
        # 3. Loading, Processing, and saving NuminaMath-CoT
        if try_numina:
            print("Starting NuminaMath-CoT processing...")
            numina_samples = load_and_process_numina_data(tokenizer)
            num_numina_samples = len(numina_samples)
            
            # Upload NuminaMath-CoT processed train and test files separately
            numina_file_name = "numina_math.json"
            numina_output_base_dir = "/tmp/numina_proc/"
            gcs_base_dir = "numina_math_proc/"

            for split in ["train", "test"]:
                local_dir = os.path.join(numina_output_base_dir, split)
                file_path = os.path.join(local_dir, numina_file_name)
                if os.path.isfile(file_path):
                    print(f"NuminaMath {split.capitalize()} Processed File '{numina_file_name}' exists in directory '{local_dir}'.")
                    
                    # Upload to GCS in subdirectory for split
                    gcs_dir = os.path.join(gcs_base_dir, split)
                    numina_gcs_path = os.path.join(gcs_dir, numina_file_name)
                    numina_bucket = storage_client.bucket(GCS_BUCKET)
                    numina_blob = numina_bucket.blob(numina_gcs_path)
                    with open(file_path, "rb") as f:
                        numina_blob.upload_from_file(f)
                    print(f"Uploaded NuminaMath {split} processed file to gs://{GCS_BUCKET}/{numina_gcs_path}")
                else:
                    print(f"ERROR: NuminaMath {split.capitalize()} Processed File '{numina_file_name}' does NOT exist in directory '{local_dir}'.")
                    raise Exception(f"CANNOT FIND UPLOADED NUMINAMATH {split.upper()} JSON FILE")
            
        return_str = "Successfully processed: "
        if try_metamath:
            return_str += f"{num_metamathqa_samples} MetaMathQA samples, " 
        if try_bespoke:
            return_str += f"{num_bespoke_stratos_samples} BespokeStratos samples, "
        if try_numina:
            return_str += f"{num_numina_samples} NuminaMath-CoT samples"
        
        return return_str 
    except Exception as e:
        print(f"Error: {str(e)}")
        return f'Error: {str(e)}'


@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function entry point.
    Args:
        request (flask.Request): The request object.
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
    """
    print(f"Received HTTP Request: {request}")
    result = process_data()
    print(f"Processing result: {result}")

    return result 


@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    """Pub/Sub Cloud Function entry point.
    Args:
        cloud_event: The CloudEvent object.
    Returns:
        None
    """
    print(f"Received Pub/Sub message: {cloud_event.data}")
    result = process_data()
    print(f"Processing result: {result}")
    return result

def load_and_process_metamathqa_data(tokenizer, output_dir, file_name):
    dataset_name = 'meta-math/MetaMathQA'
    all_samples = {}
    # ========================================================================
    # DATASET 2: MetaMathQA (Augmented Math - 50K subset)
    # Domain: Synthetic math reasoning with augmentation
    # TAKE THE WHOLE DATASET
    # ========================================================================
    try:
        print("Loading MetaMathQA (Augmented math reasoning)")
        metamath = load_dataset(dataset_name, split="train")
        for i in range(len(metamath)):
            query = metamath[i]['query']
            cot_text = metamath[i]['response']
            query_type = metamath[i]['type']

            # Find the answer in the response
            out_match = re.search(r"The answer is:(.*)", cot_text)
            if out_match:
                answer = out_match.group(1).strip()
                cot_reasoning = cot_text[:out_match.start()].rstrip()
                # print(f"Correct dissection of Reasoning: {cot_reasoning[:15]}")
            else:
                # If not found, save the entire response as cot_reasoning and answer as empty string
                # MAY NEED TO TREAT TEXT AS 10:1 RATIO
                cot_reasoning = cot_text
                print(f"Incorrect dissection of Reasoning: {cot_reasoning[:15]}")
                answer = ""

            if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            cot_reas_tokens = tokenizer(cot_reasoning, truncation=False, add_special_tokens=False)

            try:
              cot_token_count = len(cot_reas_tokens["input_ids"])
              # print(f"cot_token_count = {cot_token_count}")
            except Exception as e:
              cot_token_count = 0
              print(f"cot_token_count = {cot_token_count}")
              print(f"cot_reas_tokens = {cot_reas_tokens}")
              print(f"cot_reasoning = {cot_reasoning}")
              print(f"query = {query}")
              print(f"query_type = {query_type}")
              print(f"answer = {answer}")
              print(f"cot_text = {cot_text}")
              print(f"i = {i}")
              print(f"erroring sequence: {e}")
              raise e


            # Save sample in new structure
            sample = {
                "query": query,
                "cot_reasoning": cot_reasoning,
                "answer": answer,
                "cot_token_budget": cot_token_count,
                "query_type": query_type
            }
            all_samples[f"{i}"] = sample

            if i % 10000 == 0:
                print(f"   ✓ Loaded {i} samples from MetaMathQA")

        # Choose a file path to dump intermediate or final output (save as JSONL)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(all_samples, fout, ensure_ascii=False, indent=2)

        print(f"   ✓ Loaded {len(all_samples)} samples from MetaMathQA")
    except Exception as e:
        print(f"   ⚠ Could not load MetaMathQA: {e}")

    return all_samples
    
def load_and_process_bespoke_stratos(tokenizer, output_dir, file_name):
    dataset_name = "bespokelabs/Bespoke-Stratos-17k"
    all_samples = {}
   # ========================================================================
    # DATASET 3: Bespoke-Stratos-17K (Mixed Reasoning: Code + Math + Science)
    # Domain: Code (APPS/TACO), Hard Math (AIME/Olympiads), Science/Puzzles
    # ========================================================================
    try:
        print("Loading Bespoke-Stratos-17k (Code, math, science reasoning)")
        bespoke = load_dataset(dataset_name, split="train")
        print(f"Loaded {len(bespoke)} samples from Bespoke-Stratos")
        
        # Add memory usage tracking
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Memory usage before processing: {memory_before:.1f} MB")
        successful_samples = 0
        failed_samples = 0
        
        for i in range(len(bespoke)):
            system_prompt = bespoke[i]["system"]
            conversation = bespoke[i]["conversations"]
            
            # Debug first few samples
            if i < 3:
                print(f"Debug sample {i}:")
                print(f"  System prompt: {system_prompt[:100]}...")
                print(f"  Conversation length: {len(conversation)}")
                for j, entry in enumerate(conversation):
                    print(f"    Entry {j}: from={entry.get('from')}, value_length={len(entry.get('value', ''))}")
            
            query = None
            cot_reasoning = ""
            answer = ""
            full_cot_str = None
            
            # The query is typically the 'user' message
            for entry in conversation:
                if entry.get("from") == "user" and entry.get("value"):
                    query = entry["value"]
                elif entry.get("from") == "assistant" and entry.get("value"):
                    # The entire reasoning & answer is in 'value'
                    full_cot_str = entry["value"]

            if query and full_cot_str:
                # Try to extract COT text between <|begin_of_thought|> and <|end_of_thought|>
                reas_match = re.search(r"<\|begin_of_thought\|>([\s\S]+?)<\|end_of_thought\|>", full_cot_str)
                if reas_match:
                    cot_reasoning = reas_match.group(1).strip()
                    # print(f"Correct dissection of Reasoning: {cot_reasoning[:15]}")
                else:
                    cot_reasoning = ""
                    print(f"Incorrect dissection of Reasoning: {full_cot_str[:15]}")

                # Try to extract the answer using the token <|begin_of_solution|>
                answer_match = re.search(r"<\|begin_of_solution\|>(.*)<\|end_of_solution\|>", full_cot_str, re.DOTALL)
                if answer_match:
                    answer = answer_match.group(1).strip()
                else:
                    answer = ""

                # Tokenize COT rationale for budget (fall back to full_cot_str if nothing found)
                cot_text_for_count = cot_reasoning if cot_reasoning else (full_cot_str or "")
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                cot_token_count = len(tokenizer(cot_text_for_count, truncation=False, add_special_tokens=False)["input_ids"])

                # Save this as a structured example, and include 'raw_conversation'
                sample = {
                    "query": system_prompt + query,
                    "cot_reasoning": cot_reasoning,
                    "answer": answer,
                    "cot_token_budget": cot_token_count,
                    "raw_conversation": conversation
                }   
                all_samples[f"{i}"] = sample
                successful_samples += 1
            else:
                all_samples[f"{i}"] = "NOT SAVED; PROBLEM WITH PARSING"
                failed_samples += 1
                if i < 10:  # Only print first 10 failures to avoid spam
                    print(f"   ⚠ Could not load item {i} for Bespoke-Stratos, query: {query}, full chain of thought: {full_cot_str}")
            
            if i % 1000 == 0:
                memory_current = process.memory_info().rss / 1024 / 1024  # MB
                print(f"   ✓ Loaded {i + 1} samples from Bespoke-Stratos (Memory: {memory_current:.1f} MB)")
            
            if i == len(bespoke) - 1:
                memory_final = process.memory_info().rss / 1024 / 1024  # MB
                print(f"   ✓ Loaded {i + 1} samples from Bespoke-Stratos (Final Memory: {memory_final:.1f} MB)")

        # Choose a file path to dump intermediate or final output (save as JSONL)
        # output_dir = "/tmp/bespokestratos_proc"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(all_samples, fout, ensure_ascii=False, indent=2)
        print(f"   ✓ Processing complete: {successful_samples} successful, {failed_samples} failed out of {len(bespoke)} total samples")

    except Exception as e:
        print(f"   ⚠ Could not load Bespoke-Stratos: {e}")

    return all_samples


"""
NOTE FOR NUMINA COT TEXT DATA:
This implementation assumes that the model needs the latent reasoning structure of the few tokens prior to answering the question. 
THEREFORE, we do not care if there is a structure as below where the $boxed{}$ component is in the middle of an ouptut, 
WE ARE GOING TO ASSUME that all the tokens used to structure the output just prior to $boxed{}$ component as a part of the REASONING

REMEMBER: this is a very specific implementation in order tog et these samples without TOO MUCH work. 

"
Bleach is a common name for a solution of sodium hypochlorite (NaOCl) in water. The molecular weight of sodium hypochlorite is calculated by adding the atomic weights of sodium (Na), oxygen (O), and chlorine (Cl). The atomic weights are approximately:

- Sodium (Na): 22.99 g/mol
- Oxygen (O): 16.00 g/mol
- Chlorine (Cl): 35.45 g/mol

The molecular weight of sodium hypochlorite (NaOCl) is:

NaOCl = Na + O + Cl
= 22.99 g/mol + 16.00 g/mol + 35.45 g/mol
= 74.44 g/mol

So, the molecular weight of sodium hypochlorite, which is the active ingredient in bleach, is approximately 
$\boxed{74.44}$ g/mol. 
However, commercial bleach solutions contain a certain percentage of sodium hypochlorite dissolved in water, 
so the overall molecular weight of the solution would be different and depends on the concentration of the sodium hypochlorite in the solution.
"
"""

def load_and_process_numina_data(tokenizer, output_dir="/tmp/numina_proc/", file_name="numina_math.json"):
    dataset_name = 'AI-MO/NuminaMath-CoT'
    all_samples = {}
    # ========================================================================
    # DATASET 1: AI-MO/NuminaMath-CoT (Competition Math - 860K samples)
    # Domain: High school to olympiad-level math
    # Source: Chinese exams, AIME, Olympiads
    # ========================================================================
    try:
        print("\n1. Loading AI-MO/NuminaMath-CoT (Competition math)...")
        numina_train = load_dataset(dataset_name, split="train")  # Use subset
        print("Processing Numina: Training Set")
        numina_train_samples = {}
        for i, item in enumerate(numina_train):
            sample = process_numina_item(item, tokenizer, i, 'train')
            numina_train_samples[f"{i}"] = sample           

        # Save to file
        os.makedirs(output_dir + 'train/', exist_ok=True)
        output_path = os.path.join(output_dir, 'train', file_name)
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(numina_train_samples, fout, ensure_ascii=False, indent=2)
        
        print(f"   ✓ Loaded {len(numina_train_samples)} Training samples from NuminaMathCOT")
        all_samples['train'] = numina_train_samples
        
        print("Processing Numina: Test Set")
        numina_test = load_dataset(dataset_name, split="test")
        numina_test_samples = {}
        for i, item in enumerate(numina_test):
            sample = process_numina_item(item, tokenizer, i, 'test')
            numina_test_samples[f"{i}"] = sample 
            
        # Save to file
        os.makedirs(output_dir + 'test/', exist_ok=True)
        output_path = os.path.join(output_dir, 'test', file_name)
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(numina_test_samples, fout, ensure_ascii=False, indent=2)
        
        print(f"   ✓ Loaded {len(numina_test_samples)} Test samples from NuminaMathCOT")
        all_samples['test'] = numina_test_samples
            
    except Exception as e:
        print(f"   ⚠ Could not load NuminaMath: {e}")

    return all_samples

def process_numina_item(item, tokenizer, row_num, data_split):
    query = item['problem']
    output_text = item['solution']
    sample = {}
    
    # Extract reasoning (everything before the first "\boxed{...}") and answer (inside the first \boxed{...})
    cot_text = ""
    answer = ""
    box_match = re.search(r"\$\\boxed\{(.*?)\}\$", output_text)
    if box_match:
        # Reasoning: everything before the box (strip whitespace)
        box_start_idx = box_match.start()
        cot_text = output_text[:box_start_idx].strip()
        # The answer is the content inside the box
        answer = box_match.group(1).strip()
    else:
        # If no box found, treat entire output as reasoning and answer as empty
        cot_text = output_text.strip()
        answer = ""
    
    if cot_text.strip() and query.strip() and len(cot_text) > 50:
        # Tokenize COT for budget calculation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        cot_token_count = len(tokenizer(cot_text, truncation=False, add_special_tokens=False)["input_ids"])
        
        sample = {
            "idx": row_num,
            "query": query,
            "cot_reasoning": cot_text,
            "answer": answer, 
            "cot_token_budget": cot_token_count,
            "dataset_split": data_split
        }
        
        # all_samples[f"{sample_count}"] = sample
        # sample_count += 1
        print_split = 50000 if data_split == 'train' else 10
        if row_num % print_split == 0:
            print(f"   ✓ Processed {row_num} samples from NuminaMath")
    
    return sample


# # -> Dict[str, BudgetDataset]
# def load_and_process_benchmark_data(dataset: str):
#     """
#     Load and prepare data from reasoning benchmarks with actual CoT traces

#     Datasets:
#     1. AI-MO/NuminaMath-CoT - Competition math with CoT (860K samples)
#     2. meta-math/MetaMathQA - Augmented math reasoning (subset: 50K)
#     3. bespokelabs/Bespoke-Stratos-17k - Mixed reasoning: code, math, science (17K)

#     Returns:
        
#     """
#     print("Loading benchmark datasets with CoT traces from HuggingFace...")

#     # Initialize tokenizer for counting tokens in CoT
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     queries, budgets, cot_traces = [], [], []

#     # ========================================================================
#     # DATASET 1: AI-MO/NuminaMath-CoT (Competition Math - 860K samples)
#     # Domain: High school to olympiad-level math
#     # Source: Chinese exams, AIME, Olympiads
#     # ========================================================================
#     try:
#         print("\n1. Loading AI-MO/NuminaMath-CoT (Competition math)...")
#         numina_train = load_dataset("AI-MO/NuminaMath-CoT", split="train[:100000]")  # Use subset
#         numina_test = load_dataset("AI-MO/NuminaMath-CoT", split="test")

#         start_idx = len(queries)

#         for dataset_name, dataset in [("train", numina_train), ("test", numina_test)]:
#             print(f"Processing Numina: {dataset_name}")
#             for item in dataset:
#                 query = item['problem']
#                 cot_text = item['solution']

#                 if cot_text.strip() and query.strip() and len(cot_text) > 50:
#                     queries.append(query)
#                     cot_traces.append(cot_text)

#         print(f"   ✓ Loaded {len(queries) - start_idx} samples from NuminaMath")
#     except Exception as e:
#         print(f"   ⚠ Could not load NuminaMath: {e}")

#     # ========================================================================
#     # DATASET 2: MetaMathQA (Augmented Math - 50K subset)
#     # Domain: Synthetic math reasoning with augmentation
#     # TAKE THE WHOLE DATASET
#     # ========================================================================
#     try:
#         print("\n2. Loading MetaMathQA (Augmented math reasoning)...")
#         metamath = load_dataset("meta-math/MetaMathQA", split="train[:50000]")

#         start_idx = len(queries)
#         print("Processing MetaMathQA")
#         for item in tqdm(metamath, desc="Processing MetaMathQA"):
#             query = item['query']
#             cot_text = item['response']

#             if cot_text.strip() and query.strip() and len(cot_text) > 50:
#                 queries.append(query)
#                 cot_traces.append(cot_text)

#         print(f"   ✓ Loaded {len(queries) - start_idx} samples from MetaMathQA")
#     except Exception as e:
#         print(f"   ⚠ Could not load MetaMathQA: {e}")

#     # ========================================================================
#     # DATASET 3: Bespoke-Stratos-17K (Mixed Reasoning: Code + Math + Science)
#     # Domain: Code (APPS/TACO), Hard Math (AIME/Olympiads), Science/Puzzles
#     # ========================================================================
#     try:
#         print("\n3. Loading Bespoke-Stratos-17k (Code, math, science reasoning)...")
#         bespoke = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")

#         start_idx = len(queries)

#         for item in tqdm(bespoke, desc="Processing Bespoke-Stratos"):
#             # Extract question and reasoning trace
#             if 'messages' in item and len(item['messages']) >= 2:
#                 # Format: [{'role': 'user', 'content': query}, {'role': 'assistant', 'content': response}]
#                 messages = item['messages']
#                 query = None
#                 cot_text = None

#                 for msg in messages:
#                     if msg['role'] == 'user':
#                         query = msg['content']
#                     elif msg['role'] == 'assistant':
#                         cot_text = msg['content']

#                 if query and cot_text and query.strip() and cot_text.strip():
#                     queries.append(query)
#                     cot_traces.append(cot_text)

#         print(f"   ✓ Loaded {len(queries) - start_idx} samples from Bespoke-Stratos")
#     except Exception as e:
#         print(f"   ⚠ Could not load Bespoke-Stratos: {e}")

#     # ========================================================================
#     # Compute actual token budgets from CoT traces
#     # ========================================================================
#     print(f"\n5. Computing token budgets from {len(queries)} CoT traces...")

#     if len(queries) == 0:
#         raise ValueError(
#             "No datasets loaded successfully! Please check your internet connection "
#             "and HuggingFace access. You may need to: pip install datasets --upgrade"
#         )

#     for i in tqdm(range(len(queries)), desc="Tokenizing CoT traces"):
#         cot_tokens = tokenizer(
#             cot_traces[i],
#             truncation=False,
#             add_special_tokens=False
#         )
#         budget = len(cot_tokens['input_ids'])
#         budgets.append(budget)

#     budgets_array = np.array(budgets)

#     print(f"\nBudget statistics:")
#     print(f"  Total samples: {len(budgets)}")
#     print(f"  Min: {budgets_array.min()} tokens")
#     print(f"  Max: {budgets_array.max()} tokens")
#     print(f"  Mean: {budgets_array.mean():.1f} tokens")
#     print(f"  Median: {np.median(budgets_array):.1f} tokens")
#     print(f"  Std: {budgets_array.std():.1f} tokens")
#     print(f"  P25: {np.percentile(budgets_array, 25):.1f} tokens")
#     print(f"  P75: {np.percentile(budgets_array, 75):.1f} tokens")
#     print(f"  P90: {np.percentile(budgets_array, 90):.1f} tokens")

#     # ========================================================================
#     # Auto-adjust expert ranges based on data distribution
#     # ========================================================================
#     print("\n6. Auto-adjusting expert ranges based on data distribution...")

#     config.expert_ranges = auto_adjust_expert_ranges(budgets, config.n_experts)

#     print("\nAdjusted expert ranges (based on quantiles):")
#     regime_names = ['Trivial', 'Easy', 'Medium', 'Hard'][:config.n_experts]
#     for i, (min_b, max_b) in enumerate(config.expert_ranges):
#         print(f"  Expert {i} ({regime_names[i]:>8}): {min_b:>5} - {max_b:>5} tokens")

#     # ========================================================================
#     # Assign difficulty labels
#     # ========================================================================
#     print("\n7. Assigning difficulty labels...")

#     difficulties = []
#     for budget in budgets:
#         assigned = False
#         for expert_id, (min_b, max_b) in enumerate(config.expert_ranges):
#             if min_b <= budget < max_b:
#                 difficulties.append(expert_id)
#                 assigned = True
#                 break
#         if not assigned:
#             difficulties.append(config.n_experts - 1)

#     # ========================================================================
#     # Split into train/val/test
#     # ========================================================================
#     print("\n8. Creating train/val/test splits...")

#     total_size = len(queries)

#     # Alternative: Use numpy's default random generator for better forward compatibility
#     # np.random.default_rng is recommended in numpy documentation over RandomState.
#     # https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng
#     indices = np.random.default_rng(42).permutation(total_size)

#     train_size = int(0.8 * total_size)
#     val_size = int(0.1 * total_size)

#     train_idx = indices[:train_size]
#     val_idx = indices[train_size:train_size + val_size]
#     test_idx = indices[train_size + val_size:]

#     datasets = {}
#     for split_name, idx_list in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
#         split_queries = [queries[i] for i in idx_list]
#         split_budgets = [budgets[i] for i in idx_list]
#         split_difficulties = [difficulties[i] for i in idx_list]

#         datasets[split_name] = BudgetDataset(
#             queries=split_queries,
#             actual_budgets=split_budgets,
#             difficulty_labels=split_difficulties,
#             tokenizer=tokenizer,
#             config=config,
#             split=split_name
#         )

#     print("\nDataset sizes:")
#     print(f"  Train: {len(datasets['train'])} samples")
#     print(f"  Val: {len(datasets['val'])} samples")
#     print(f"  Test: {len(datasets['test'])} samples")

#     np.savez(
#         cache_file,
#         queries=queries,
#         budgets=budgets,
#         difficulties=difficulties,
#         train_idx=train_idx,
#         val_idx=val_idx,
#         test_idx=test_idx,
#         expert_ranges=config.expert_ranges
#     )

#     print("✓ Data loading complete!")

#     return datasets, tokenizer
