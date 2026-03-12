import json
import time
import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from llm_utils import QwenLLM, LlamaLLM, GeminiLLM, ChatGPTLLM, DeepSeekLLM, GPT51LLM, Colors

def build_belief_attribution_prompt(vqa, demographics, context_qas, include_demographics, include_context, max_context_qa=None):
    """Build prompt for belief attribution task"""
    # Build system message for Theory of Mind evaluation
    system_parts = [
        "You are an expert psychologist specializing in Theory of Mind and belief attribution.",
        "Your task: analyze conversation transcripts to infer what the participant believes about causal relationships.",
        "Focus on understanding their mental model - what they think causes what, not what is objectively true.",
        "Consider their background, conversation patterns, and implicit beliefs expressed through their responses.",
        "Base your inference strictly on evidence from their statements, not general assumptions."
    ]
    
    # Build user prompt components
    prompt_parts = []
    
    # Add demographics if requested
    if include_demographics and demographics:
        demo_text = "Person's Background:\n"
        for key, value in demographics.items():
            demo_text += f"- {key.replace('_', ' ').title()}: {value}\n"
        prompt_parts.append(demo_text.strip())
    
    # Add context conversations if requested
    if include_context and context_qas:
        # Limit context QAs if max_context_qa is specified
        limited_context_qas = context_qas[:max_context_qa] if max_context_qa else context_qas
        context_text = "Conversation History:\n"
        for i, qa in enumerate(limited_context_qas, 1):
            context_text += f"Q{i}: {qa['question']}\n"
            context_text += f"A{i}: {qa['answer']}\n\n"
        prompt_parts.append(context_text.strip())
    
    # Add the main task question
    task_question = vqa["task_question"]
    prompt_parts.append(f"Task: {task_question}")
    
    # Dynamically generate answer options from the dict
    answer_options = vqa["answer_options"]
    options_text = "Answer options:\n"
    option_keys = list(answer_options.keys())
    for key in option_keys:
        options_text += f"{key}) {answer_options[key]}\n"
    prompt_parts.append(options_text.strip())
    
    # Add clear instruction - only answer needed
    options_str = ", ".join(option_keys)
    prompt_parts.append(f"Answer with ONLY the letter ({options_str}). No explanation needed.")
    
    return " ".join(system_parts), "\n\n".join(prompt_parts)

def build_belief_update_prompt(vqa, demographics, context_qas, include_demographics, include_context, max_context_qa=None):
    """Build prompt for belief update task"""
    # Build system message for belief update evaluation
    system_parts = [
        "You are an expert in survey research and human psychology.",
        "Your task: predict how this person would respond to a specific survey question based on their background and conversation.",
        "Consider their demographics, expressed opinions, and conversation patterns.",
        "Focus on understanding their likely response pattern, not what would be objectively correct.",
        "Base your prediction on evidence from their profile and statements."
    ]
    
    # Build user prompt components
    prompt_parts = []
    
    # Add demographics if requested
    if include_demographics and demographics:
        demo_text = "Person's Background:\n"
        for key, value in demographics.items():
            demo_text += f"- {key.replace('_', ' ').title()}: {value}\n"
        prompt_parts.append(demo_text.strip())
    
    # Add context conversations if requested
    if include_context and context_qas:
        # Limit context QAs if max_context_qa is specified
        limited_context_qas = context_qas[:max_context_qa] if max_context_qa else context_qas
        context_text = "Previous Conversation:\n"
        for i, qa in enumerate(limited_context_qas, 1):
            context_text += f"Q{i}: {qa['question']}\n"
            context_text += f"A{i}: {qa['answer']}\n\n"
        prompt_parts.append(context_text.strip())
    
    # Add the survey question
    task_question = vqa["task_question"]
    question_type = vqa.get("question_type", "unknown")
    
    if question_type == "opinion":
        prompt_parts.append(f"Survey Question: {task_question}")
        scale = vqa.get("scale", [1, 10])
        prompt_parts.append(f"Scale: {scale[0]} to {scale[1]}")
        prompt_parts.append(f"Based on this person's profile and conversation, what number would they likely choose? Respond with ONLY the number.")
    
    elif question_type == "reason_evaluation":
        reason_text = vqa.get("reason_text", "")
        prompt_parts.append(f"Survey Question: {task_question}")
        prompt_parts.append(f"Context: This asks about the influence of: {reason_text}")
        scale = vqa.get("scale", [1, 5])
        prompt_parts.append(f"Scale: {scale[0]} to {scale[1]} (1=no influence, {scale[1]}=very strong influence)")
        prompt_parts.append(f"Based on this person's profile and conversation, what rating would they likely give? Respond with ONLY the number.")
    
    return " ".join(system_parts), "\n\n".join(prompt_parts)

def process_single_question(llm_or_model, vqa, include_demographics, include_context, temperature, debug=False, swap_data=None, max_retries=30, max_context_qa=None):
    """Process a single question and return the result"""
    try:
        # Create a new LLM instance for each thread if model name is passed
        if isinstance(llm_or_model, str):
            model = llm_or_model
            if model.startswith("meta-llama"):
                llm = LlamaLLM(model=model)
            elif model.startswith("gemini"):
                llm = GeminiLLM(model=model)
            elif model == "gpt-5.1":
                llm = GPT51LLM(model=model)
            elif model.startswith(("gpt-", "o1", "o3")):
                llm = ChatGPTLLM(model=model)
            elif model.startswith("deepseek"):
                llm = DeepSeekLLM(model=model)
            else:
                llm = QwenLLM(model=model)
        else:
            llm = llm_or_model
        # Use swapped data if swap experiment is enabled
        if swap_data:
            demographics = swap_data.get("demographics", {})
            context_qas = swap_data.get("context_qas", [])
        else:
            demographics = vqa.get("demographics", {})
            context_qas = vqa.get("context_qas", [])
        
        # Determine task type and build appropriate prompt
        task_type = vqa.get("task_type", "belief_attribution")
        
        if task_type == "belief_attribution":
            system_message, user_prompt = build_belief_attribution_prompt(
                vqa, demographics, context_qas, include_demographics, include_context, max_context_qa
            )
            
            # Debug: Display constructed prompt
            if debug:
                print(f"\n{Colors.format('='*60, Colors.BOLD)}")
                print(f"{Colors.format('CONSTRUCTED PROMPT', Colors.BOLD + Colors.CYAN)}")
                print(f"{Colors.format('='*60, Colors.BOLD)}")
                print(f"{Colors.format('System Message:', Colors.BOLD + Colors.YELLOW)}")
                print(f"{system_message}\n")
                print(f"{Colors.format('User Prompt:', Colors.BOLD + Colors.YELLOW)}")
                print(f"{user_prompt}\n")
                print(f"{Colors.format('Answer Options:', Colors.BOLD + Colors.YELLOW)}")
                for key, value in vqa["answer_options"].items():
                    print(f"  {key}) {value}")
                if max_context_qa:
                    print(f"{Colors.format('Max Context QAs:', Colors.BOLD + Colors.YELLOW)} {max_context_qa}")
                print(f"{Colors.format('Temperature:', Colors.BOLD + Colors.YELLOW)} {temperature}")
                print(f"{Colors.format('Model:', Colors.BOLD + Colors.YELLOW)} {llm.__class__.__name__}")
                print(f"\n{Colors.format('Press Enter to send request...', Colors.BOLD + Colors.GREEN)}")
                input()
            
            # Use function calling approach
            answer_options = vqa["answer_options"]
            fallback_result = llm.generate_with_fallback(
                user_prompt,
                system_message=system_message,
                temperature=temperature,
                debug=debug,
                max_retries=max_retries,
                task_type="belief_attribution",
                answer_options=answer_options
            )
            
            generated_response = fallback_result["raw_response"]
            generated_answer = fallback_result["extracted_answer"]
            method_used = fallback_result["method_used"]
            
            # Debug: Display response and parsing results
            if debug:
                print(f"\n{Colors.format('='*60, Colors.BOLD)}")
                print(f"{Colors.format('RESPONSE & ANSWER PARSING', Colors.BOLD + Colors.CYAN)}")
                print(f"{Colors.format('='*60, Colors.BOLD)}")
                print(f"{Colors.format('Raw Response:', Colors.BOLD + Colors.YELLOW)}")
                print(f"{generated_response}\n")
                print(f"{Colors.format('Extracted Answer:', Colors.BOLD + Colors.YELLOW)} {Colors.format(str(generated_answer), Colors.GREEN)}")
                print(f"{Colors.format('Method Used:', Colors.BOLD + Colors.YELLOW)} {Colors.format(method_used, Colors.BLUE)}")
                print(f"{Colors.format('Correct Answer:', Colors.BOLD + Colors.YELLOW)} {Colors.format(vqa['answer'], Colors.GREEN)}")
                is_correct_preview = (generated_answer == vqa['answer'])
                result_color = Colors.GREEN if is_correct_preview else Colors.RED
                print(f"{Colors.format('Result:', Colors.BOLD + Colors.YELLOW)} {Colors.format('✓ CORRECT' if is_correct_preview else '✗ INCORRECT', result_color)}")
                print(f"\n{Colors.format('Press Enter to continue to next question...', Colors.BOLD + Colors.GREEN)}")
                input()
            
            # Get the correct answer
            correct_answer = vqa['answer']
            
            # Evaluate correctness
            is_correct = (generated_answer == correct_answer)
            
        elif task_type == "belief_update":
            system_message, user_prompt = build_belief_update_prompt(
                vqa, demographics, context_qas, include_demographics, include_context, max_context_qa
            )
            
            # Debug: Display constructed prompt
            if debug:
                print(f"\n{Colors.format('='*60, Colors.BOLD)}")
                print(f"{Colors.format('CONSTRUCTED PROMPT', Colors.BOLD + Colors.CYAN)}")
                print(f"{Colors.format('='*60, Colors.BOLD)}")
                print(f"{Colors.format('System Message:', Colors.BOLD + Colors.YELLOW)}")
                print(f"{system_message}\n")
                print(f"{Colors.format('User Prompt:', Colors.BOLD + Colors.YELLOW)}")
                print(f"{user_prompt}\n")
                scale = vqa.get("scale", [1, 10])
                print(f"{Colors.format('Scale:', Colors.BOLD + Colors.YELLOW)} {scale[0]} to {scale[1]}")
                print(f"{Colors.format('Question Type:', Colors.BOLD + Colors.YELLOW)} {vqa.get('question_type', 'unknown')}")
                if 'reason_text' in vqa:
                    print(f"{Colors.format('Reason:', Colors.BOLD + Colors.YELLOW)} {vqa['reason_text']}")
                if max_context_qa:
                    print(f"{Colors.format('Max Context QAs:', Colors.BOLD + Colors.YELLOW)} {max_context_qa}")
                print(f"{Colors.format('Temperature:', Colors.BOLD + Colors.YELLOW)} {temperature}")
                print(f"{Colors.format('Model:', Colors.BOLD + Colors.YELLOW)} {llm.__class__.__name__}")
                print(f"\n{Colors.format('Press Enter to send request...', Colors.BOLD + Colors.GREEN)}")
                input()
            
            # Use function calling approach
            scale = vqa.get("scale", [1, 10])
            fallback_result = llm.generate_with_fallback(
                user_prompt,
                system_message=system_message,
                temperature=temperature,
                debug=debug,
                max_retries=max_retries,
                task_type="belief_update",
                scale=scale
            )
            
            generated_response = fallback_result["raw_response"]
            generated_answer = fallback_result["extracted_answer"]
            method_used = fallback_result["method_used"]
            
            # Debug: Display response and parsing results
            if debug:
                print(f"\n{Colors.format('='*60, Colors.BOLD)}")
                print(f"{Colors.format('RESPONSE & ANSWER PARSING', Colors.BOLD + Colors.CYAN)}")
                print(f"{Colors.format('='*60, Colors.BOLD)}")
                print(f"{Colors.format('Raw Response:', Colors.BOLD + Colors.YELLOW)}")
                print(f"{generated_response}\n")
                print(f"{Colors.format('Extracted Answer:', Colors.BOLD + Colors.YELLOW)} {Colors.format(str(generated_answer), Colors.GREEN)}")
                print(f"{Colors.format('Method Used:', Colors.BOLD + Colors.YELLOW)} {Colors.format(method_used, Colors.BLUE)}")
                print(f"{Colors.format('Correct Answer:', Colors.BOLD + Colors.YELLOW)} {Colors.format(str(vqa['user_answer']), Colors.GREEN)}")
                
                # Calculate correctness preview
                is_correct_preview = False
                if generated_answer is not None and vqa['user_answer'] is not None:
                    scale_range = scale[1] - scale[0]
                    tolerance = 1 if scale_range <= 5 else 2
                    is_correct_preview = abs(generated_answer - vqa['user_answer']) <= tolerance
                
                result_color = Colors.GREEN if is_correct_preview else Colors.RED
                print(f"{Colors.format('Result:', Colors.BOLD + Colors.YELLOW)} {Colors.format('✓ CORRECT' if is_correct_preview else '✗ INCORRECT', result_color)}")
                if generated_answer is not None and vqa['user_answer'] is not None:
                    diff = abs(generated_answer - vqa['user_answer'])
                    print(f"{Colors.format('Difference:', Colors.BOLD + Colors.YELLOW)} {diff} (tolerance: {tolerance})")
                print(f"\n{Colors.format('Press Enter to continue to next question...', Colors.BOLD + Colors.GREEN)}")
                input()
            
            # Get the correct answer
            correct_answer = vqa['user_answer']
            
            # For belief_update, we evaluate based on exact match or close proximity
            is_correct = False
            if generated_answer is not None and correct_answer is not None:
                # Exact match or within 1 point for ordinal scales
                scale_range = scale[1] - scale[0]
                tolerance = 1 if scale_range <= 5 else 2  # Stricter tolerance for smaller scales
                is_correct = abs(generated_answer - correct_answer) <= tolerance
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return {
            'vqa': vqa,
            'generated_answer': generated_answer,
            'generated_response': generated_response,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'context_qas_count': len(context_qas),
            'task_type': task_type,
            'method_used': method_used
        }
        
    except Exception as e:
        return {
            'vqa': vqa,
            'generated_answer': None,
            'generated_response': None,
            'correct_answer': vqa.get('answer') or vqa.get('user_answer', ''),
            'is_correct': False,
            'context_qas_count': len(vqa.get('context_qas', [])),
            'task_type': vqa.get("task_type", "unknown"),
            'method_used': 'error',
            'error': str(e),
            'original_error': repr(e)
        }

def create_swap_mapping(dataset):
    """Create mapping to swap data between consecutive prolific IDs"""
    prolific_ids = []
    id_to_data = {}
    
    for vqa in dataset:
        pid = vqa.get("prolific_id", "")
        if pid and pid not in id_to_data:
            prolific_ids.append(pid)
            id_to_data[pid] = {
                "demographics": vqa.get("demographics", {}),
                "context_qas": vqa.get("context_qas", [])
            }
    
    # Create swap mapping: each ID maps to next ID's data
    swap_mapping = {}
    for i, pid in enumerate(prolific_ids):
        next_pid = prolific_ids[(i + 1) % len(prolific_ids)]
        swap_mapping[pid] = id_to_data[next_pid]
    
    return swap_mapping

def evaluate_belief_inference(benchmark_path, model="qwen-plus", temperature=0, include_demographics=True, include_context=True, max_workers=20, debug=False, swap_experiment=False, max_context_qa=None):
    """
    Evaluate Theory of Mind belief inference questions using Qwen model
    """
    # Initialize appropriate LLM based on model type
    if model.startswith("meta-llama"):
        llm = LlamaLLM(model=model)
    elif model.startswith("gemini"):
        llm = GeminiLLM(model=model)
    elif model == "gpt-5.1":
        llm = GPT51LLM(model=model)
    elif model.startswith(("gpt-", "o1", "o3")):
        llm = ChatGPTLLM(model=model)
    elif model.startswith("deepseek"):
        llm = DeepSeekLLM(model=model)
    else:
        llm = QwenLLM(model=model)
    
    # Load benchmark data
    vqa_dataset = []
    with open(benchmark_path, "r") as file:
        content = file.read().strip()
        
        # Handle formatted JSON objects separated by newlines
        json_objects = []
        current_object = ""
        brace_count = 0
        
        for line in content.split('\n'):
            current_object += line + '\n'
            brace_count += line.count('{') - line.count('}')
            
            if brace_count == 0 and current_object.strip():
                try:
                    json_obj = json.loads(current_object.strip())
                    json_objects.append(json_obj)
                    current_object = ""
                except json.JSONDecodeError:
                    continue
        
        vqa_dataset = json_objects
    
    # Separate data by task type and difficulty
    task_types = set()
    difficulty_datasets = {"simple": [], "medium": [], "hard": [], "all": []}
    
    for vqa in vqa_dataset:
        task_type = vqa.get("task_type", "belief_attribution")
        task_types.add(task_type)
        
        # For belief_attribution, use context_length as difficulty
        if task_type == "belief_attribution":
            difficulty = vqa.get("context_length", "unknown")
            if difficulty not in difficulty_datasets:
                difficulty_datasets[difficulty] = []
            difficulty_datasets[difficulty].append(vqa)
        
        # For belief_update, group all together since difficulty doesn't apply the same way
        elif task_type == "belief_update":
            difficulty_datasets["all"].append(vqa)
    
    # Remove empty difficulty levels
    difficulty_datasets = {k: v for k, v in difficulty_datasets.items() if v}
    
    # Create swap mapping if swap experiment is enabled
    swap_mapping = create_swap_mapping(vqa_dataset) if swap_experiment else None
    
    # Results for each difficulty
    all_results = {}
    all_detailed_results = []  # Collect all detailed results for logging
    total_errors = 0
    
    print(f"Evaluating {len(vqa_dataset)} questions across {len(difficulty_datasets)} groups...")
    print(f"Task types found: {', '.join(task_types)}")
    
    for difficulty, dataset in difficulty_datasets.items():
        if not dataset:
            continue
            
        print(f"\n{'='*60}")
        if difficulty == "all":
            # Get task type for this group
            sample_task_type = dataset[0].get("task_type", "unknown")
            print(f"EVALUATING TASK TYPE: {sample_task_type.upper()}")
        else:
            print(f"EVALUATING DIFFICULTY LEVEL: {difficulty.upper()}")
        print(f"{'='*60}")
        print(f"Questions in this group: {len(dataset)}")
        
        correct = 0
        total = 0
        answers = []
        absolute_errors = []  # For MAE calculation
        signed_errors = []  # For MSE (Mean Signed Error) calculation
        
        # Process questions (parallel or sequential based on debug mode)
        if debug:
            # Sequential processing for debug mode - use LLM instance for debug mode
            results = []
            for i, vqa in enumerate(dataset):
                print(f"\n{Colors.format('[DEBUG]', Colors.BOLD + Colors.YELLOW)} Processing question {Colors.format(str(i+1), Colors.CYAN)}/{Colors.format(str(len(dataset)), Colors.CYAN)} in {Colors.format(difficulty, Colors.GREEN)} difficulty")
                
                # Get swap data if experiment is enabled
                swap_data = None
                if swap_experiment and swap_mapping:
                    pid = vqa.get("prolific_id", "")
                    swap_data = swap_mapping.get(pid)
                
                result = process_single_question(llm, vqa, include_demographics, include_context, temperature, debug, swap_data, max_retries=30, max_context_qa=max_context_qa)
                results.append(result)
        else:
            # Parallel processing for normal mode - pass model name to create separate instances
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all questions
                future_to_index = {}
                for i, vqa in enumerate(dataset):
                    # Get swap data if experiment is enabled
                    swap_data = None
                    if swap_experiment and swap_mapping:
                        pid = vqa.get("prolific_id", "")
                        swap_data = swap_mapping.get(pid)
                    
                    future = executor.submit(process_single_question, model, vqa, include_demographics, include_context, temperature, debug, swap_data, max_retries=30, max_context_qa=max_context_qa)
                    future_to_index[future] = i
            
                # Process completed results
                results = [None] * len(dataset)
                
                for future in tqdm(as_completed(future_to_index), total=len(dataset), desc=f"Processing {difficulty}"):
                    index = future_to_index[future]
                    result = future.result()
                    results[index] = result
                    
                    # Monitor for errors
                    if result.get('method_used') == 'error' or result.get('generated_answer') is None:
                        total_errors += 1
                    
                    # Add small delay between API calls to respect rate limits
                    time.sleep(0.5)
        
        # Process results in order
        for i, result in enumerate(results):
            if result is None:
                continue
                
            vqa = result['vqa']
            generated_answer = result['generated_answer']
            generated_response = result['generated_response']
            correct_answer = result['correct_answer']
            is_correct = result['is_correct']
            context_qas_count = result['context_qas_count']
            
            if is_correct:
                correct += 1
                answers.append(1)
            else:
                answers.append(0)
            
            # Calculate absolute error and signed error for numeric tasks
            if vqa.get('task_type') == 'belief_update' and generated_answer is not None and correct_answer is not None:
                abs_error = abs(generated_answer - correct_answer)
                signed_error = generated_answer - correct_answer  # e = y_hat - y
                absolute_errors.append(abs_error)
                signed_errors.append(signed_error)
            
            total += 1
                
            # Print details for verbose mode (only if not in debug mode)
            if not debug:
                group_label = difficulty.upper() if difficulty != "all" else vqa.get("task_type", "unknown").upper()
                print(f"\n[{group_label}] Question {total}:")
                print(f"Task Type: {vqa.get('task_type', 'unknown')}")
                print(f"Context QAs: {context_qas_count}")
                print(f"Task: {vqa['task_question']}")
                print(f"Generated: {generated_answer} (Response: {generated_response})")
                print(f"Correct: {correct_answer}")
                print(f"Method Used: {result.get('method_used', 'unknown')}")
                print(f"Result: {'✓' if is_correct else '✗'}")
                
                # Show different fields based on task type
                if vqa.get('task_type') == 'belief_attribution':
                    print(f"Source QA: {vqa.get('source_qa', {}).get('question', 'N/A')}")
                elif vqa.get('task_type') == 'belief_update':
                    print(f"Question Type: {vqa.get('question_type', 'N/A')}")
                    if 'reason_text' in vqa:
                        print(f"Reason: {vqa['reason_text']}")
                
                if 'error' in result:
                    print(f"Error: {result['error']}")
                print("-" * 50)
        
        # Store detailed results for each question
        for i, result in enumerate(results):
            if result is None:
                continue
                
            vqa = result['vqa']
            generated_answer = result['generated_answer']
            generated_response = result['generated_response']
            correct_answer = result['correct_answer']
            is_correct = result['is_correct']
            
            # Store detailed result with VQA information (excluding demographics and context_qas)
            # Use id+prolific_id for unique anchoring
            question_anchor = f"{vqa.get('id', f'{difficulty}_{i+1}')}_{vqa.get('prolific_id', 'unknown')}"
            vqa_filtered = {k: v for k, v in vqa.items() if k not in ['demographics', 'context_qas']}
            detailed_result = {
                'question_anchor': question_anchor,
                'question_id': vqa.get('id', f"{difficulty}_{i+1}"),
                'prolific_id': vqa.get('prolific_id', 'unknown'),
                'vqa': vqa_filtered,
                'generated_answer': generated_answer,
                'generated_response': generated_response,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'method_used': result.get('method_used', 'unknown')
            }
            all_detailed_results.append(detailed_result)
        
        # Calculate and store results for this difficulty
        accuracy = correct / total if total > 0 else 0
        mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
        mse = sum(signed_errors) / len(signed_errors) if signed_errors else None
        
        all_results[difficulty] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'answers': answers,
            'mae': mae,
            'mse': mse,
            'absolute_errors': absolute_errors,
            'signed_errors': signed_errors
        }
        
        print(f"\n{'='*50}")
        if difficulty == "all":
            sample_task_type = dataset[0].get("task_type", "unknown")
            print(f"RESULTS FOR {sample_task_type.upper()} TASK")
        else:
            print(f"RESULTS FOR {difficulty.upper()} DIFFICULTY")
        print(f"{'='*50}")
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        if mae is not None:
            print(f"MAE (Mean Absolute Error): {mae:.3f}")
        if mse is not None:
            print(f"MSE (Mean Signed Error): {mse:+.3f}")
        print(f"Context QAs per question: {len(dataset[0]['context_qas']) if dataset else 0}")
    
    # Display summary results
    print(f"\n{'='*60}")
    print(f"SUMMARY RESULTS ACROSS ALL DIFFICULTIES")
    print(f"{'='*60}")
    
    total_all = sum(result['total'] for result in all_results.values())
    correct_all = sum(result['correct'] for result in all_results.values())
    accuracy_all = correct_all / total_all if total_all > 0 else 0
    
    # Calculate overall MAE and MSE
    all_absolute_errors = []
    all_signed_errors = []
    
    for result in all_results.values():
        if result['absolute_errors']:
            all_absolute_errors.extend(result['absolute_errors'])
        if result.get('signed_errors'):
            all_signed_errors.extend(result['signed_errors'])
    
    overall_mae = sum(all_absolute_errors) / len(all_absolute_errors) if all_absolute_errors else None
    overall_mse = sum(all_signed_errors) / len(all_signed_errors) if all_signed_errors else None
    
    print(f"Overall accuracy: {accuracy_all:.2%} ({correct_all}/{total_all})")
    if overall_mae is not None:
        print(f"Overall MAE: {overall_mae:.3f}")
    if overall_mse is not None:
        print(f"Overall MSE: {overall_mse:+.3f}")
    print(f"\nBreakdown by group:")
    for group_name in all_results:
        result = all_results[group_name]
        if group_name == "all":
            task_type = difficulty_datasets[group_name][0].get("task_type", "unknown")
            group_label = f"{task_type}"
        else:
            group_label = group_name
        
        context_size = len(difficulty_datasets[group_name][0]['context_qas']) if difficulty_datasets[group_name] else 0
        mae_str = f", MAE: {result['mae']:.3f}" if result['mae'] is not None else ""
        mse_str = f", MSE: {result['mse']:+.3f}" if result.get('mse') is not None else ""
        print(f"  {group_label.capitalize():<12} ({context_size:2d} context): {result['accuracy']:.2%} ({result['correct']}/{result['total']}){mae_str}{mse_str}")
    
    # Execution summary and error logging
    invalid_responses = sum(1 for result in all_detailed_results if result.get('generated_answer') is None)
    total_responses = len(all_detailed_results)
    
    print(f"\n{'='*60}")
    print(f"EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Benchmark file: {benchmark_path}")
    print(f"Model: {model}")
    print(f"Total errors encountered: {total_errors}")
    print(f"Invalid responses: {invalid_responses}/{total_responses}")
    
    if invalid_responses > 0:
        print(f"[WARNING] {invalid_responses} invalid responses detected!")
        
        # Log error details
        error_details = []
        for i, result in enumerate(all_detailed_results):
            if result.get('generated_answer') is None:
                error_details.append({
                    'question_index': i,
                    'question_anchor': result.get('question_anchor', 'unknown'),
                    'method_used': result.get('method_used', 'unknown')
                })
        
        error_log = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'benchmark_file': benchmark_path,
            'model': model,
            'total_questions': total_responses,
            'invalid_responses': invalid_responses,
            'total_errors': total_errors,
            'error_details': error_details
        }
        
        clean_model_name = model.replace("/", "-")
        benchmark_name = os.path.splitext(os.path.basename(benchmark_path))[0]
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        error_filename = f"error_log_{clean_model_name}-{benchmark_name}-{timestamp}.json"
        with open(error_filename, 'w') as f:
            json.dump(error_log, f, indent=2)
        print(f"Detailed error log saved to: {error_filename}")
    
    # Check for high invalid response rate (>5%)
    error_rate = invalid_responses / total_responses if total_responses > 0 else 0
    if error_rate > 0.05:
        print(f"\n{'='*60}")
        print(f"HIGH INVALID RESPONSE RATE DETECTED")
        print(f"{'='*60}")
        print(f"Error rate: {error_rate:.1%} ({invalid_responses}/{total_responses})")
        print(f"Consider rerunning this benchmark")
        
        # Save high error rate log
        error_log = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'benchmark_path': benchmark_path,
            'model': model,
            'error_rate': error_rate,
            'invalid_responses': invalid_responses,
            'total_responses': total_responses,
            'reason': 'high_invalid_response_rate'
        }
        
        clean_model_name = model.replace("/", "-")
        benchmark_name = os.path.splitext(os.path.basename(benchmark_path))[0]
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        error_filename = f"high_error_rate_{clean_model_name}-{benchmark_name}-{timestamp}.json"
        error_filepath = os.path.join(os.getcwd(), error_filename)
        with open(error_filepath, "w") as f:
            json.dump(error_log, f, indent=2)
        print(f"High error rate log saved to {error_filepath}")
        
        return all_results, all_detailed_results, True  # Return flag indicating high error rate
    
    return all_results, all_detailed_results, False

def main():
    parser = argparse.ArgumentParser(description="Evaluate Theory of Mind belief inference benchmark with Qwen")
    parser.add_argument("--benchmark_path", type=str, default="sample_prompt_v3.jsonl", 
                       help="Path to the benchmark JSONL file")
    parser.add_argument("--model", type=str, default="qwen-plus", 
                       choices=[
                           "qwen-max", 
                           "qwen-plus", 
                           "qwen-turbo",
                           "qwen2.5-72b-instruct",
                           "qwen2.5-32b-instruct", 
                           "qwen2.5-14b-instruct",
                           "qwen2.5-7b-instruct",
                           "qwen2.5-3b-instruct",
                           "qwen2.5-1.5b-instruct",
                           "qwen2.5-0.5b-instruct",
                           "meta-llama/llama-3.3-70b-instruct",
                           "gemini-1.5-pro",
                           "gemini-1.5-flash",
                           "gemini-2.5-flash-lite",
                           "gemini-2.0-flash",
                           "deepseek-r1-0528",
                           "gpt-4o",
                           "gpt-4o-mini",
                           "o1",
                           "o1-mini",
                           "o3-mini",
                           "gpt-5",
                           "gpt-5-mini",
                           "gpt-5.1"
                       ], 
                       help="Model to use for evaluation")
    parser.add_argument("--temperature", type=float, default=0.1, 
                       help="Temperature for generation")
    parser.add_argument("--no-demographics", action="store_true",
                       help="Exclude demographics from prompt")
    parser.add_argument("--no-context", action="store_true",
                       help="Exclude context QAs from prompt")
    parser.add_argument("--max-workers", type=int, default=20,
                       help="Maximum number of parallel workers for API calls")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode: sequential processing with full prompt/response display")
    parser.add_argument("--swap-experiment", action="store_true",
                       help="Use next participant's demographics/context for prediction (extreme test)")
    parser.add_argument("--max-context-qa", type=int, default=None,
                       help="Maximum number of context QAs to use (e.g., 5 for first 5 QAs)")
    parser.add_argument("--output-path", type=str, default=".",
                       help="Directory path for output files (default: current directory)")
    
    args = parser.parse_args()
    
    try:
        results, all_detailed_results, has_high_error_rate = evaluate_belief_inference(
            benchmark_path=args.benchmark_path,
            model=args.model,
            temperature=args.temperature,
            include_demographics=not args.no_demographics,
            include_context=not args.no_context,
            max_workers=args.max_workers,
            debug=args.debug,
            swap_experiment=args.swap_experiment,
            max_context_qa=args.max_context_qa
        )
        
        # Note high error rate but continue processing
        if has_high_error_rate:
            print("Note: High invalid response rate detected, consider rerunning")
        
        # Calculate summary statistics from results
        total_all = sum(result['total'] for result in results.values())
        correct_all = sum(result['correct'] for result in results.values())
        accuracy_all = correct_all / total_all if total_all > 0 else 0
        
        # Calculate overall MAE and MSE
        all_absolute_errors = []
        all_signed_errors = []
        
        for result in results.values():
            if result['absolute_errors']:
                all_absolute_errors.extend(result['absolute_errors'])
            if result.get('signed_errors'):
                all_signed_errors.extend(result['signed_errors'])
        
        overall_mae = sum(all_absolute_errors) / len(all_absolute_errors) if all_absolute_errors else None
        overall_mse = sum(all_signed_errors) / len(all_signed_errors) if all_signed_errors else None
        
        # Prepare enhanced results with detailed information
        enhanced_results = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model': args.model,
            'benchmark_path': args.benchmark_path,
            'temperature': args.temperature,
            'include_demographics': not args.no_demographics,
            'include_context': not args.no_context,
            'swap_experiment': args.swap_experiment,
            'max_context_qa': args.max_context_qa,
            'overall_metrics': {
                'total_questions': total_all,
                'correct_answers': correct_all,
                'overall_accuracy': accuracy_all,
                'overall_mae': overall_mae,
                'overall_mse': overall_mse
            },
            'group_results': results,
            'all_question_details': all_detailed_results
        }
        
        # Save enhanced results
        clean_model_name = args.model.replace("/", "-")
        benchmark_name = os.path.splitext(os.path.basename(args.benchmark_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_filename = f"{clean_model_name}-{benchmark_name}-{timestamp}.json"
        results_filepath = os.path.join(args.output_path, results_filename)
        
        # Ensure output directory exists
        os.makedirs(args.output_path, exist_ok=True)
        
        with open(results_filepath, "w") as f:
            json.dump(enhanced_results, f, indent=2)
        
        print(f"\nEnhanced results saved to {results_filepath}")
        print(f"Total detailed question records: {len(enhanced_results['all_question_details'])}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print(f"Benchmark file: {args.benchmark_path}")
        print(f"Model: {args.model}")
        print(f"Parameters: temp={args.temperature}, demographics={not args.no_demographics}, context={not args.no_context}")
        
        # Save crash log for debugging
        error_info = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'error': str(e),
            'original_error': repr(e),
            'benchmark_path': args.benchmark_path,
            'model': args.model,
            'parameters': {
                'temperature': args.temperature,
                'include_demographics': not args.no_demographics,
                'include_context': not args.no_context,
                'max_workers': args.max_workers,
                'swap_experiment': args.swap_experiment,
                'max_context_qa': args.max_context_qa
            }
        }
        
        clean_model_name = args.model.replace("/", "-")
        benchmark_name = os.path.splitext(os.path.basename(args.benchmark_path))[0]
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        error_filename = f"crash_log_{clean_model_name}-{benchmark_name}-{timestamp}.json"
        error_filepath = os.path.join(args.output_path, error_filename)
        
        # Ensure output directory exists
        os.makedirs(args.output_path, exist_ok=True)
        
        with open(error_filepath, 'w') as f:
            json.dump(error_info, f, indent=2)
        print(f"Error details saved to: {error_filepath}")

if __name__ == "__main__":
    main()
