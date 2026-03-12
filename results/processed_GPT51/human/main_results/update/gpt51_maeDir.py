import json
from pathlib import Path
from collections import defaultdict

def normalize_scale(value, scale):
    """Normalize value from scale to [1,5] range."""
    if scale == [1, 5]:
        return value
    elif scale == [1, 10]:
        return (value - 1) * (5 - 1) / (10 - 1) + 1
    else:
        scale_min, scale_max = scale
        return (value - scale_min) * (5 - 1) / (scale_max - scale_min) + 1

def calculate_simplified_mae(generated, correct, scale):
    """Calculate simplified normalized MAE: for [1,10] divide by 2, for [1,5] keep as is."""
    if scale == [1, 10]:
        return abs(generated - correct) / 2.0
    else:
        return abs(generated - correct)

def calculate_uas_mae_norm(mae_list):
    """Calculate UAS normalized MAE: max(0, min(1, 1 - MAE/MAE_max))."""
    if not mae_list:
        return None
    mae = sum(mae_list) / len(mae_list)
    mae_max = 4.0  # for 1-5 scale
    return max(0.0, min(1.0, 1.0 - mae / mae_max))

def sign(value):
    """Return sign of value: +1 for positive, -1 for negative, 0 for zero."""
    return 1 if value > 0 else (-1 if value < 0 else 0)

def calculate_directional_accuracy(questions, topic):
    """Calculate directional accuracy for update tasks."""
    base_questions = {'zoning': '1.1', 'surveillance': '2.1', 'healthcare': '3.1'}
    opinion_questions = {
        'zoning': ['1.1', '1.2', '1.3', '1.4'],
        'surveillance': ['2.1', '2.6', '2.7', '2.8'],
        'healthcare': ['3.1', '3.6', '3.7', '3.8', '3.9']
    }
    
    base_q = base_questions[topic]
    topic_questions = opinion_questions[topic]
    
    # Group by prolific_id
    prolific_groups = defaultdict(dict)
    for key, data in questions.items():
        prolific_id = data['prolific_id']
        question_id = data['question_id']
        if question_id in topic_questions:
            prolific_groups[prolific_id][question_id] = data
    
    correct_directions = 0
    total_comparisons = 0
    correct_up_up = 0
    correct_down_down = 0
    wrong_direction = 0
    
    for prolific_id, user_questions in prolific_groups.items():
        if base_q not in user_questions:
            continue
        
        base_data = user_questions[base_q]
        
        for question_id in topic_questions:
            if question_id != base_q and question_id in user_questions:
                current_data = user_questions[question_id]
                
                gt_delta = current_data['user_answer'] - base_data['user_answer']
                pred_delta = current_data['generated_answer'] - base_data['generated_answer']
                
                gt_sign = sign(gt_delta)
                pred_sign = sign(pred_delta)
                
                if gt_sign == pred_sign:
                    correct_directions += 1
                    if gt_sign == 1:
                        correct_up_up += 1
                    elif gt_sign == -1:
                        correct_down_down += 1
                elif gt_sign != 0 and pred_sign != 0:
                    wrong_direction += 1
                
                total_comparisons += 1
    
    if total_comparisons == 0:
        return None
    
    # Calculate two-stage weighted accuracy
    stage1_numerator = correct_directions + wrong_direction
    stage1_accuracy = stage1_numerator / total_comparisons
    
    stage2_denominator = correct_up_up + correct_down_down + wrong_direction
    stage2_accuracy = (correct_up_up + correct_down_down) / stage2_denominator if stage2_denominator > 0 else 0.0
    
    weighted_accuracy = 0.3 * stage1_accuracy + 0.7 * stage2_accuracy
    
    return {
        'weighted': weighted_accuracy,
        'stage1': stage1_accuracy,
        'stage2': stage2_accuracy
    }

def extract_opinion_questions(detailed_results):
    """Extract opinion questions with scale 1-10 from detailed results."""
    questions = {}
    for item in detailed_results:
        vqa = item.get('vqa', {})
        question_id = vqa.get('question_id', '')
        question_type = vqa.get('question_type', '')
        scale = vqa.get('scale', [])
        prolific_id = vqa.get('prolific_id', '')
        
        if question_type == 'opinion' and scale == [1, 10]:
            user_answer = vqa.get('user_answer')
            generated_answer = item.get('generated_answer')
            
            if user_answer is not None and generated_answer is not None:
                key = f"{prolific_id}_{question_id}"
                questions[key] = {
                    'prolific_id': prolific_id,
                    'question_id': question_id,
                    'user_answer': user_answer,
                    'generated_answer': generated_answer
                }
    
    return questions

def extract_topic_from_filename(filename):
    """Extract topic from filename."""
    for topic in ['healthcare', 'surveillance', 'zoning']:
        if topic in filename:
            return topic
    return None

def process_file(file_path):
    """Process a single update JSON file and recalculate metrics."""
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    detailed_results = data.get('all_question_details', [])
    
    # Recalculate metrics
    absolute_errors = []
    simplified_errors = []
    
    for item in detailed_results:
        generated = item.get('generated_answer')
        correct = item.get('correct_answer')
        vqa = item.get('vqa', {})
        scale = vqa.get('scale', [1, 10])
        
        if generated is not None and correct is not None:
            # Normalized MAE
            norm_generated = normalize_scale(generated, scale)
            norm_correct = normalize_scale(correct, scale)
            absolute_errors.append(abs(norm_generated - norm_correct))
            
            # Simplified MAE
            simplified_errors.append(calculate_simplified_mae(generated, correct, scale))
    
    # Calculate metrics
    normalized_mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
    simplified_mae = sum(simplified_errors) / len(simplified_errors) if simplified_errors else None
    uas_mae_norm = calculate_uas_mae_norm(simplified_errors)
    
    # Calculate directional accuracy
    topic = extract_topic_from_filename(file_path.name)
    dir_results = None
    if topic:
        questions = extract_opinion_questions(detailed_results)
        dir_results = calculate_directional_accuracy(questions, topic)
    
    # Update overall_metrics
    data['overall_metrics']['overall_mae'] = normalized_mae
    data['overall_metrics']['simplified_mae'] = simplified_mae
    data['overall_metrics']['uas_mae_norm'] = uas_mae_norm
    if dir_results:
        data['overall_metrics']['weighted_directional_accuracy'] = dir_results['weighted']
        data['overall_metrics']['stage1_directional_accuracy'] = dir_results['stage1']
        data['overall_metrics']['stage2_directional_accuracy'] = dir_results['stage2']
    else:
        data['overall_metrics']['weighted_directional_accuracy'] = None
        data['overall_metrics']['stage1_directional_accuracy'] = None
        data['overall_metrics']['stage2_directional_accuracy'] = None
    
    # Update group_results if exists
    if 'group_results' in data:
        for group_name, group_data in data['group_results'].items():
            group_data['mae'] = normalized_mae
            group_data['simplified_mae'] = simplified_mae
    
    # Save updated file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return {
        'file': file_path.name,
        'normalized_mae': normalized_mae,
        'simplified_mae': simplified_mae,
        'uas_mae_norm': uas_mae_norm,
        'weighted_dir': weighted_dir
    }

def main():
    """Main function to process all update files."""
    
    update_dir = Path(__file__).parent
    
    print("Processing GPT-5.1 update files...\n")
    
    # Process files in folders 1-5
    results = []
    for run_num in range(1, 6):
        run_dir = update_dir / str(run_num)
        if not run_dir.exists():
            print(f"Warning: Folder {run_num} not found")
            continue
        
        print(f"Processing Run {run_num}:")
        
        # Find all JSON files in this run
        json_files = sorted(run_dir.glob('*.json'))
        
        for json_file in json_files:
            result = process_file(json_file)
            results.append(result)
            
            print(f"  ✓ {result['file']}")
            print(f"    Normalized MAE: {result['normalized_mae']:.4f}")
            print(f"    Simplified MAE: {result['simplified_mae']:.4f}")
            print(f"    UAS MAE Norm: {result['uas_mae_norm']:.4f}")
            if result['weighted_dir'] is not None:
                print(f"    Weighted Dir Acc: {result['weighted_dir']:.2%}")
            print()
    
    print(f"\nCompleted! Processed {len(results)} files total.")

if __name__ == "__main__":
    main()

