import pandas as pd
import json
from typing import List, Dict, Any, Tuple
import re
import random
import os
from collections import Counter

def load_original_tree_data():
    """Load the original tree data to reconstruct conversations properly"""
    data_folder = "data"
    try:
        df_trees_original = pd.read_json(os.path.join(data_folder, "trees.jsonl"), lines=True)
        df_messages_inverted = pd.read_json(os.path.join(data_folder, "messages_inverted.jsonl"), lines=True)
        print(f"Loaded {len(df_trees_original)} original trees and {len(df_messages_inverted)} inverted messages")
        return df_trees_original, df_messages_inverted
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure trees.jsonl and messages_inverted.jsonl exist in the data folder")
        return None, None

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization function (word-based) for analysis"""
    if not text:
        return []
    # Split on whitespace and basic punctuation
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    return tokens

def estimate_token_count(text: str, avg_chars_per_token: float = 4.0) -> int:
    """Estimate token count based on character count"""
    if not text:
        return 0
    return int(len(text) / avg_chars_per_token)

def analyze_token_lengths(conversations: List[Dict[str, Any]], 
                         sample_size: int = 1000) -> Dict[str, Any]:
    """Analyze token lengths in conversations to determine optimal max_seq_length"""
    
    print(f"\nAnalyzing token lengths (sample size: {sample_size})...")
    
    # Sample conversations for analysis
    sample_conversations = random.sample(conversations, min(sample_size, len(conversations))) if conversations else []
    
    chat_history_lengths = []
    target_response_lengths = []
    total_lengths = []
    
    for conv in sample_conversations:
        chat_history = conv.get('chat_history', '')
        target_response = conv.get('target_response', '')
        
        # Estimate token counts
        chat_tokens = estimate_token_count(chat_history)
        target_tokens = estimate_token_count(target_response)
        total_tokens = chat_tokens + target_tokens
        
        chat_history_lengths.append(chat_tokens)
        target_response_lengths.append(target_tokens)
        total_lengths.append(total_tokens)
    
    # Guard: handle empty lists to avoid division by zero
    if not chat_history_lengths or not target_response_lengths or not total_lengths:
        return {
            'chat_history': {'mean': 0, 'median': 0, 'p95': 0, 'p99': 0, 'max': 0, 'min': 0},
            'target_response': {'mean': 0, 'median': 0, 'p95': 0, 'p99': 0, 'max': 0, 'min': 0},
            'total': {'mean': 0, 'median': 0, 'p95': 0, 'p99': 0, 'max': 0, 'min': 0},
            'empty': True,
        }

    # Calculate statistics
    stats = {
        'chat_history': {
            'mean': sum(chat_history_lengths) / len(chat_history_lengths),
            'median': sorted(chat_history_lengths)[len(chat_history_lengths)//2],
            'p95': sorted(chat_history_lengths)[int(len(chat_history_lengths) * 0.95)],
            'p99': sorted(chat_history_lengths)[int(len(chat_history_lengths) * 0.99)],
            'max': max(chat_history_lengths),
            'min': min(chat_history_lengths)
        },
        'target_response': {
            'mean': sum(target_response_lengths) / len(target_response_lengths),
            'median': sorted(target_response_lengths)[len(target_response_lengths)//2],
            'p95': sorted(target_response_lengths)[int(len(target_response_lengths) * 0.95)],
            'p99': sorted(target_response_lengths)[int(len(target_response_lengths) * 0.99)],
            'max': max(target_response_lengths),
            'min': min(target_response_lengths)
        },
        'total': {
            'mean': sum(total_lengths) / len(total_lengths),
            'median': sorted(total_lengths)[len(total_lengths)//2],
            'p95': sorted(total_lengths)[int(len(total_lengths) * 0.95)],
            'p99': sorted(total_lengths)[int(len(total_lengths) * 0.99)],
            'max': max(total_lengths),
            'min': min(total_lengths)
        }
    }
    
    return stats

def recommend_max_sequence_length(token_stats: Dict[str, Any]) -> Tuple[int, str]:
    """Recommend optimal max sequence length based on token analysis"""
    
    total_p95 = token_stats['total']['p95']
    total_p99 = token_stats['total']['p99']
    total_max = token_stats['total']['max']
    
    # Common max sequence lengths
    common_lengths = [512, 1024, 2048, 4096, 8192]
    
    # Find the best fit
    recommended_length = 1024  # Default
    reasoning = []
    
    if total_p95 <= 512:
        recommended_length = 512
        reasoning.append("95% of conversations fit within 512 tokens")
    elif total_p95 <= 1024:
        recommended_length = 1024
        reasoning.append("95% of conversations fit within 1024 tokens")
    elif total_p95 <= 2048:
        recommended_length = 2048
        reasoning.append("95% of conversations fit within 2048 tokens")
    elif total_p95 <= 4096:
        recommended_length = 4096
        reasoning.append("95% of conversations fit within 4096 tokens")
    else:
        recommended_length = 8192
        reasoning.append("Conversations are very long, using 8192 tokens")
    
    # Check if we're losing too much data
    if total_p99 > recommended_length:
        reasoning.append(f"Warning: 1% of conversations exceed {recommended_length} tokens (max: {total_max})")
    
    return recommended_length, "; ".join(reasoning)

def truncate_conversation(conversation: Dict[str, Any], 
                         max_seq_length: int,
                         truncation_strategy: str = "sliding_window") -> Dict[str, Any]:
    """Truncate conversation to fit within max sequence length"""
    
    chat_history = conversation.get('chat_history', '')
    target_response = conversation.get('target_response', '')
    
    # Estimate current token counts
    chat_tokens = estimate_token_count(chat_history)
    target_tokens = estimate_token_count(target_response)
    total_tokens = chat_tokens + target_tokens
    
    if total_tokens <= max_seq_length:
        return conversation  # No truncation needed
    
    # Calculate how much we need to truncate
    excess_tokens = total_tokens - max_seq_length
    
    if truncation_strategy == "sliding_window":
        # Keep recent messages (sliding window approach)
        lines = chat_history.split('\n')
        truncated_lines = []
        current_tokens = target_tokens  # Start with target response
        
        # Add lines from the end (most recent) until we hit the limit
        for line in reversed(lines):
            line_tokens = estimate_token_count(line)
            if current_tokens + line_tokens <= max_seq_length:
                truncated_lines.insert(0, line)
                current_tokens += line_tokens
            else:
                break
        
        truncated_chat_history = '\n'.join(truncated_lines)
        
    elif truncation_strategy == "truncate_start":
        # Truncate from the beginning (keep recent context)
        target_reserve = min(target_tokens + 100, max_seq_length // 4)  # Reserve space for target
        available_tokens = max_seq_length - target_reserve
        
        lines = chat_history.split('\n')
        truncated_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = estimate_token_count(line)
            if current_tokens + line_tokens <= available_tokens:
                truncated_lines.append(line)
                current_tokens += line_tokens
            else:
                break
        
        truncated_chat_history = '\n'.join(truncated_lines)
    
    else:  # "truncate_end"
        # Truncate from the end (keep early context)
        available_tokens = max_seq_length - target_tokens
        
        lines = chat_history.split('\n')
        truncated_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = estimate_token_count(line)
            if current_tokens + line_tokens <= available_tokens:
                truncated_lines.append(line)
                current_tokens += line_tokens
            else:
                break
        
        truncated_chat_history = '\n'.join(truncated_lines)
    
    # Create truncated conversation
    truncated_conv = conversation.copy()
    truncated_conv['chat_history'] = truncated_chat_history
    truncated_conv['truncated'] = True
    truncated_conv['original_total_tokens'] = total_tokens
    truncated_conv['truncated_total_tokens'] = estimate_token_count(truncated_chat_history) + target_tokens
    
    return truncated_conv

def apply_tokenization_constraints(conversations: List[Dict[str, Any]], 
                                 max_seq_length: int = 1024,
                                 truncation_strategy: str = "sliding_window") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply tokenization constraints to conversations"""
    
    print(f"\nApplying tokenization constraints (max_seq_length: {max_seq_length}, strategy: {truncation_strategy})...")
    
    processed_conversations = []
    truncation_stats = {
        'total_conversations': len(conversations),
        'truncated_conversations': 0,
        'dropped_conversations': 0,
        'avg_truncation_ratio': 0.0
    }
    
    truncation_ratios = []
    
    for conv in conversations:
        chat_history = conv.get('chat_history', '')
        target_response = conv.get('target_response', '')
        
        # Estimate token counts
        chat_tokens = estimate_token_count(chat_history)
        target_tokens = estimate_token_count(target_response)
        total_tokens = chat_tokens + target_tokens
        
        if total_tokens <= max_seq_length:
            # No truncation needed
            processed_conversations.append(conv)
        else:
            # Truncate the conversation
            truncated_conv = truncate_conversation(conv, max_seq_length, truncation_strategy)
            
            # Check if truncation was successful
            final_tokens = truncated_conv['truncated_total_tokens']
            if final_tokens <= max_seq_length:
                processed_conversations.append(truncated_conv)
                truncation_stats['truncated_conversations'] += 1
                
                # Calculate truncation ratio
                original_tokens = truncated_conv['original_total_tokens']
                truncation_ratio = (original_tokens - final_tokens) / original_tokens
                truncation_ratios.append(truncation_ratio)
            else:
                # Still too long, drop the conversation
                truncation_stats['dropped_conversations'] += 1
    
    # Calculate average truncation ratio
    if truncation_ratios:
        truncation_stats['avg_truncation_ratio'] = sum(truncation_ratios) / len(truncation_ratios)
    
    return processed_conversations, truncation_stats

def extract_conversation_paths(tree_data: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """Extract all possible conversation paths from a tree structure"""
    paths = []
    
    def extract_paths_from_node(node: Dict[str, Any], current_path: List[Dict[str, Any]]):
        """Recursively extract all conversation paths"""
        if not isinstance(node, dict):
            return
        
        # Add current node to path
        current_path.append({
            'message_id': node.get('message_id'),
            'text': node.get('text', ''),
            'role': node.get('role'),
            'lang': node.get('lang')
        })
        
        # Get replies
        replies = node.get('replies', [])
        
        if not replies:
            # End of path, save it
            paths.append(current_path.copy())
        else:
            # Continue with each reply (creating branching paths)
            for reply in replies:
                extract_paths_from_node(reply, current_path.copy())
    
    # Start from the prompt
    if 'prompt' in tree_data:
        extract_paths_from_node(tree_data['prompt'], [])
    
    return paths

def create_conversations_from_trees(df_trees: pd.DataFrame) -> List[Dict[str, Any]]:
    """Create conversations from tree structures by following conversation paths"""
    conversations = []
    
    for idx, tree_row in df_trees.iterrows():
        tree_data = tree_row.to_dict()
        tree_id = tree_data.get('message_tree_id', f'tree_{idx}')
        
        # Extract all conversation paths from this tree
        paths = extract_conversation_paths(tree_data)
        
        # Create a conversation for each path
        for path_idx, path in enumerate(paths):
            if len(path) > 1:  # Only include paths with multiple messages
                # Invert roles for this path, preserving unknown roles
                inverted_path = []
                for msg in path:
                    if msg['role'] == 'prompter':
                        inverted_role = 'assistant'
                    elif msg['role'] == 'assistant':
                        inverted_role = 'prompter'
                    else:
                        inverted_role = msg['role']
                    inverted_path.append({
                        'role': inverted_role,
                        'content': msg['text']
                    })
                
                conversations.append({
                    'conversation_id': f'{tree_id}_path_{path_idx}',
                    'messages': inverted_path,
                    'source': 'tree_path',
                    'original_tree_id': tree_id,
                    'path_length': len(inverted_path)
                })
    
    return conversations

def create_conversations_from_messages(df_messages: pd.DataFrame, 
                                     max_messages_per_conversation: int = 6,
                                     min_messages_per_conversation: int = 2) -> List[Dict[str, Any]]:
    """Create conversation sequences from individual messages"""
    conversations = []
    current_conversation = []
    
    # Sort messages by message_id to maintain order
    df_messages = df_messages.sort_values('message_id')
    
    for _, msg in df_messages.iterrows():
        current_conversation.append({
            'role': msg['role'],
            'content': msg['text']
        })
        
        # If we reach the max messages per conversation, save it and start a new one
        if len(current_conversation) >= max_messages_per_conversation:
            if len(current_conversation) >= min_messages_per_conversation:
                conversations.append({
                    'conversation_id': f'msg_conv_{len(conversations)}',
                    'messages': current_conversation.copy(),
                    'source': 'messages',
                    'path_length': len(current_conversation)
                })
            current_conversation = []
    
    # Add the last conversation if it has enough messages
    if len(current_conversation) >= min_messages_per_conversation:
        conversations.append({
            'conversation_id': f'msg_conv_{len(conversations)}',
            'messages': current_conversation,
            'source': 'messages',
            'path_length': len(current_conversation)
        })
    
    return conversations

def filter_conversations(conversations: List[Dict[str, Any]], 
                        min_length: int = 2,
                        max_length: int = 20,
                        min_content_length: int = 10) -> List[Dict[str, Any]]:
    """Filter conversations based on quality criteria"""
    filtered = []
    
    for conv in conversations:
        # Check length
        if not (min_length <= len(conv['messages']) <= max_length):
            continue
        
        # Check content quality
        valid_messages = 0
        for msg in conv['messages']:
            if len(msg['content'].strip()) >= min_content_length:
                valid_messages += 1
        
        # At least half of messages should have substantial content
        if valid_messages >= len(conv['messages']) * 0.5:
            filtered.append(conv)
    
    return filtered

def create_chat_format_with_role_tokens(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert conversations to chat format with role tokens in text.
    Create one training sample for every 'user' turn (the model will learn to speak as the user).
    """
    formatted_conversations = []

    for conv in conversations:
        messages = conv['messages']

        # Build examples for each inverted 'prompter' turn (renders as "User: ") with at least 1 prior message as context
        for i, target_message in enumerate(messages):
            # After inversion, 'prompter' corresponds to the human/user side
            if target_message['role'] != 'prompter':
                continue
            if i == 0:
                continue

            # Build chat history from messages before i
            history_lines = []
            for msg in messages[:i]:
                role_prefix = "Assistant: " if msg['role'] == 'assistant' else "User: "
                history_lines.append(role_prefix + msg['content'].strip())
            chat_history = "\n".join(history_lines)

            # Target is the current 'prompter' (renders as User)
            target_role_prefix = "User: "
            target_text = target_role_prefix + target_message['content'].strip()

            formatted_conv = {
                'conversation_id': f"{conv['conversation_id']}_u{i}",
                'chat_history': chat_history,
                'target_response': target_text,
                'source': conv.get('source', 'unknown'),
                'path_length': conv.get('path_length', len(messages)),
                'target_role': target_message['role']
            }
            formatted_conversations.append(formatted_conv)

    return formatted_conversations

def split_dataset(conversations: List[Dict[str, Any]], 
                 train_ratio: float = 0.9, 
                 val_ratio: float = 0.05, 
                 test_ratio: float = 0.05,
                 random_seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Split dataset into train/validation/test sets"""
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Shuffle conversations
    shuffled_conversations = conversations.copy()
    random.shuffle(shuffled_conversations)
    
    total_count = len(shuffled_conversations)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)
    
    # Split the data
    train_data = shuffled_conversations[:train_end]
    val_data = shuffled_conversations[train_end:val_end]
    test_data = shuffled_conversations[val_end:]
    
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }

def create_finetuning_dataset(split_data: Dict[str, List[Dict[str, Any]]], 
                             include_metadata: bool = True) -> None:
    """Create fine-tuning datasets in JSONL format for each split"""
    
    data_folder = "data"
    
    for split_name, conversations in split_data.items():
        filename = os.path.join(data_folder, f"finetuning_dataset_{split_name}.jsonl")
        
        with open(filename, "w", encoding="utf-8") as f:
            for conv in conversations:
                # Create clean format for fine-tuning
                clean_conv = {
                    "conversation_id": conv['conversation_id'],
                    "chat_history": conv['chat_history'],
                    "target_response": conv['target_response']
                }
                if include_metadata:
                    clean_conv.update({
                        "source": conv.get('source', 'unknown'),
                        "path_length": conv.get('path_length', 0),
                        "target_role": conv.get('target_role', 'unknown')
                    })
                
                f.write(json.dumps(clean_conv, ensure_ascii=False) + "\n")
        
        print(f"Saved {len(conversations)} conversations to {filename}")

def main():
    """Main function to create conversation dataset"""
    print("Loading data...")
    df_trees_original, df_messages_inverted = load_original_tree_data()
    
    if df_trees_original is None or df_messages_inverted is None:
        return
    
    all_conversations = []
    
    # Create conversations from tree structures (properly following conversation paths)
    print("\nCreating conversations from tree structures...")
    tree_conversations = create_conversations_from_trees(df_trees_original)
    all_conversations.extend(tree_conversations)
    print(f"Created {len(tree_conversations)} conversations from trees")
    
    # Filter conversations for quality
    print("\nFiltering conversations for quality...")
    filtered_conversations = filter_conversations(all_conversations)
    print(f"Filtered from {len(all_conversations)} to {len(filtered_conversations)} conversations")
    
    # Convert to chat format with role tokens
    print("\nConverting to chat format with role tokens...")
    formatted_conversations = create_chat_format_with_role_tokens(filtered_conversations)
    print(f"Formatted {len(formatted_conversations)} conversations")
    
    # Analyze token lengths and recommend max sequence length
    token_stats = analyze_token_lengths(formatted_conversations)
    if token_stats.get('empty'):
        recommended_length, reasoning = 512, "No conversations to analyze; defaulting to 512"
    else:
        recommended_length, reasoning = recommend_max_sequence_length(token_stats)
    
    print(f"\nToken Length Analysis:")
    print(f"Chat History - Mean: {token_stats['chat_history']['mean']:.1f}, P95: {token_stats['chat_history']['p95']:.1f}, Max: {token_stats['chat_history']['max']}")
    print(f"Target Response - Mean: {token_stats['target_response']['mean']:.1f}, P95: {token_stats['target_response']['p95']:.1f}, Max: {token_stats['target_response']['max']}")
    print(f"Total - Mean: {token_stats['total']['mean']:.1f}, P95: {token_stats['total']['p95']:.1f}, Max: {token_stats['total']['max']}")
    
    print(f"\nRecommended max_sequence_length: {recommended_length}")
    print(f"Reasoning: {reasoning}")
    
    # Apply tokenization constraints
    max_seq_length = recommended_length  # You can override this if needed
    truncation_strategy = "sliding_window"  # Options: "sliding_window", "truncate_start", "truncate_end"
    
    processed_conversations, truncation_stats = apply_tokenization_constraints(
        formatted_conversations, max_seq_length, truncation_strategy
    )
    
    print(f"\nTokenization Results:")
    print(f"Total conversations: {truncation_stats['total_conversations']}")
    print(f"Truncated conversations: {truncation_stats['truncated_conversations']}")
    print(f"Dropped conversations: {truncation_stats['dropped_conversations']}")
    print(f"Average truncation ratio: {truncation_stats['avg_truncation_ratio']:.2%}")
    
    # Split into train/validation/test
    print("\nSplitting dataset into train/validation/test...")
    split_data = split_dataset(processed_conversations, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05)
    
    print(f"Train set: {len(split_data['train'])} conversations")
    print(f"Validation set: {len(split_data['validation'])} conversations")
    print(f"Test set: {len(split_data['test'])} conversations")
    
    # Statistics
    print(f"\nFinal dataset statistics:")
    total_conversations = len(processed_conversations)
    print(f"Total conversations: {total_conversations}")
    
    # Analyze conversation lengths
    conv_lengths = [conv['path_length'] for conv in processed_conversations]
    print(f"Average conversation length: {sum(conv_lengths) / len(conv_lengths):.1f} messages")
    print(f"Min conversation length: {min(conv_lengths)} messages")
    print(f"Max conversation length: {max(conv_lengths)} messages")
    
    # Source distribution
    source_counts = {}
    for conv in processed_conversations:
        source = conv.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\nSource distribution:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    
    # Target role distribution
    target_role_counts = {}
    for conv in processed_conversations:
        role = conv.get('target_role', 'unknown')
        target_role_counts[role] = target_role_counts.get(role, 0) + 1
    
    print(f"\nTarget role distribution:")
    for role, count in target_role_counts.items():
        print(f"  {role}: {count}")
    
    # Create JSONL datasets for each split
    print("\nCreating fine-tuning datasets...")
    create_finetuning_dataset(split_data)
    
    # Show sample conversations
    print("\nSample conversations:")
    for i, conv in enumerate(processed_conversations[:3]):
        print(f"\nConversation {i+1} (ID: {conv['conversation_id']}, Source: {conv.get('source', 'unknown')}):")
        if conv.get('truncated', False):
            print(f"[TRUNCATED - Original: {conv['original_total_tokens']} tokens, Final: {conv['truncated_total_tokens']} tokens]")
        print(f"Chat History:")
        print(conv['chat_history'])
        print(f"Target Response:")
        print(conv['target_response'])
        print("-" * 50)
    
    data_folder = "data"
    print(f"\nDataset creation complete! You can now use the generated files for LLM fine-tuning:")
    print(f"- {os.path.join(data_folder, 'finetuning_dataset_train.jsonl')}")
    print(f"- {os.path.join(data_folder, 'finetuning_dataset_validation.jsonl')}") 
    print(f"- {os.path.join(data_folder, 'finetuning_dataset_test.jsonl')}")
    print(f"\nRecommended training parameters:")
    print(f"- max_sequence_length: {max_seq_length}")
    print(f"- truncation_strategy: {truncation_strategy}")

if __name__ == "__main__":
    main()
