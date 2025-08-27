import pandas as pd
import json
from typing import List, Dict, Any
import re
import random
import os

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
                # Invert roles for this path
                inverted_path = []
                for msg in path:
                    inverted_role = 'assistant' if msg['role'] == 'prompter' else 'prompter'
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
    """Convert conversations to chat format with role tokens in text"""
    formatted_conversations = []
    
    for conv in conversations:
        # Create the conversation history with role tokens
        chat_history = ""
        messages = conv['messages']
        
        for i, msg in enumerate(messages[:-1]):  # All messages except the last one
            role_prefix = "Assistant: " if msg['role'] == 'assistant' else "User: "
            chat_history += role_prefix + msg['content'].strip() + "\n"
        
        # The last message is the target response
        target_message = messages[-1]
        target_role_prefix = "Assistant: " if target_message['role'] == 'assistant' else "User: "
        target_text = target_role_prefix + target_message['content'].strip()
        
        # Create the formatted conversation
        formatted_conv = {
            'conversation_id': conv['conversation_id'],
            'chat_history': chat_history.strip(),
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
    
    # Create conversations from individual messages
    print("\nCreating conversations from individual messages...")
    message_conversations = create_conversations_from_messages(df_messages_inverted)
    all_conversations.extend(message_conversations)
    print(f"Created {len(message_conversations)} conversations from messages")
    
    # Filter conversations for quality
    print("\nFiltering conversations for quality...")
    filtered_conversations = filter_conversations(all_conversations)
    print(f"Filtered from {len(all_conversations)} to {len(filtered_conversations)} conversations")
    
    # Convert to chat format with role tokens
    print("\nConverting to chat format with role tokens...")
    formatted_conversations = create_chat_format_with_role_tokens(filtered_conversations)
    print(f"Formatted {len(formatted_conversations)} conversations")
    
    # Split into train/validation/test
    print("\nSplitting dataset into train/validation/test...")
    split_data = split_dataset(formatted_conversations, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05)
    
    print(f"Train set: {len(split_data['train'])} conversations")
    print(f"Validation set: {len(split_data['validation'])} conversations")
    print(f"Test set: {len(split_data['test'])} conversations")
    
    # Statistics
    print(f"\nFinal dataset statistics:")
    total_conversations = len(formatted_conversations)
    print(f"Total conversations: {total_conversations}")
    
    # Analyze conversation lengths
    conv_lengths = [conv['path_length'] for conv in formatted_conversations]
    print(f"Average conversation length: {sum(conv_lengths) / len(conv_lengths):.1f} messages")
    print(f"Min conversation length: {min(conv_lengths)} messages")
    print(f"Max conversation length: {max(conv_lengths)} messages")
    
    # Source distribution
    source_counts = {}
    for conv in formatted_conversations:
        source = conv.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\nSource distribution:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    
    # Target role distribution
    target_role_counts = {}
    for conv in formatted_conversations:
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
    for i, conv in enumerate(formatted_conversations[:3]):
        print(f"\nConversation {i+1} (ID: {conv['conversation_id']}, Source: {conv.get('source', 'unknown')}):")
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

if __name__ == "__main__":
    main()