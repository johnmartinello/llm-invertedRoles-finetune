import pandas as pd
import json
import os

# Create data folder if it doesn't exist
data_folder = "data"

# Read from data folder
df_messages = pd.read_json(os.path.join(data_folder, "messages.jsonl"), lines=True)
df_trees = pd.read_json(os.path.join(data_folder, "trees.jsonl"), lines=True)

print("Original df_messages columns:", df_messages.columns.tolist())
print("Original df_trees columns:", df_trees.columns.tolist())


essential_message_columns = ['message_id', 'text', 'role', 'lang']
df_messages_clean = df_messages[essential_message_columns].copy()

print(f"\nCleaned df_messages shape: {df_messages_clean.shape}")
print("Cleaned df_messages columns:", df_messages_clean.columns.tolist())

def extract_messages_from_tree(tree_data):
    """Extract all messages from a tree structure recursively"""
    messages = []
    
    def extract_from_node(node, tree_id):
        if isinstance(node, dict):
            message_data = {
                'tree_id': tree_id,
                'message_id': node.get('message_id'),
                'text': node.get('text'),
                'role': node.get('role'),
                'lang': node.get('lang')
            }
            messages.append(message_data)
            
            replies = node.get('replies', [])
            for reply in replies:
                extract_from_node(reply, tree_id)
    
    if 'prompt' in tree_data:
        extract_from_node(tree_data['prompt'], tree_data.get('message_tree_id'))
    
    return messages

# Clean df_trees by extracting all messages from the nested structure
all_tree_messages = []
for _, tree_row in df_trees.iterrows():
    tree_messages = extract_messages_from_tree(tree_row.to_dict())
    all_tree_messages.extend(tree_messages)

# Create clean dataframe from tree messages
df_trees_clean = pd.DataFrame(all_tree_messages)

print(f"\nCleaned df_trees shape: {df_trees_clean.shape}")
print("Cleaned df_trees columns:", df_trees_clean.columns.tolist())

def invert_roles(df):
    """Invert roles: prompter becomes assistant and assistant becomes prompter"""
    df_inverted = df.copy()
    
    # Create a mapping for role inversion
    role_mapping = {
        'prompter': 'assistant',
        'assistant': 'prompter'
    }
    
    # Apply the role inversion
    df_inverted['role'] = df_inverted['role'].map(role_mapping)
    
    # Add a suffix to indicate this is inverted data
    df_inverted['role_inverted'] = True
    
    return df_inverted

# Create inverted versions of both dataframes
df_messages_inverted = invert_roles(df_messages_clean)
df_trees_inverted = invert_roles(df_trees_clean)

# Display sample of cleaned data
print("\nSample of cleaned df_messages:")
print(df_messages_clean.head())

print("\nSample of cleaned df_trees:")
print(df_trees_clean.head())

print("\nSample of inverted df_messages:")
print(df_messages_inverted.head())

print("\nSample of inverted df_trees:")
print(df_trees_inverted.head())

# Save cleaned dataframes as JSONL (original roles) to data folder
with open(os.path.join(data_folder, "messages_clean.jsonl"), "w", encoding="utf-8") as f:
    for _, row in df_messages_clean.iterrows():
        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

with open(os.path.join(data_folder, "trees_clean.jsonl"), "w", encoding="utf-8") as f:
    for _, row in df_trees_clean.iterrows():
        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

# Save inverted dataframes as JSONL to data folder
with open(os.path.join(data_folder, "messages_inverted.jsonl"), "w", encoding="utf-8") as f:
    for _, row in df_messages_inverted.iterrows():
        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

with open(os.path.join(data_folder, "trees_inverted.jsonl"), "w", encoding="utf-8") as f:
    for _, row in df_trees_inverted.iterrows():
        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

print("\nCleaned dataframes saved to data folder:")
print(f"- '{os.path.join(data_folder, 'messages_clean.jsonl')}' and '{os.path.join(data_folder, 'trees_clean.jsonl')}' (original roles)")
print(f"- '{os.path.join(data_folder, 'messages_inverted.jsonl')}' and '{os.path.join(data_folder, 'trees_inverted.jsonl')}' (inverted roles)")

# Show some statistics
print(f"\nStatistics:")
print(f"Original messages: {len(df_messages)}")
print(f"Cleaned messages: {len(df_messages_clean)}")
print(f"Original trees: {len(df_trees)}")
print(f"Extracted tree messages: {len(df_trees_clean)}")
print(f"Unique tree IDs: {df_trees_clean['tree_id'].nunique()}")

# Show role distribution before and after inversion
print(f"\nRole distribution in original cleaned data:")
print(df_messages_clean['role'].value_counts())
print(df_trees_clean['role'].value_counts())

print(f"\nRole distribution in inverted data:")
print(df_messages_inverted['role'].value_counts())
print(df_trees_inverted['role'].value_counts())