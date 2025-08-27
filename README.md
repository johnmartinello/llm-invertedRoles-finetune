# LLM Inverted Roles Fine-tuning 

## Overview

This project is a fun experiment that instead of the AI acting as an assistant and humans asking questions, this model is trained to act as the human while humans take on the role of the assistant. 

## Project Purpose

The experiment aims to:
- Train an AI model to generate human-like questions and prompts
- Modify a dataset where the AI "asks" and humans "answer"
- Explore the dynamics of role-reversed conversations


## Technical Approach

### Base Model
The project uses [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B), a 7.6 billion parameter language model from Alibaba Cloud. This model was chosen for its strong performance in conversational tasks and its ability to handle diverse language patterns.

### Dataset
The training data comes from the [OpenAssistant Conversations Dataset (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1), which contains over 88,000 human-AI conversations. The dataset includes:
- Multi-turn conversations between humans and AI assistants
- Conversations in 35 different languages
- High-quality, human-reviewed interactions

### Role Inversion Process

1. **Data Processing**: The original conversations are processed to extract message pairs
2. **Role Swapping**: Human messages (prompter role) become assistant responses, and AI messages (assistant role) become human prompts
3. **Conversation Reconstruction**: New conversation flows are created where the AI generates human-like questions and humans provide assistant-like answers

### Fine-tuning Method
The model is fine-tuned using **QLoRA (Quantized Low-Rank Adaptation)**, which allows efficient training of large language models on consumer hardware by:
- Using 4-bit quantization to reduce memory usage
- Applying low-rank adapters to update only a small subset of parameters
- Maintaining model performance while significantly reducing computational requirements

## Project Structure

- **Data Processing**: Scripts to clean and invert the conversation roles
- **Dataset Creation**: Tools to generate training, validation, and test splits
- **Model Training**: QLoRA fine-tuning implementation
- **Evaluation**: Methods to assess the model's performance in role-reversed conversations

## Results



