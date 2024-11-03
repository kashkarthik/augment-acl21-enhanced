"""
@author: Kaushik Karthikeyan

Custom Script that performs both Back-translation and Paraphrasing on original dataset.

External libraries used:
Google Translate API for Back-Translation
Hugging Face's transformer models for paraphrasing

Generates: A new set of augmented question-answer pairs that preserve the orignal meaning but differnt in structure and diction 


"""
import pandas as pd
from deep_translator import GoogleTranslator
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

# Load original data
data_path = '../data/original_dataset.csv'
df = pd.read_csv(data_path)

# Back-Translation Function
def back_translate(text, src='en', intermediate_lang='fr'):
    translated = GoogleTranslator(source='en', target=intermediate_lang).translate(text)
    back_translated = GoogleTranslator(source=intermediate_lang, target='en').translate(translated)
    return back_translated

# Paraphrasing Function
model_name = 'tuner007/pegasus_paraphrase'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def paraphrase(text, num_return_sequences=2, num_beams=5):
    batch = tokenizer.prepare_seq2seq_batch([text], truncation=True, padding='longest', max_length=60, return_tensors="pt")
    translated = model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences)
    paraphrases = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return paraphrases

# Augmentation Process
def augment_data(df):
    # Back-Translation
    languages = ['fr', 'de', 'es']
    augmented_data = []
    for index, row in df.iterrows():
        question = row['Question']
        answer = row['Answer']
        label = row['Label']
        for lang in languages:
            aug_question = back_translate(question, intermediate_lang=lang)
            aug_answer = back_translate(answer, intermediate_lang=lang)
            augmented_data.append({'Question': aug_question, 'Answer': aug_answer, 'Label': label})
    
    # Paraphrasing
    for index, row in df.iterrows():
        question = row['Question']
        answer = row['Answer']
        label = row['Label']
        para_questions = paraphrase(question)
        para_answers = paraphrase(answer)
        for pq, pa in zip(para_questions, para_answers):
            augmented_data.append({'Question': pq, 'Answer': pa, 'Label': label})
    
    # Create DataFrame
    augmented_df = pd.DataFrame(augmented_data)
    return augmented_df

# Generate augmented data
augmented_df = augment_data(df)

# Combine with original data
combined_df = pd.concat([df, augmented_df], ignore_index=True)
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Save augmented data
combined_df.to_csv('../data/augmented_dataset.csv', index=False)
