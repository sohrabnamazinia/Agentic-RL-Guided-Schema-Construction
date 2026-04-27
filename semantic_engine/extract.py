import numpy as np
import pandas as pd
import openai
from tqdm import tqdm
import ast
import random
import json

from sentence_transformers import SentenceTransformer

from sklearn.cluster import KMeans

'''
Semantic Engine (NJIT version)

Extracts fields using clustering based approach.
1. Use LLM to extract "concepts" based on a set of questions
2. Embed and cluster concepts for various k
3. Feed clusters into LLM to categorize concepts into fields
4. Return set of candidate fields
5. Run for all form types, generating candidate fields for each form type

'''

def extractConcepts(data, f_type, run=True):

    if not run:
        return []

    concepts = []

    prompt = """
                        You are a semantic engine within a Navy form filling system.

                        Your task: Extract semantic concepts from Navy maintenance forms and return them as a Python list of strings. Output ONLY the list. No explanation, no preamble, no extra text.

                        Focus on extracting:
                        - Objects referenced
                        - Condition of each object
                        - Location
                        - Hazard or problem
                        - Mitigation actions

                        Example input:
                            Protective netting ripped near starboard frame 32; mesh hanging below deck.
                            Loose guard rail on port side; temporary tape barrier installed.
                            Cracked ladder rung midship; corrosion visible.
                            Missing safety signage near hatch entrance.

                        Example output:
                        ["protective netting", "ripped", "starboard frame 32", "fall risk", "mesh hanging", "guard rail", "loose", "port side", "tape barrier", "ladder rung", "cracked", "midship", "corrosion", "safety signage", "missing"]

                        Now extract concepts from the following forms:
                     """

    length = len(data)

    for i in range(0, length, 5):
        sub_data = data.iloc[i:i+5]
        forms = ""

        for j, row in sub_data.iterrows():
            text = f"{row['problem']} | {row['recc_sol']} | {row['actual_sol']}"
            forms += f"\n {text}"
        # print(forms)

        response = []
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-70B-Instruct",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": forms}
            ],
            max_tokens=2000,
            temperature=0.0
        )

        response = ast.literal_eval(response.choices[0].message.content.strip())
        
        concepts.extend(response)

        concepts = list(set(concepts))

    return concepts

def clusterConcepts(concepts, max_k=10):

    embeddings = []
    clusters = []

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(concepts)
    
    for k in range(4, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        cluster_groups = [[] for _ in range(k)]
        for concept, label in zip(concepts, labels):
            cluster_groups[label].append(concept)
        
        clusters.append(cluster_groups)

    return clusters


def groupFields(clusters):

    fields = {}

    prompt = """
                You are a semantic engine within a Navy form filling system.

                Your task: Given a list of semantic concepts from Navy maintenance forms, return the single field/category they all belong to. Output ONLY the category name. No explanation, no preamble, no extra text.

                Example input 1:
                    ["Ripped", "Cracked", "Loose"]
                Example output 1:
                    Condition

                Example input 2:
                    ["Starboard frame 32", "Port side", "Midship"]
                Example output 2:
                    Location

                Example input 3:
                    ["Fall risk", "Trip hazard"]
                Example output 3:
                    Hazard

                Example input 4:
                    ["Tape barrier", "Barricade installed"]
                Example output 4:
                    Mitigation

                Now categorize the following concepts:
            """

    for K in clusters:
        for cluster in K:
            response = []
            
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(cluster)}
                ],
                max_tokens=2000,
                temperature=0.0
            )

            field = response.choices[0].message.content.strip()
            sample = random.sample(cluster, min(10, len(cluster)))
            
            if field in fields:
                fields[field].extend(sample)
                fields[field] = list(set(fields[field]))
            else:
                fields[field] = sample

    return fields


all_fields = {}

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

data_path = './grouped_form_data.csv'

print('Reading data...')

df = pd.read_csv(data_path)
form_types = df['form_type_label'].unique()

print('Processing following form types: \n')
print(form_types)
print('\n')

for f_type in form_types[1:]:

    data = df[df['form_type_label'] == f_type]

    concepts = extractConcepts(data.iloc[0:50], f_type, False) # return type: list --- there are 500 samples of each form type. I choose 50 to parse through for concepts, you can choose more

    if len(concepts):
        with open(f'concepts_{f_type}.txt', 'w') as f:
            for concept in concepts:
                f.write(concept + '\n')
    else:
        with open(f'concepts_{f_type}.txt', 'r') as f:
            concepts = [line.strip() for line in f.readlines()]

    clusters = clusterConcepts(concepts)
    
    fields = groupFields(clusters)

    all_fields[f_type] = fields
    
    print('Finished processing form type: ', f_type, '\n')

    with open('fields.json', 'w') as f:
        json.dump(all_fields, f, indent=4)
