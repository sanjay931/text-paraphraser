import json
from nltk.translate.bleu_score import sentence_bleu

# Function to read JSON file
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


files = ['quora.json','squad_2.json','cnn_news.json','MSCOCO.json']

weights = (0, 1, 0, 0)  # Weights for uni-gram, bi-gram, tri-gram, and 4-gram
for file in files:
    blue_scores = []
    eval_data =  read_json(f'./eval_results/{file}')
    for index, row in enumerate(eval_data):
        try:
            references, generated = row['reference'], row['generated']
            references = [each_reference.split() for each_reference in references]
            total_score_of_row = []
            for prediction in generated:
                prediction = prediction.split()
                score = sentence_bleu(references, prediction, weights=weights)
                total_score_of_row.append(score)
            avg_score = sum(total_score_of_row)/len(total_score_of_row)
            blue_scores.append(avg_score)
        except Exception as e:
            print(f"Error in score calculation for index {index} on file {file}")
    print(f"Avg BLUE score for {file} is {sum(blue_scores)/len(blue_scores)}")

