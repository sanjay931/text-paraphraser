from bert_score import score
from datasets import load_dataset
from ast import literal_eval
from paraphraser import paraphrase


def calculate_bert_score(sources, targets, lang='en'):
    P, R, F1 = score(targets, sources, lang=lang)
    avg_P = P.mean().item()
    avg_R = R.mean().item()
    avg_F1 = F1.mean().item()
    return avg_P, avg_R, avg_F1




# Load the dataset
dataset = load_dataset("humarin/chatgpt-paraphrases")

# Access the train split
train_dataset = dataset['train']

# Extract unique sources
sources = set(train_dataset['source'])
print(sources)

# Initialize a dictionary to store sampled data
sampled_data = {}

# Function to sample 500 data points from each source
def sample_from_source(source_name, num_samples=500):
    source_dataset = train_dataset.filter(lambda example: example['source'] == source_name)
    # Sample the data (if fewer than num_samples, take all available data)
    if len(source_dataset) < num_samples:
        return source_dataset
    return source_dataset.shuffle(seed=42).select(range(num_samples))

# Sample data from each source
for source in sources:
    sampled_data[source] = sample_from_source(source)

# Print the number of samples for each source
for source, sampled_ds in sampled_data.items():
    print(f"Source: {source}, Number of samples: {len(sampled_ds)}")
# Function to print first 5 examples of a specific source
def calculate_avg_score_per_source(example_source):
    if example_source in sampled_data:
        example_dataset = sampled_data[example_source]
        print(f"Examples from source: {example_source}")
        scores = []
        for i in range(max(0, len(example_dataset))):
            try:

              example = example_dataset[i]  # Access each example as a dictionary
              input = example['text']
              targets = literal_eval(example['paraphrases'])
              sources = paraphrase(input)
              
              avg_P, avg_R, avg_F1 = calculate_bert_score(sources, targets, lang='en')
              scores.append([avg_P, avg_R, avg_F1])
            except Exception as e:
              print("error")

        return scores
    
# Print examples for each source
for source in sources:
    scores = calculate_avg_score_per_source(source)
    avg_P = sum([score[0] for score in scores]) / len(scores)
    avg_R = sum([score[1] for score in scores]) / len(scores)
    avg_F1 = sum([score[2] for score in scores]) / len(scores)
    print("avg_P: ", avg_P ,"\n", "avg_R: ", avg_R, "\n","avg_F1: ",avg_F1)