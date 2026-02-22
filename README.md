

# About the project
This project implements a paraphrasing model using the [T5 base](https://huggingface.co/google-t5/t5-base) model with transfer learning. The model is trained on a diverse dataset containing questions and sentences from various sources to generate high-quality paraphrases.

Dataset

The [dataset](https://huggingface.co/datasets/humarin/chatgpt-paraphrases) comprises a total of 419,197 entries from three different sources:

    Quora Dataset: 247,138 questions
    Squad 2.0 Dataset: 91,983 texts
    CNN News Dataset: 80,076 texts

## Dataset Structure

The dataset is organized in a CSV file with the following columns:

    text: An original sentence or question from the datasets.
    paraphrases: A list of 5 paraphrases for each text.
    category: Indicates whether the entry is a question or a sentence.
    source: Specifies the source of the text (quora, squad_2, cnn_news).



## Instructions for training the model

1. Clone the repository <br>
```git clone https://github.com/sanjay931/text-paraphraser```

2. Checkout to dev branch <br>
```git checkout dev```

3. Create an environment of your choice and activate it. <br>

4. Navigate to the directory <br>
```cd text-paraphraser```

5. Install the requirements <br>
```pip install -r requirements.txt``` 

To train the model, create a folder ```src/data``` and add the csv file of data available [here](https://huggingface.co/datasets/humarin/chatgpt-paraphrases). <br>

6. To start training, navigate to ```src``` directory and run the following command: <br>
```python3 train.py``` <br>

This will save the trained model at ```src/trained_model``` <br>

7. Add HUGGINGFACE_TOKEN to ```.env``` file. <br>

8. Upload the model to HuggingFace so that it becomes accessible from anywhere. <br>
```python3 upload_to_hub.py```

9. To run the inference, use the script file ```inference.py```. <br>
```python3 inference.py```

