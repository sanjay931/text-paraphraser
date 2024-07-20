# text-paraphraser
Paraphraser tool that paraphrases news, questions and short documents

# About the project
This project implements a paraphrasing model using the T5 base model with transfer learning. The model is trained on a diverse dataset containing questions and sentences from various sources to generate high-quality paraphrases.

Dataset

The dataset comprises a total of 419,197 entries from three different sources:

    Quora Dataset: 247,138 questions
    Squad 2.0 Dataset: 91,983 texts
    CNN News Dataset: 80,076 texts

Dataset Structure

The dataset is organized in a CSV file with the following columns:

    text: An original sentence or question from the datasets.
    paraphrases: A list of 5 paraphrases for each text.
    category: Indicates whether the entry is a question or a sentence.
    source: Specifies the source of the text (quora, squad_2, cnn_news).
