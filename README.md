<div align="center">

# Summarization

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

</div>

## Project Description

This project is to retrieve top k releated sentences in one meeting transcript for an input category.

## Model Structure

1. Utilize sentence bert model `` to encode each sentence in the meeting transcript into an embedding, build Faiss index.
2. For the input category, catenate the key words and phrases into on query, utilize sentence bert model `` to encode the query into an embeddings and do Faiss similarity search. 

## Data Description

## How to Run

Install dependencies

```bash
# clone project
git clone git@github.com:zhiyuanpeng/GrooveSummerization.git

# download data
cd GrooveSummerization
mkdir data
cd data
download Anonymized Transcripts to ./ as AnonymizedTranscripts

# create conda environment
conda env create -f environment.yml
```
Run the script

```bash
python semantic.py
```

Example

```bash
# meeting id

meeting id is the the number in the file name, like meeting id of Meeting4.txt is 4

# choosing category from [next_steps, questions, pain_points, competitors, pricing, differentiation] will concatenate the words and phrases in predefined categories as a query
Meeting id:4
Category:next_steps
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.87it/s]
Semantic Retrieval 63 with score 0.56: yep so for us our scheduler links if you re like hey Catherine heres the time me my colleague heres times that my colleagues and I are available or heres a link to our entire calendars book a time that works for you.
>>>>>>>>Has key phrases: schedule
Semantic Retrieval 181 with score 0.54: Going into the email tools here when it comes to the scheduler functionalities so out of the gate you can see Okay, you can select a time zone, if you want.
>>>>>>>>Has key phrases: schedule
Semantic Retrieval 209 with score 0.54: Right and so, then I can go in and say Okay, I want this to display and Eastern time thats my time zone, you can pick it to be different and then select meetings insert times and then again, you would see that.
>>>>>>>>Has key phrases: meeting

# set category as exit will exit the program
Meeting id:4
Category:exit

# choosing other categories like "write an email to your advisor" will utilize "write an email to your advisor" as query
Meeting id:4
Category:write an email to your advisor
```
## Model Performance

Wait for human labeled data to evaluate it.
