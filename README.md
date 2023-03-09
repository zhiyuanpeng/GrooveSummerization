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
git clone https://github.com/zhiyuanpeng/
cd EmailEfficiency

# [OPTIONAL] create conda environment
conda env create -f environment.yml

## Model Performance

Wait for human labeled data to evaluate it.
