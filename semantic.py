#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   semantic.py
@Time    :   2023/03/02 10:44:29
@Author  :   Zhiyuan
@Version :   1.0
@Contact :   zpeng@scu.edu
@License :   (C)Copyright 2020-2021, Zhiyuan Peng@SCU
@Desc    :   semantic search to retrieve sentences for each category
'''
import pandas as pd
from os.path import join, exists, basename
import os
import glob
from sentence_transformers import SentenceTransformer
from utils.data_process import read_transcript
from tqdm import tqdm
import faiss
import numpy as np

cwd = os.getcwd()
transcript_folder = join(cwd, "data/AnonymizedTranscripts")
next_steps = ["calendar invite", "call you", "catch up with", "central time", "eastern time", 
              "eight o'clock", "eleven o'clock", "five o'clock", "follow up", "four o'clock", 
              "get back to you", "get that out to you", "get this out to you", "go forward", 
              "in a few weeks", "in a week", "in two weeks", "meeting", "morning", "mountain time", 
              "moving forward", "my calendar", "my schedule", "next friday", "next monday", 
              "next step", "next steps", "next thursday", "next tuesday", "next wednesday", 
              "next week", "nine o'clock", "on the calendar", "one o'clock", "pacific time", 
              "reach out to", "reconnect", "schedule", "send you", "send you an email", 
              "set up a call", "seven o'clock", "six o'clock", "ten o'clock"]
questions = ["question", "can you get back to me", "?"]
pain_points = ["aren't appearing", "aren't showing", "did not work", "didn't work", "do not appear", 
               "does not work", "doesn't work", "don't appear", "isn't working", "issues with", 
               "not appearing", "not showing", "not working", "troubleshoot", "troubleshooting", 
               "Having a hard time", "We don’t like", "Challenging" , "Challenges", "annoying"]
competitors= ["cirrus", "outreach", "revenuegrid", "salesforceiq", "salesloft", "toutapp", 
              "yesware", "Gong", "loft", "sales loft", "Chorus"]
pricing = ["ballpark figure", "ballpark pricing", "charge us", "charge you", "cheaper", 
           "commercial plan", "cost", "discount", "dollar", "dollars", "expensive", "fee", 
           "good price", "grand", "hundred a month", "hundred dollars", "inexpensive", "license", 
           "license fee", "list price", "list pricing", "more expensive", "my pricing", "our price", 
           "our pricing", "paying", "per month", "per seat", "per server", "per user", "plan", 
           "price", "pricing", "sharpen the pencil", "standard price", "thousand a month", 
           "thousand dollars"]
differentiation = ["additional vendors", "alternative solutions", "competition", 
                   "competitive landscape", "competitor", "competitors", "differentiate", 
                   "differentiator", "differentiators", "evaluation process", 
                   "exploring alternatives", "look at alternatives", "look for alternatives", 
                   "look into alternatives", "looking at alternatives", "looking for alternatives", 
                   "looking into alternatives", "other solution", "other solutions", "other tools", 
                   "other vendors", "our competitors", "short list", "stack up", "stacks up", 
                   "versus the alternative", "salesforce native"]

class semantic_search():

    def __init__(self, transcript_folder):
        self.transcript_folder = transcript_folder
        self.meetings_pd = self.load_csv()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="cuda")
        self.meetings_index = self.load_index()
    
    def load_csv(self):
        print("Start loading csv files")
        meetings_pd = {}
        for txt_file in tqdm(glob.glob(join(self.transcript_folder, "*.txt"))):
            doc_id = basename(txt_file).strip(".txt")[7:]
            csv_file = txt_file.replace(".txt", ".csv")
            if not exists(csv_file):
                f_doc_ids, f_sent_ids, f_names, f_starts, f_ends, f_sents = read_transcript(csv_file)
                data = pd.DataFrame.from_dict({"doc_id": f_doc_ids, "sent_id": f_sent_ids, "name": f_names, 
                                    "start": f_starts, "end": f_ends, "sent": f_sents})
                data.to_csv(csv_file)
            meeting = pd.read_csv(csv_file)
            meetings_pd[doc_id] = meeting
        return meetings_pd
    
    def load_index(self):
        print("Start loading index files")
        meetings_index = {}
        for txt_file in tqdm(glob.glob(join(self.transcript_folder, "*.txt"))):
            doc_id = basename(txt_file).strip(".txt")[7:]
            index_file = txt_file.replace(".txt", ".index")
            if not exists(index_file):
                df = pd.read_csv(txt_file.replace(".txt", ".csv"))
                embeddings = self.model.encode(df["sent"], show_progress_bar=True, 
                                               convert_to_numpy=True, normalize_embeddings=True)
                num, dim = embeddings.shape
                faiss_indexs = faiss.IndexFlatIP(dim)
                faiss_indexs = faiss.IndexIDMap(faiss_indexs)
                faiss_indexs.add_with_ids(embeddings, np.array([i for i in range(num)]))
                faiss.write_index(faiss_indexs, index_file)
            else:
                faiss_indexs = faiss.read_index(index_file)
            meetings_index[doc_id] = faiss_indexs
        return meetings_index

    def search(self, topic, query, meeting_id, topk, show=True):
        sentences = []
        q_embedding = self.model.encode(query, show_progress_bar=True, 
                                        convert_to_numpy=True, normalize_embeddings=True)
        index = self.meetings_index[meeting_id]
        dists, ids = index.search(q_embedding.reshape(1, -1), k=topk)
        for ii, i in enumerate(ids[0]):
            sent = self.meetings_pd[meeting_id].iloc[[i]].iloc[0]["sent"]
            if show:
                score = str(round(dists[0][ii], ndigits=2))
                print(f"Semantic Retrieval {i} with score {score}: {sent}")
                for phrase in topic:
                    if phrase in sent:
                        print(f">>>>>>>>Has key phrases: {phrase}")
            sentences.append(sent)
        return sent

def main():
    topk = 3
    groove = semantic_search(transcript_folder)
    while True:
        meeting_id = input("Meeting id:")
        category = input("Category:")
        if category == "next_steps":
            query = " ".join(next_steps)
            topic = next_steps
        elif category == "questions":
            query = " ".join(questions)
            topic = questions
        elif category == "pain_points":
            query = " ".join(pain_points)
            topic = pain_points
        elif category == "competitors":
            query = " ".join(competitors)
            topic = competitors
        elif category == "pricing":
            query = " ".join(pricing)
            topic = pricing
        elif category == "differentiation":
            query = " ".join(differentiation)
            topic = differentiation
        elif category == "exit":
            break
        else:
            query = category
        summarization = groove.search(topic, query, meeting_id, topk)

if __name__ == "__main__":
    main()

