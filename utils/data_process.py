#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   data_process.py
@Time    :   2023/03/02 11:03:26
@Author  :   Zhiyuan Peng
@Version :   1.0
@Contact :   zpeng@scu.edu
@License :   (C)Copyright 2020-2023 Zhiyuan Peng@Santa Clara University
@Desc    :   Split the transcripts into sentence and store in pands datafram
'''
import os
from os.path import join, basename
import glob
import pandas as pd

cwd = os.getcwd()
transcript_folder = join(cwd, "data/AnonymizedTranscripts/*.txt")

def read_transcript(file_path):
    """read one transcript into the fomat:
    (doc_id, sentence_id, name, sentence)

    Args:
        file_path (_type_): _description_
    """
    with open(file_path) as f:
        acc = 0
        file_name = basename(file_path)
        doc_id = int(file_name.strip(".txt")[7:])
        sent_id, name, sent, start, end = 0, "", "", "", ""
        doc_ids, sent_ids, names, sents, starts, ends = [], [], [], [], [], []
        for line in f:
            if line == "\n":
                acc = 0
                doc_ids.append(doc_id)
                sent_ids.append(sent_id)
                names.append(name)
                starts.append(start)
                ends.append(end)
                sents.append(sent)
            else:
                if acc == 0:
                    sent_id = int(line.strip("\n"))
                if acc == 1:
                    start, end = line.strip("\n").split(" --> ")
                if acc == 2:
                    name, sent = line.strip("\n").split(": ")
                    sent = sent.strip('"')
                acc += 1
    return (doc_ids, sent_ids, names, starts, ends, sents)

def docs2dataframe(transcript_folder, to_path):
    """read all docs to a single dataframe

    Args:
        transcript_folder (_type_): _description_
        to_path (_type_): _description_
    """
    doc_ids, sent_ids, names, starts, ends, sents  = [], [], [], [], [], []
    for file_path in glob.glob(transcript_folder):
        f_doc_ids, f_sent_ids, f_names, f_starts, f_ends, f_sents = read_transcript(file_path)
        doc_ids.extend(f_doc_ids)
        sent_ids.extend(f_sent_ids) 
        names.extend(f_names)
        starts.extend(f_starts)
        ends.extend(f_ends) 
        sents.extend(f_sents)
    combo = zip(doc_ids, sent_ids, names, starts, ends, sents)
    combo = sorted(combo, key= lambda x: (x[0], x[1]))
    doc_ids, sent_ids, names, starts, ends, sents = zip(*combo)
    data = pd.DataFrame.from_dict({"doc_id": doc_ids, "sent_id": sent_ids, "name": names, 
                                   "start": starts, "end": ends, "sent": sents})
    data.to_csv(to_path)

def doc2dataframe(transcript_folder):
    """read each single doc to a dataframe

    Args:
        transcript_folder (_type_): _description_
        to_path (_type_): _description_
    """
    for file_path in glob.glob(transcript_folder):
        f_doc_ids, f_sent_ids, f_names, f_starts, f_ends, f_sents = read_transcript(file_path)
        data = pd.DataFrame.from_dict({"doc_id": f_doc_ids, "sent_id": f_sent_ids, "name": f_names, 
                                    "start": f_starts, "end": f_ends, "sent": f_sents})
        data.to_csv(file_path.replace(".txt", ".csv"))

def main():
    to_path = join(cwd, "data", "transcripts.csv")
    # docs2dataframe(transcript_folder, to_path)
    doc2dataframe(transcript_folder)

if __name__ == "__main__":
    main()





