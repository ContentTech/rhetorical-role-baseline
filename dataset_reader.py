import json
import os

MAX_DOC_LENGTH = 512
# MAX_DOC_LENGTH = 3072

class InputDocument:
    """Represents a document that consists of sentences and a label for each sentence"""
    
    def __init__(self, sentences, labels, auxiliary_labels, doc_name, start_ids, end_ids):
        """sentences: array of sentences labels: array of labels for each sentence """
        self.sentences = sentences
        self.labels = labels
        self.auxiliary_labels = auxiliary_labels
        self.doc_name = doc_name
        self.start_ids = start_ids
        self.end_ids = end_ids

    def get_sentence_count(self):
        return len(self.sentences)



class DocumentsDataset:
    def __init__(self, path, max_docs=-1):
        self.path = path    
        self.length = None
        self.max_docs = max_docs
    
    #Adapter functions for Iterator 
    def __iter__(self):
        return self.readfile()
    
    def __len__(self):
        return self.calculate_len()    
    
    
    def calculate_len(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length
    
    def readfile(self):
        """Yields InputDocuments """
        read_docs = 0
        with open(self.path, encoding="utf-8") as f:
            sentences, tags, auxiliary_tags = [], [], []
            start_ids, end_ids = [],[]
            prev_tag = None
            doc_name=''
            for line in f:
                new_doc = False
                if self.max_docs >= 0 and read_docs >= self.max_docs:
                    return
                line = line.strip()
                if not line:
                    if len(sentences) != 0:
                        if len(sentences) > MAX_DOC_LENGTH:
                            print(f"The length of doc {doc_name} exceeds {MAX_DOC_LENGTH}.")
                        else:
                            end_ids[-1] = prev_tag
                            read_docs += 1
                            yield InputDocument(sentences, tags, auxiliary_tags, doc_name, start_ids, end_ids)
                        sentences, tags, lsp_tags = [], [], []
                        start_ids, end_ids = [],[]
                        prev_tag = None
                        doc_name = ''
                elif not line.startswith("###"):
                    ls = line.split("\t")
                    if len(ls) < 2:
                        continue
                    elif len(ls) > 2:
                        tag, auxiliary_tag, sentence = ls[0], ls[1], ls[2]
                    else:
                        tag, auxiliary_tag, sentence = ls[0], "mask", ls[1]

                    if prev_tag is not None and prev_tag != tag:
                        end_ids[-1] = prev_tag
                        start_ids.append(tag)
                        end_ids.append("mask")
                    else:
                        if prev_tag is None:
                            start_ids.append(tag)
                        else:
                            start_ids.append("mask")
                        end_ids.append("mask")

                    sentences += [sentence]
                    tags += [tag]
                    auxiliary_tags += [auxiliary_tag]
                    prev_tag = tag

                elif line.startswith("###"):
                    doc_name = line.replace("###","").strip()

