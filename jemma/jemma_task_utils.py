"""
jemma_task_utils

author: @anjandash
license: MIT
"""

import sys
import torch
import javalang
import configparser

import numpy as np
import pandas as pd
import jemma_utils as ju

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from subprocess import Popen, PIPE
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy  = accuracy_score(y_true=labels, y_pred=pred)
    recall    = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1_score_ = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1_score_}       

def print_eval_scores(labels, pred, save_path):
    accuracy  = accuracy_score(y_true=labels, y_pred=pred)
    recall    = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1_score_ = f1_score(y_true=labels, y_pred=pred, average='macro')

    # with open(save_path + "/results.txt", "w+") as f:
    #     output = "accuracy:" + accuracy + "\nf1_score:" + f1_score_ + "\nrecall:" + recall + "\nprecision:" + precision
    #     f.write(output)

    print("\n**********************")
    print("\nACCURACY: ", accuracy)
    print("\nF1 SCORE: ", f1_score_)
    print("\nRECALL:   ", recall)
    print("\nPRECISION:", precision)
    print("\n**********************")     



metric_headers = {
    "SLOC": "source_lines_of_code",
    "NMRT": "num_returns",
    "MXIN": "max_indent",
    "NMLT": "num_literals",
    "CMPX": "cyclomatic_complexity",
    "NUID": "num_unique_identifiers",
    "NMOP": "num_operators",
    "NAME": "method_name",
    "NMTK": "num_tokens",
    "NTID": "num_identifiers",
    "NMPR": "num_parameters",
    "TLOC": "total_lines_of_code",
}

representation_headers = {
    "TKNB": "method_tokens",
    "TKNA": "method_tokens_spaced",
    "TEXT": "method_text",
    "FTGR": "method_ftgr",
    "C2VC": "method_c2vc",
    "C2SQ": "method_c2sq",
    "ASTS": "method_asts",
}



"""
# ----------------------
#  NOTE: Available dicts
# ----------------------
# keywords          = {}
# literal_keywords  = {}
# operators         = {}
# symbols           = {}
# ----------------------
"""

keywords = {
    'ABSTRACT':     'abstract',
    'ASSERT':       'assert',
    'BOOLEAN':      'boolean',
    'BREAK':        'break',
    'BYTE':         'byte',
    'CASE':         'case',
    'CATCH':        'catch',
    'CHAR':         'char',
    'CLASS':        'class', 
    'CONST':        'const',
    'CONTINUE':     'continue',

    'DEFAULT':      'default',
    'DO':           'do',
    'DOUBLE':       'double',
    'ELSE':         'else',
    'ENUM':         'enum',
    'EXTENDS':      'extends',
    'FINAL':        'final',
    'FINALLY':      'finally',
    'FLOAT':        'float',
    'FOR':          'for',
    'GOTO':         'goto',

    'IF':           'if',
    'IMPLEMENTS':   'implements',
    'IMPORT':       'import',
    'INSTANCEOF':   'instanceof',
    'INT':          'int',
    'INTERFACE':    'interface',

    'LONG':         'long',
    'NATIVE':       'native',
    'NEW':          'new',
    'PACKAGE':      'package', 
    'PRIVATE':      'private',
    'PROTECTED':    'protected',
    'PUBLIC':       'public',
    'RETURN':       'return',

    'SHORT':        'short',
    'STATIC':       'static',
    'STRICTFP':     'strictfp',
    'SUPER':        'super',
    'SWITCH':       'switch',
    'SYNCHRONIZED': 'synchronized',

    'THIS':         'this',
    'THROW':        'throw',
    'THROWS':       'throws',
    'TRANSIENT':    'transient',
    'TRY':          'try',
    'VOID':         'void',
    'VOLATILE':     'volatile',
    'WHILE':        'while',

    # 'UNDERSCORE':   '_',          # EXCLUDED! not sure if the grammar supports it
    # 'NONSEALED':    'non-sealed', # EXCLUDED! not sure if the grammar supports it
    #  ------------   ------------  # EXCLUDED! String is not a Java keyword, String is a class
}

literal_keywords = {
    'TRUE':     'true',
    'FALSE':    'false',
    'NULL':     'null',
}

operators = {
    'PLUS':     '+',
    'SUB':      '-',
    'STAR':     '*',
    'SLASH':    '/',
    'PERCENT':  '%',
    'AMP':      '&',
    'BAR':      '|',
    'CARET':    '^',
    'BANG':     '!',
    'EQ':       '=',
    
    'PLUSEQ':   '+=',
    'SUBEQ':    '-=',
    'STAREQ':   '*=',
    'SLASHEQ':  '/=',
    'PERCENTEQ':'%=',
    'AMPEQ':    '&=',
    'BAREQ':    '|=',
    'CARETEQ':  '^=',
    'BANGEQ':   '!=',
    'EQEQ':     '==',

    'LT':       '<',
    'GT':       '>',
    'LTEQ':     '<=',
    'GTEQ':     '>=',
    'LTLT':     '<<',
    'GTGT':     '>>',
    'GTGTGT':   '>>>',
    'LTLTEQ':   '<<=',
    'GTGTEQ':   '>>=',
    'GTGTGTEQ': '>>>=',

    'PLUSPLUS': '++',
    'SUBSUB':   '--',
    'AMPAMP':   '&&',
    'BARBAR':   '||',        

    'QUES':     '?',
    'COLON':    ':',   
    'TILDE':    '~',
    'ARROW':    '->',   

    'INSTANCEOF':'instanceof',
    'COLONCOLON':'::', 

}

symbols = {
    'DOT':      '.',
    'COMMA':    ',',
    'SEMI':     ';',    
    'LPAREN':   '(',
    'RPAREN':   ')',
    'LBRACE':   '{',
    'RBRACE':   '}',
    'LBRACKET': '[',
    'RBRACKET': ']',

    'MONKEYS_AT':'@',
    'ELLIPSIS':  '...', 
}


keywords_reverse = {v:k for k, v in keywords.items()}
literal_keywords_reverse = {v:k for k, v in literal_keywords.items()}
operators_reverse = {v:k for k, v in operators.items()}
symbols_reverse = {v:k for k, v in symbols.items()}


def get_replaced_token(code_token):
    if code_token.upper() in keywords_reverse.keys():
        return keywords_reverse[code_token.upper()]
    elif code_token.upper() in literal_keywords_reverse.keys():
        return literal_keywords_reverse[code_token.upper()]
    elif code_token.upper() in operators_reverse.keys():
        return operators_reverse[code_token.upper()]
    elif code_token.upper() in symbols_reverse.keys():
        return symbols_reverse[code_token.upper()]
    return code_token


def get_actual_token(code_token):
    if code_token.upper() in keywords.keys():
        return keywords[code_token.upper()]
    elif code_token.upper() in literal_keywords.keys():
        return literal_keywords[code_token.upper()]
    elif code_token.upper() in operators.keys():
        return operators[code_token.upper()]
    elif code_token.upper() in symbols.keys():
        return symbols[code_token.upper()]
    return code_token


# ********** #
#            #
# ********** #


def gen_TKNA_from_method_text(method_id, method_text):
    tokens = list(javalang.tokenizer.tokenize(method_text))
    tokens = [str(t) for t in tokens]
    tokens = [t[t.index('"')+1:t.rindex('"')] for t in tokens]
    return " ".join(tokens)


def gen_TKNB_from_method_text(method_id, method_text):
    tokens = list(javalang.tokenizer.tokenize(method_text))
    tokens = [str(t) for t in tokens]
    tokens = [t[t.index('"')+1:t.rindex('"')] for t in tokens]
    tokens = [get_replaced_token(t) for t in tokens]
    return ",".join(tokens)


def gen_C2VC_from_method_text(method_id, method_text):
    with open(sys.path[0] + "/dependencies/code2vec/Input.java", "w+") as f:
        method_lines = method_text.split('\n')
        for line in method_lines:
            f.write(line + "\n")

    command = "python3 " + sys.path[0] + "/dependencies/code2vec/JavaExtractor/extract.py --file " + sys.path[0] + "/dependencies/code2vec/Input.java --max_path_length 8 --max_path_width 2 --num_threads 8 --jar " + sys.path[0] + "/dependencies/code2vec/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar" 
    process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)  
    outputs = process.communicate()[0].decode('utf-8')

    try:
        return (outputs.split(" ", 1)[1])
    except:
        # raise Error
        print(outputs)
        print("_WARNING_ There was a problem while generating code2vec representation for", method_id)
        print("_WARNING_ A dummy code2vec representation has been generated instead, to avoid problems. Please check whether the method text is suitable AST path extraction:", method_id)
        return "METHOD_NAME,0,METHOD_NAME"


def gen_C2SQ_from_method_text(method_id, method_text):
    with open(sys.path[0] + "/dependencies/code2seq/Input.java", "w+") as f:
        method_lines = method_text.split('\n')
        for line in method_lines:
            f.write(line + "\n")

    command = "python3 " + sys.path[0] + "/dependencies/code2seq/JavaExtractor/extract.py --file " + sys.path[0] + "/dependencies/code2seq/Input.java --max_path_length 8 --max_path_width 2 --num_threads 8 --jar " + sys.path[0] + "/dependencies/code2seq/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar" 
    process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)  
    outputs = process.communicate()[0].decode('utf-8')

    try:
        return (outputs.split(" ", 1)[1])
    except:
        # raise Error
        print("_WARNING_ There was a problem while generating code2seq representation for", method_id)
        print("_WARNING_ A dummy code2seq representation has been generated instead, to avoid problems. Please check whether the method text is suitable AST path extraction:", method_id)        
        return "METHOD_NAME,0,METHOD_NAME"


def gen_FTGR_from_method_text(method_id, method_text): # TODO
    class_id   = ju.get_class_id_by_method_id(method_id)
    class_name = ju.get_class_name(class_id)
    class_path = ju.get_class_path(class_id)

    # get class_text by reading the <class_name>.java file
    # create (or LOCATE) .proto file
    # spawn nodes (run spawnNodesDb*)
    pass


def gen_REPR_from_method_text(method_id, method_text): 
    pass


def gen_representation(representation, method_id, custom_method_text):

    masked_text = custom_method_text 

    if   representation == "TKNA":
        return gen_TKNA_from_method_text(method_id, custom_method_text)
    elif representation == "TKNB":
        return gen_TKNB_from_method_text(method_id, custom_method_text)
    elif representation == "C2VC":
        return gen_C2VC_from_method_text(method_id, custom_method_text)
    elif representation == "C2SQ":
        return gen_C2SQ_from_method_text(method_id, custom_method_text)
    # elif representation == "FTGR":
    #     return gen_FTGR_from_method_text(method_id, custom_method_text)



# ********** #
#            #
# ********** #

def get_metrics(metric, method_ids):

    mt = sys.path[0] + "/jemma_datasets/properties/Jemma_Properties_Methods_" + metric + ".csv"
    df = pd.read_csv(mt, header=0)
    df = df[df["method_id"].isin(method_ids)]
    return df

def get_properties(property, method_ids):

    mt = sys.path[0] + "/jemma_datasets/properties/Jemma_Properties_Methods_" + property + ".csv"
    df = pd.read_csv(mt, header=0)
    df = df[df["method_id"].isin(method_ids)]
    return df    


def get_representations(representation, method_ids):
    mt = sys.path[0] + "/jemma_datasets/representations/Jemma_Representations_Methods_" + representation + ".csv"
    df = pd.read_csv(mt, header=0)
    df = df[df["method_id"].isin(method_ids)]
    return df


def run_model(df_train, df_test, train_model_name, label_head, reprs_head):

        # finetune *** text-classification *** 

        config = configparser.ConfigParser()
        config.read(sys.path[0] + "/tmp/config/config.ini")           

        train_data = df_train
        test_data  = df_test

        X = list(train_data[reprs_head])
        y = list(train_data[label_head])

        tokenizer = AutoTokenizer.from_pretrained(train_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(train_model_name, num_labels=len(set(y)))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
        X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

        train_dataset = Dataset(X_train_tokenized, y_train)
        val_dataset = Dataset(X_val_tokenized, y_val)        

        save_path = sys.path[0] + "/tmp/models/finetuned_" + train_model_name[train_model_name.find("/")+1:] + "_" + label_head[label_head.find("/")+1:]
        args = TrainingArguments(
            output_dir=save_path,
            seed=config.getint("train", "seed"),
            evaluation_strategy=config.get("train", "evaluation_strategy"),
            save_strategy=config.get("train", "save_strategy"),
            eval_steps=config.getint("train", "eval_steps"),
            save_steps=config.getint("train", "save_steps"),
            per_device_train_batch_size=config.getint("train", "per_device_train_batch_size"),
            per_device_eval_batch_size=config.getint("train", "per_device_eval_batch_size"),
            num_train_epochs=config.getint("train", "num_train_epochs"),
            load_best_model_at_end=config.getboolean("train", "load_best_model_at_end"))


        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

        trainer.train()

        # ********************** #    
        
        X_test = list(test_data[reprs_head])
        X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
        test_dataset = Dataset(X_test_tokenized)    
        raw_pred, _, _ = trainer.predict(test_dataset)       # Make prediction
        y_pred = np.argmax(raw_pred, axis=1)                 # Preprocess raw predictions

        df = pd.DataFrame(y_pred)
        df.to_csv(save_path + "/eval_pred.csv", index=False, header=["pred_label"])
        eval_data = pd.read_csv(save_path + "/eval_pred.csv", header=0)
        print_eval_scores(test_data["label"], eval_data["pred_label"], save_path)    

        return ""    




def run_models(metric, representation, train_methods, test_methods, models):
    """
    """
    method_ids = train_methods + test_methods
    label_head = metric_headers.get(metric, "NOT FOUND")
    reprs_head = representation_headers.get(representation, "NOT FOUND")

    dm = get_metrics(metric, method_ids)
    dr = get_representations(representation, method_ids)

    df = pd.merge(dr, dm, on="method_id")
    df_train = df[df["method_id"].isin(train_methods)]
    df_test  = df[df["method_id"].isin(test_methods)]    

    for model in models:
        model_res = run_model(df_train, df_test, model, label_head, reprs_head)
        print(model)
        print("------------------")
        print("Training results: ")
        print("------------------")
        print(model_res)
        print("------------------", end="\n\n")        


if __name__ == "__main__":
    
    dfm = pd.read_csv(sys.path[0]+"/jemma_datasets/properties/Jemma_Properties_Methods_CMPX.csv", header=0)
    dfm = dfm[dfm["cyclomatic_complexity"] < 5]

    dfx = dfm.head(10000)
    dfx = dfx.sample(frac=1).reset_index(drop=True)

    train_methods = dfx.head(8000)["method_id"].tolist()
    test_methods  = dfx.tail(2000)["method_id"].tolist()
    run_models("CMPX", "TKNA", train_methods, test_methods, ["bert-base-uncased"])