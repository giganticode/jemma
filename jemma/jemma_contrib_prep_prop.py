import os, sys, csv, time
import pandas as pd
import subprocess
import argparse

# TLOC - total lines of code
# SLOC - source lines of code /Users/anjandash/Desktop/calculate_java_sloc2.py
# NMTK - num tokens
# NMPR - num parameters

# NUID - num unique identifiers
# NTID - num total identifiers
# CMPX - complexity
# MXIN - max indent

# NMOP - num operators
# NMLT - num literals
# NMRT - num return statements
# NAME - method name

# ------------------------------------
# NMLC - Num of calls with same class [LOCAL] 

# Num of calls with other class in same package [PACKAGE]
# Num of calls with other class in other package [PROJECT]
# Num of api calls [API]


# ------------------------------------

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

keywords_list = list(keywords.keys())
literal_keywords_list = list(literal_keywords.keys())
operators_list = list(operators.keys())
symbols_list = list(symbols.keys())

#### **** ####
grammar_list = keywords_list + literal_keywords_list + operators_list + symbols_list
#### **** ####


def get_mtext(method_id, df_methods):
    df = df_methods[df_methods["method_id"] == method_id]
    if df.shape[0] == 1:
        return df.iloc[0]["method_text"]

def get_mtkns(method_id, df_methods):
    df = df_methods[df_methods["method_id"] == method_id]
    if df.shape[0] == 1:
        return df.iloc[0]["method_tokens"]

def get_mtkn_props(mid, method_tokens):
    mtkns_split = method_tokens.split(",")
    nmtk = len(mtkns_split)

    lit_count = 0
    num_d_quotes = method_tokens.count('\"')
    num_s_quotes = method_tokens.count('\'')

    if num_d_quotes % 2 == 0 and num_s_quotes % 2 == 0:
        lit_count = int((num_d_quotes + num_s_quotes) / 2)

    nmlt = 0
    nmrt = 0 
    nmop = 0
    ntid = 0
    idf_set = set()
    for i, token in enumerate(mtkns_split):
        if (token.startswith('\"') and token.endswith('\"')) or (token.startswith('\'') and token.endswith('\'')):
            nmlt += 1
        if token == "RETURN":
            nmrt += 1
        if token in operators_list:
            nmop += 1
        if token not in grammar_list:
            ntid += 1
            idf_set.add(token)

    if lit_count != nmlt:
        print(f"WARNING: The num literals value may be incorrect!")
        print(f"Please verify for mid:", mid)
        print(f"{lit_count} or {nmlt} ...?")
        print(f"---------------------------")

    return nmtk, len(idf_set), ntid, nmop, nmlt, nmrt

def get_nmpr_props(method_id, methods_csv):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        method_signature =  df.iloc[0]["method_signature"] 
        params = (method_signature[method_signature.rfind("("):method_signature.rfind(")")]).strip()
        if len(params) > 1:
            num_params = ((params.count(",") + 1) if "," in params else 1)
        else:
            num_params = 0
        return num_params

def get_contrib_project_id(method_id, methods_csv):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        return df.iloc[0]["project_id"]

def get_contrib_class_id(project_id, class_name, file_path, classes_csv):
    """
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[(df['project_id'] == project_id) & (df["class_name"] == class_name)]

    if df.shape[0] == 1:
        return df.iloc[0]["class_id"]
    else: 
        for i, row in df.iterrows():
            if file_path.endswith(row["class_path"][10:]):
                return row["class_id"]

def get_contrib_method_id(class_id, method_name, st_ln, en_ln, methods_csv):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[(df['class_id'] == class_id) & (df["method_name"] == method_name)]

    if df.shape[0] == 1:
        return df.iloc[0]["method_id"]
    else:
        for i, row in df.iterrows():
            stln_onfile = row["start_line"]
            enln_onfile = row["start_line"]

            if (int(stln_onfile)) == int(st_ln) and (int(enln_onfile)) == int(en_ln):
                return row["method_id"]
            elif (int(stln_onfile) + 2) >= int(st_ln) and (int(enln_onfile) + 2) >= int(en_ln) and (int(stln_onfile) - 2) <= int(st_ln) and (int(enln_onfile) - 2) <= int(en_ln) :
                return row["method_id"]

def run_cmd(cmd):
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        #print(stdout, stderr)
        process.wait()
    except Exception as e:
        print("ERROR:", str(e))    

def get_metrixpp_props(project_id, project_path, method_id, methods_csv, classes_csv):

    outputs_path = sys.path[0] + "/contrib/tmp/"
    metrixx_path = sys.path[0] + "/dependencies/metrix++/metrixplusplus/metrix++.py"

    output_dbfile  = os.path.join(outputs_path,  project_id + "_.db")
    output_cvfile  = os.path.join(outputs_path,  project_id + "_.csv")
    output_mtfile  = os.path.join(outputs_path,  project_id + "_cmpx.csv")      

    if not os.path.exists(output_cvfile):              
        collect_cmd = ("python2 " + metrixx_path + " collect --db-file=" + output_dbfile + " --std.code.lines.total --std.code.lines.code --std.code.complexity.maxindent --std.code.complexity.cyclomatic -- " + project_path)
        export_cmd  = ("python2 " + metrixx_path + " export --db-file="  + output_dbfile + " > " + output_cvfile + " -- " + project_path)            
        run_cmd(collect_cmd)
        run_cmd(export_cmd)


    # linking method_ids
    target_data = []
    with open(output_cvfile, "r") as rd:
        csv_reader = csv.reader(rd)
        csv_header = next(csv_reader)

        # print(csv_header)
        #['file', 'region', 'type', 'modified', 'line start', 'line end', 'std.code.complexity:cyclomatic', 'std.code.complexity:maxindent', 'std.code.lines:code', 'std.code.lines:total']

        for i, row in enumerate(csv_reader):
            filepath, regionname, regiontype, md, st, en, cyclomatic, maxindent, num_source_lines, total_lines = row
            if regiontype == "function":
                c_name = filepath.split("/")[-1][:-5]
                c_id   = get_contrib_class_id(project_id, c_name, filepath, classes_csv)
                m_id   = get_contrib_method_id(c_id, regionname, st, en, methods_csv)

                if m_id == method_id:
                    return regionname, cyclomatic, maxindent, num_source_lines, total_lines

    return ["ERR", "ERR", "ERR", "ERR", "ERR"]




def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--project_path', '-p', help='path to project', required=True)
    # args = parser.parse_args()    
    project_path = "/Users/anjandash/Desktop/CONTRIB_PROJECTS/MatthewJWalls-anotherGame/"

    meta_meth_path = sys.path[0] + "/contrib/meta_methods.csv"
    meta_clss_path = sys.path[0] + "/contrib/meta_classes.csv"
    repr_text_path = sys.path[0] + "/contrib/representations_text.csv"
    repr_tknb_path = sys.path[0] + "/contrib/representations_tknb.csv"

    with open(sys.path[0] + "/contrib/properties_name.csv", "w+") as w_name, \
        open(sys.path[0] + "/contrib/properties_cmpx.csv", "w+") as w_cmpx,  \
        open(sys.path[0] + "/contrib/properties_mxin.csv", "w+") as w_mxin,  \
        open(sys.path[0] + "/contrib/properties_sloc.csv", "w+") as w_sloc,  \
        open(sys.path[0] + "/contrib/properties_tloc.csv", "w+") as w_tloc,  \
        open(sys.path[0] + "/contrib/properties_nmtk.csv", "w+") as w_nmtk,  \
        open(sys.path[0] + "/contrib/properties_nuid.csv", "w+") as w_nuid,  \
        open(sys.path[0] + "/contrib/properties_ntid.csv", "w+") as w_ntid,  \
        open(sys.path[0] + "/contrib/properties_nmop.csv", "w+") as w_nmop,  \
        open(sys.path[0] + "/contrib/properties_nmlt.csv", "w+") as w_nmlt,  \
        open(sys.path[0] + "/contrib/properties_nmrt.csv", "w+") as w_nmrt,  \
        open(sys.path[0] + "/contrib/properties_nmpr.csv", "w+") as w_nmpr:

        cr_name = csv.writer(w_name)
        cr_name.writerow(["method_id", "method_name"])

        cr_cmpx = csv.writer(w_cmpx)
        cr_cmpx.writerow(["method_id", "cyclomatic_complexity"])        

        cr_mxin = csv.writer(w_mxin)
        cr_mxin.writerow(["method_id", "max_indent"])        

        cr_sloc = csv.writer(w_sloc)
        cr_sloc.writerow(["method_id", "source_lines_of_code"])    

        cr_tloc = csv.writer(w_tloc)
        cr_tloc.writerow(["method_id", "total_lines_of_code"])    

        cr_nmtk = csv.writer(w_nmtk)
        cr_nmtk.writerow(["method_id", "num_tokens"])    

        cr_nuid = csv.writer(w_nuid)
        cr_nuid.writerow(["method_id", "num_unique_identifiers"])    

        cr_ntid = csv.writer(w_ntid)
        cr_ntid.writerow(["method_id", "num_identifiers"])    

        cr_nmop = csv.writer(w_nmop)
        cr_nmop.writerow(["method_id", "num_operators"])    

        cr_nmlt = csv.writer(w_nmlt)
        cr_nmlt.writerow(["method_id", "num_literals"])    

        cr_nmrt = csv.writer(w_nmrt)
        cr_nmrt.writerow(["method_id", "num_returns"])    

        cr_nmpr = csv.writer(w_nmpr)
        cr_nmpr.writerow(["method_id", "num_parameters"])                                                                                


        if os.path.exists(repr_text_path) and os.path.exists(repr_tknb_path) and os.path.exists(meta_meth_path):

            df_methods_text = pd.read_csv(repr_text_path, header=0)
            df_methods_tknb = pd.read_csv(repr_tknb_path, names=["method_id", "method_tokens"]) # pd.read_csv(repr_tknb_path, header=0)

            with open(meta_meth_path, "r") as rd:
                csv_reader = csv.reader(rd)
                csv_header = next(csv_reader)

                mid_id = csv_header.index("method_id")

                for i,row in enumerate(csv_reader):
                    mid = row[mid_id]
                    mtext = get_mtext(mid, df_methods_text)
                    mtkns = get_mtkns(mid, df_methods_tknb)
                    project_id = get_contrib_project_id(mid, meta_meth_path)
                    print("Checking ... #", i, mid)

                    if mtext:
                        nmtk, nuid, ntid, nmop, nmlt, nmrt = get_mtkn_props(mid, mtkns)
                        name, cmpx, mxin, sloc, tloc = get_metrixpp_props(project_id, project_path, mid, meta_meth_path, meta_clss_path)
                        nmpr = get_nmpr_props(mid, meta_meth_path)

                        cr_name.writerow([mid, name])
                        cr_cmpx.writerow([mid, cmpx])
                        cr_mxin.writerow([mid, mxin])
                        cr_sloc.writerow([mid, sloc])
                        cr_tloc.writerow([mid, tloc])
                        cr_nmtk.writerow([mid, nmtk])
                        cr_nuid.writerow([mid, nuid])
                        cr_ntid.writerow([mid, ntid])
                        cr_nmop.writerow([mid, nmop])
                        cr_nmlt.writerow([mid, nmlt])
                        cr_nmrt.writerow([mid, nmrt])
                        cr_nmpr.writerow([mid, nmpr])


if __name__ == "__main__":
    s = time.perf_counter()
    main()
    print(time.perf_counter() - s, "seconds!")


# much of the code is here:
# --------------------------
# /Volumes/AJDRV/DESKTOP_DATA/code/