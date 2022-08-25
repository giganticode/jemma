import os
import sys
import csv 
import time
import uuid
import json
import argparse
import javalang
import jsonlines
import subprocess
import pandas as pd
import concurrent.futures
import codeprep.api.text as cp

from graph_pb2 import Graph
from graph_pb2 import FeatureNode
from graph_pb2 import FeatureEdge

from jemma_task_utils import gen_C2VC_from_method_text
from jemma_task_utils import gen_C2SQ_from_method_text

'''
# ----------------------
#  NOTE: Available dicts
# ----------------------
# keywords          = {}
# operators         = {}
# symbols           = {}
# literal_keywords  = {}
# ----------------------

'''

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

relational_operators = {
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
}

keywords = {v:k for k,v in keywords.items()}
literal_keywords = {v:k for k,v in keywords.items()}
operators = {v:k for k,v in operators.items()}
symbols = {v:k for k,v in symbols.items()}


def get_normalized_token(code_token):
    if code_token.strip() in keywords.keys():
        return keywords[code_token.strip()]
    elif code_token.strip() in literal_keywords.keys():
        return literal_keywords[code_token.strip()]
    elif code_token.strip() in operators.keys():
        return operators[code_token.strip()]
    elif code_token.strip() in symbols.keys():
        return symbols[code_token.strip()]
    return code_token

def get_contrib_project_id(method_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        return df.iloc[0]['project_id']
    return None   

def get_contrib_class_id(method_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        return df.iloc[0]['class_id']
    return None       

def get_contrib_method_path(method_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        project_path_local = "/".join(project_path.split("/")[:-2])
        method_path_onfile = "/".join((df.iloc[0]['method_path']).split("/")[2:])
        actual_method_path = os.path.join(project_path_local, method_path_onfile)

        return actual_method_path
    return None    

def get_contrib_start_line(method_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        return df.iloc[0]['start_line']
    return None    

def get_contrib_end_line(method_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        return df.iloc[0]['end_line']
    return None

def runcmd(cmd):
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        #print(stdout, stderr)
        process.wait()
    except Exception as e:
        print("ERROR:", str(e))    

def get_all_class_paths(project_path):
    project_path = project_path
    fullclasspath = ''
    for root, dirs, files in os.walk(project_path):
        for edir in dirs:
            fullclasspath = fullclasspath + ':' + os.path.join(root, edir) + '/'
    return fullclasspath

def get_all_jar_paths(project_path):
    project_path = project_path
    fulljarpath = ''
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.startswith("."):
                continue
            if file.endswith(".jar"):
                fulljarpath = fulljarpath + ':' + os.path.join(root, file)
    return fulljarpath

def make_proto(project_path, jar_path, class_path):
    extractor_path  = sys.path[0] + '/dependencies/featgraph/features-javac-master/extractor/target/features-javac-extractor-1.0.0-SNAPSHOT-jar-with-dependencies.jar'
    all_class_paths = get_all_class_paths(project_path)
   
    command  = f'javac -cp \"{extractor_path + jar_path + all_class_paths}\" -Xplugin:FeaturePlugin \"{class_path}\"'
    # print("runnning ... ")
    # print(command2)

    runcmd(command)

def get_stps_enps(method_id, method_path):
    start_line = get_contrib_start_line(method_id)
    end_line   = get_contrib_end_line(method_id)

    with open(method_path, "r") as rd:
        lines = rd.readlines()

        tot_pos = 0
        start_pos = 0
        end_pos = 0
        for i, line in enumerate(lines):

            if (i+1) == start_line:
                start_blocks = len(line) - len(line.lstrip())
                start_pos = tot_pos + start_blocks

            if (i+1) == end_line:
                end_blocks = len(line)
                end_pos = tot_pos + end_blocks
                break
            tot_pos += len(line)
    
        return start_pos, end_pos

def getMethods(pId, cId, mId, df_mount):
    mrows = df_mount[df_mount['method_id'] == mId]
    return mrows

def getTypeText(val):
    if val == 1:
        return "TOKEN"
    elif val == 2:
        return "AST_ELEMENT"
    elif val == 3:
        return "COMMENT_LINE"
    elif val == 4:
        return "COMMENT_BLOCK"
    elif val == 5:
        return "COMMENT_JAVADOC"
    elif val == 6:
        return "AST_ROOT"
    elif val == 7:
        return "IDENTIFIER_TOKEN"
    elif val == 8:
        return "FAKE_AST"
    elif val == 9:
        return "SYMBOL"
    elif val == 10:
        return "SYMBOL_TYP"
    elif val == 11:
        return "SYMBOL_VAR"
    elif val == 12:
        return "SYMBOL_MTH"
    elif val == 13:
        return "TYPE"
    elif val == 14:
        return "METHOD_SIGNATURE"
    elif val == 15:
        return "AST_LEAF"

def listToString(ls):
    if ls == []:
        return None
    else:
        return (','.join(str(t) for t in ls))    

def populateNodes(pId, cId, mId, protopath, df_mount):
    in_nodes = {}
    with open(protopath, "rb") as f:

        try:
            g = Graph()
            g.ParseFromString(f.read())
        except Exception as e:
            print(str(e))

        SETOutput_method = sys.path[0] + '/dependencies/featgraph/nodes/'+ mId +'_.csv'
        with open(SETOutput_method, 'a+') as appendfile:
            csvAppender = csv.writer(appendfile)
            csvAppender.writerow(["project_id", "class_id", "method_id", "statement_id", "node_id", "dot_id", "type", "contents", "start_line", "end_line", "start_pos", "end_pos", "allIncoming", "allOutgoing", "CHIncoming", "CHOutgoing", "NTIncoming", "NTOutgoing", "LUIncoming", "LUOutgoing", "LWIncoming", "LWOutgoing", "CFIncoming", "CFOutgoing", "LLIncoming", "LLOutgoing", "RTIncoming", "RTOutgoing", "FAIncoming", "FAOutgoing", "GBIncoming", "GBOutgoing", "GNIncoming", "GNOutgoing"]) 


            projectId = pId
            classId   = cId        
            mrows     = getMethods(pId, cId, mId, df_mount)
      
            start_time  = time.perf_counter()
            methodfound = False

            for node in g.node:

                if (time.perf_counter() - start_time) > (60 * 120):
                    print(f'\x1b[1;37;41m ERROR \x1b[0m {mId} >>> Timeout error >>> Process took more than 120 min! Terminating process ...') 

                    SETOutput_method_TIMEOUT = SETOutput_method + '.SKIPPED.TIMEOUT.csv'
                    os.rename(SETOutput_method, SETOutput_method_TIMEOUT)
                    return

                try:
                    stmtId    = None #to be implemented
                    nodeId    = str(uuid.uuid4())
                    methodId  = None
                    nodeStPs_ = int(node.startPosition)
                    nodeEnPs_ = int(node.endPosition)

                    if not nodeStPs_ == -1: 
                        for i,m in mrows.iterrows():
                            tarmId   = str(m.method_id)
                            tarmStPs = int(m.start_pos)
                            tarmEnPs = int(m.end_pos)

                            if nodeStPs_ >= tarmStPs and nodeEnPs_ <= tarmEnPs:
                                methodId = tarmId

                    tmp_dotId = str(node.id) 
                    if tmp_dotId in in_nodes.keys():
                        methodId = in_nodes[tmp_dotId]

                    # if methodfound == True:
                    #     if methodId != mId: 
                    #         return
                            
                    if methodId == mId:
                        methodfound = True 

                        # node properties
                        nodeDtId = str(node.id) 
                        nodeType = getTypeText(node.type)
                        nodeCont = str(node.contents)
                        nodeStLn = str(node.startLineNumber)
                        nodeEnLn = str(node.endLineNumber)
                        nodeStPs = str(node.startPosition)
                        nodeEnPs = str(node.endPosition)
                        
                        # find the incoming and outgoing edges for node
                        allincoming = [] # all edges incoming to node
                        ch_incoming = []
                        nt_incoming = []
                        lu_incoming = []
                        lw_incoming = []
                        cf_incoming = []
                        ll_incoming = []
                        rt_incoming = []
                        fa_incoming = []
                        gb_incoming = []
                        gn_incoming = []

                        alloutgoing = [] # all edges outgoing from node
                        ch_outgoing = []
                        nt_outgoing = []
                        lu_outgoing = []
                        lw_outgoing = []
                        cf_outgoing = []
                        ll_outgoing = []
                        rt_outgoing = []
                        fa_outgoing = []
                        gb_outgoing = []
                        gn_outgoing = []        


                        for edge in g.edge:
                            if edge.sourceId == node.id:
                                alloutgoing.append(edge.destinationId)

                                if edge.type == FeatureEdge.AST_CHILD:
                                    ch_outgoing.append(edge.destinationId)

                                elif edge.type == FeatureEdge.NEXT_TOKEN:
                                    nt_outgoing.append(edge.destinationId)

                                elif edge.type == FeatureEdge.LAST_USE:
                                    lu_outgoing.append(edge.destinationId)
                                    
                                elif edge.type == FeatureEdge.LAST_WRITE:
                                    lw_outgoing.append(edge.destinationId)

                                elif edge.type == FeatureEdge.COMPUTED_FROM:
                                    cf_outgoing.append(edge.destinationId)

                                elif edge.type == FeatureEdge.LAST_LEXICAL_USE:
                                    ll_outgoing.append(edge.destinationId)

                                elif edge.type == FeatureEdge.RETURNS_TO:
                                    rt_outgoing.append(edge.destinationId)                                                                                                    

                                elif edge.type == FeatureEdge.FORMAL_ARG_NAME:
                                    fa_outgoing.append(edge.destinationId)

                                elif edge.type == FeatureEdge.GUARDED_BY:
                                    gb_outgoing.append(edge.destinationId)

                                elif edge.type == FeatureEdge.GUARDED_BY_NEGATION:
                                    gn_outgoing.append(edge.destinationId)                                                            

                            if edge.destinationId == node.id:
                                allincoming.append(edge.sourceId)

                                if edge.type == FeatureEdge.AST_CHILD:
                                    ch_incoming.append(edge.sourceId)

                                elif edge.type == FeatureEdge.NEXT_TOKEN:
                                    nt_incoming.append(edge.sourceId)

                                elif edge.type == FeatureEdge.LAST_USE:
                                    lu_incoming.append(edge.sourceId)
                                    
                                elif edge.type == FeatureEdge.LAST_WRITE:
                                    lw_incoming.append(edge.sourceId)

                                elif edge.type == FeatureEdge.COMPUTED_FROM:
                                    cf_incoming.append(edge.sourceId)

                                elif edge.type == FeatureEdge.LAST_LEXICAL_USE:
                                    ll_incoming.append(edge.sourceId)

                                elif edge.type == FeatureEdge.RETURNS_TO:
                                    rt_incoming.append(edge.sourceId)                                                                                                    

                                elif edge.type == FeatureEdge.FORMAL_ARG_NAME:
                                    fa_incoming.append(edge.sourceId)

                                elif edge.type == FeatureEdge.GUARDED_BY:
                                    gb_incoming.append(edge.sourceId)

                                elif edge.type == FeatureEdge.GUARDED_BY_NEGATION:
                                    gn_incoming.append(edge.sourceId)        

                        str_allincoming = listToString(allincoming)    
                        str_ch_incoming = listToString(ch_incoming)
                        str_nt_incoming = listToString(nt_incoming)     
                        str_lu_incoming = listToString(lu_incoming)
                        str_lw_incoming = listToString(lw_incoming)
                        str_cf_incoming = listToString(cf_incoming)
                        str_ll_incoming = listToString(ll_incoming)
                        str_rt_incoming = listToString(rt_incoming)
                        str_fa_incoming = listToString(fa_incoming)
                        str_gb_incoming = listToString(gb_incoming)
                        str_gn_incoming = listToString(gn_incoming)

                        str_alloutgoing = listToString(alloutgoing)    
                        str_ch_outgoing = listToString(ch_outgoing)
                        str_nt_outgoing = listToString(nt_outgoing)     
                        str_lu_outgoing = listToString(lu_outgoing)
                        str_lw_outgoing = listToString(lw_outgoing)
                        str_cf_outgoing = listToString(cf_outgoing)
                        str_ll_outgoing = listToString(ll_outgoing)
                        str_rt_outgoing = listToString(rt_outgoing)
                        str_fa_outgoing = listToString(fa_outgoing)
                        str_gb_outgoing = listToString(gb_outgoing)
                        str_gn_outgoing = listToString(gn_outgoing)

                        try:
                            if str_nt_incoming == None:
                                if not str_ch_outgoing == None:
                                    children = str_ch_outgoing.split(',')
                                    for child_dotId in children:
                                        in_nodes[child_dotId] = methodId
                        except Exception as e:
                            print(str(e))

                        wrlist = [projectId, classId, methodId, stmtId, nodeId, nodeDtId, nodeType, nodeCont, nodeStLn, nodeEnLn, nodeStPs, nodeEnPs, str_allincoming, str_alloutgoing, str_ch_incoming, str_ch_outgoing, str_nt_incoming, str_nt_outgoing, str_lu_incoming, str_lu_outgoing, str_lw_incoming, str_lw_outgoing, str_cf_incoming, str_cf_outgoing, str_ll_incoming, str_ll_outgoing, str_rt_incoming, str_rt_outgoing, str_fa_incoming, str_fa_outgoing, str_gb_incoming, str_gb_outgoing, str_gn_incoming, str_gn_outgoing]
                        csvAppender.writerow(wrlist)

                except Exception as e:
                    print('Something went wrong for pId: {} > cId: {}'.format(pId, cId))
                    print(str(e))



# parser = argparse.ArgumentParser()
# parser.add_argument('--project_path', '-p', help='path to project', required=True)
# parser.add_argument('--jar_paths', '-j', help='path to project')
# parser.add_argument('--write', '-w', help='path to project') # 'yes' or 'no' --- optional (can be hardcoded)
# args = parser.parse_args()
# write_permission = args.write


#### ******************** ####
#### NOTE: VERY IMPORTANT ####
#### ******************** ####
#### cd into / before running this script ####
print(">>> Make sure you are running Java 11 or higher!")
print(">>> Run java --version and javac --version")
response_a = input("Are you running Java 11 or higher? (yes/no) ")

print()
print(">>> Open up the terminal and type: cd / ")
response_b = input(">>> Did you cd into root directory before attempting to run this script? (yes/no) ")

good_to_go = False
if response_a == "yes" and response_b == "yes":
    good_to_go = True

project_path = "/Users/anjandash/Desktop/CONTRIB_PROJECTS/MatthewJWalls-anotherGame/"
jar_path = ""
methods_csv = sys.path[0] + "/contrib/meta_methods.csv"

if __name__ == "__main__":
    s = time.perf_counter()
    if not good_to_go:
        print()
        print("--------------------------------------------------------------")
        print(">>> Please cd into root directory before running this script!")
        print(">>> Please switch to Java 11 or higher!")
        print("--------------------------------------------------------------")
        print()
        sys.exit()

    with open(sys.path[0] + "/contrib/representations_text.csv", "r") as r_reprtext,  \
        open(sys.path[0] + "/contrib/representations_tkna.csv", "w+") as w_reprtkna, \
        open(sys.path[0] + "/contrib/representations_tknb.csv", "w+") as w_reprtknb, \
        open(sys.path[0] + "/contrib/representations_c2vc.csv", "w+") as w_reprc2vc, \
        open(sys.path[0] + "/contrib/representations_c2sq.csv", "w+") as w_reprc2sq, \
        open(sys.path[0] + "/contrib/representations_ftgr.csv", "w+") as w_reprftgr:

        csv_reader_reprtext = csv.reader(r_reprtext)
        csv_header_reprtext = next(csv_reader_reprtext)

        csv_writer_reprtkna = csv.writer(w_reprtkna)
        csv_writer_reprtkna.writerow(["method_id", "method_tokens_spaced"])

        csv_writer_reprtknb = csv.writer(w_reprtknb)
        csv_writer_reprtknb.writerow(["method_id", "method_tokens"])

        csv_writer_reprc2vc = csv.writer(w_reprc2vc)
        csv_writer_reprc2vc.writerow(["method_id", "method_c2vc"])

        csv_writer_reprc2sq = csv.writer(w_reprc2sq)
        csv_writer_reprc2sq.writerow(["method_id", "method_c2sq"])

        csv_writer_reprftgr = csv.writer(w_reprftgr)
        csv_writer_reprftgr.writerow(["method_id", "method_ftgr"])

        for i, row in enumerate(csv_reader_reprtext):
            print("********************************************************************************************", i)
            method_id, method_text = row 

            tokens = javalang.tokenizer.tokenize(method_text)
            tokens = [token.value for token in tokens]

            repr_tkna = " ".join(tokens)
            repr_tknb = ",".join([get_normalized_token(token) for token in tokens])
            repr_c2vc = gen_C2VC_from_method_text(method_id, method_text)
            repr_c2sq = gen_C2SQ_from_method_text(method_id, method_text)

            csv_writer_reprtkna.writerow([method_id, repr_tkna])
            csv_writer_reprtknb.writerow([method_id, repr_tknb])
            csv_writer_reprc2vc.writerow([method_id, repr_c2vc])
            csv_writer_reprc2sq.writerow([method_id, repr_c2sq])

            

            print(">>> Next, we will generate the feature-graph representations for the methods.")
            print(">>> It involves compiling the java files in the project.") 
            print(">>> It will create a \".proto\" file for every \".java\" file.")
            contt = "yes" #input(">>> Do you wish to continue? (yes/no) ")

            if contt == "yes" or contt == "y":
                print(">>> Attempting the feature-graph generation ... ")

                project_id = get_contrib_project_id(method_id)
                class_id   = get_contrib_class_id(method_id)
                class_path = get_contrib_method_path(method_id)
                start_pos, end_pos = get_stps_enps(method_id, class_path)

                # print("generating proto for class_path:", class_path)
                # print("--------------------------------")

                #print()
                df_mount = pd.DataFrame({"method_id": [method_id], "start_pos": [start_pos], "end_pos": [end_pos]})
                #print(df_mount.head())
                #print("--------------")


                if project_path:

                    # ******************************************* #
                    if jar_path == "" or jar_path is None:
                        jar_path = get_all_jar_paths(project_path)
                    # ******************************************* #
                    
                    make_proto(project_path, jar_path, class_path)
                    proto_path = class_path + ".proto"
                    if os.path.exists(proto_path):
                        populateNodes(project_id, class_id, method_id, proto_path, df_mount)
                        nodes_path = sys.path[0] + '/dependencies/featgraph/nodes/'+ method_id +'_.csv'
                        if os.path.exists(nodes_path):

                            # *******************************************************************************************************************************************************
                            # *******************************************************************************************************************************************************
                            # adding a new row at the bottom to signal change of methodId & invoke writing to jsonl
                            if os.path.exists(nodes_path):
                                with open(nodes_path, 'a') as afile:
                                    csvWriter = csv.writer(afile)
                                    csvWriter.writerow((['SET']*34))
                            else:
                                print(f'ERROR: {nodes_path} does not exist!')
                                continue

                            # *******************************************************************************************************************************************************
                            # *******************************************************************************************************************************************************

                            #df_datas = pd.read_csv(datafile, header=0, low_memory=False)
                            flagged_method = 0
                            counted_method = 0 

                            backbone_sequence  = []
                            excluded_sequence  = []
                            node_lbls_contents = []
                            method_target_lbl  = ""

                            remap   = {}
                            ed_dict = {}
                            CH_list = []
                            NT_list = []
                            LU_list = []
                            LW_list = []
                            CF_list = []
                            LL_list = []
                            RT_list = []
                            FA_list = []
                            GB_list = []
                            GN_list = []

                            with jsonlines.open(nodes_path+".jsonl", mode='a') as writer:
                                with open(nodes_path, 'r') as csvfile:
                                    csvReader = csv.reader(x.replace('\0', '') for x in csvfile)
                                    header = next(csvReader)
                                    #header2 = next(csvReader) # NOTE: Remove later

                                    for count, line in enumerate(csvReader):
                                        if line[0] == "project_id":
                                            continue

                                        ## ENCOUNTERED a new method (TODO: WARNING: If it doesn't encounter a new method, it doesn't do the writing => Check for the last method)
                                        if not flagged_method == line[2]:
                                            if not flagged_method == 0:

                                                remap = {}
                                                residual_sequence = [x for x in backbone_sequence if x not in excluded_sequence]                    

                                                # create remap dictionary
                                                for index_count, orig_dotId in enumerate(backbone_sequence):
                                                    remap[str(orig_dotId)] = index_count

                                                # remap residual sequence
                                                for index_count, orig_dotId in enumerate(residual_sequence):
                                                    residual_sequence[index_count] = remap[str(orig_dotId)]

                                                # remap conn_lists
                                                conn_lists  = [CH_list,    NT_list,    LU_list,    LW_list,    CF_list,    LL_list,    RT_list,    FA_list,    GB_list,    GN_list]

                                                # remove all tuples in conn_list where any component dotId doesn't exist in backbone_sequence
                                                for conn_list_count, conn_list in enumerate(conn_lists):
                                                    
                                                    pop_list = []
                                                    for cindex, dot_tuple  in enumerate(conn_list):
                                                        from_dotid, to_dotid = dot_tuple 
                                                        if from_dotid not in backbone_sequence or to_dotid not in backbone_sequence:
                                                            pop_list.append(cindex)

                                                    for pop_index in sorted(pop_list, reverse=True):
                                                        del conn_list[pop_index]                            

                                                # remap conn_lists one by one
                                                for conn_list_count, conn_list in enumerate(conn_lists):
                                                    for idx, conn_tuple in enumerate(conn_list):
                                                        ls_conn_tuple = list(conn_tuple)

                                                        err_flag = False
                                                        for index_count, orig_dotId in enumerate(ls_conn_tuple):
                                                            try: 
                                                                ls_conn_tuple[index_count] = remap[str(orig_dotId)]
                                                            except KeyError as e:
                                                                err_flag = True
                                                                print('\x1b[6;30;41m ERROR \x1b[0m dotId not found in mapped dict of backbone_sequence')

                                                        if err_flag:
                                                            continue
                                                        
                                                        remapped_conn_tuple = tuple(ls_conn_tuple)
                                                        conn_list[idx] = remapped_conn_tuple
                                                    
                                                    # for pop_index in pop_list:
                                                    #     conn_list.pop(pop_index)

                                                # add previous method edges to dictionary
                                                ed_dict["CH"] = CH_list
                                                ed_dict["NT"] = NT_list
                                                ed_dict["LU"] = LU_list
                                                ed_dict["LW"] = LW_list
                                                ed_dict["CF"] = CF_list
                                                ed_dict["LL"] = LL_list
                                                ed_dict["RT"] = RT_list
                                                ed_dict["FA"] = FA_list
                                                ed_dict["GB"] = GB_list
                                                ed_dict["GN"] = GN_list

                                                # *********************************************************************
                                                # *********************************************************************

                                                for backbone_node_idx in residual_sequence:
                                                    try:
                                                        assert 0 <= backbone_node_idx < len(node_lbls_contents)
                                                    except:
                                                        print("\x1b[6;30;41m ERROR \x1b[0m ASSERT ERROR in backbone_sequence")                            

                                                for k, v in ed_dict.items():
                                                    for from_idx, to_idx in v:
                                                        try:
                                                            assert 0 <= from_idx < len(node_lbls_contents)
                                                            assert 0 <= to_idx < len(node_lbls_contents)  
                                                        except:
                                                            print("\x1b[6;30;41m ERROR \x1b[0m ASSERT ERROR in Dict")

                                                # *********************************************************************   
                                                # *********************************************************************      
                                    
                                                # FOR METHODNAMEING
                                                # DONE No.1 >> Split MethodName by CamelCase
                                                # DONE No.2 >> Hide MethodName in node_labels 

                                                method_name_spl = cp.basic(method_target_lbl)
                                                if '<w>'  in method_name_spl: method_name_spl.remove('<w>')
                                                if '</w>' in method_name_spl: method_name_spl.remove('</w>')

                                                # node_lbls_contents = ['%LBL_PLACEHOLDER%' if token == method_target_lbl else token for token in node_lbls_contents]


                                                # *********************************************************************   
                                                # *********************************************************************      

                                                # prepare and write out the previous jsonline 
                                                # TODO: We need to add data keys as LABELS for every task, e.g. data['nulldef'] = "NULLDEF"/"NORMAL"
                                                data = {}
                                                data["backbone_sequence"] = residual_sequence
                                                data["node_labels"] = node_lbls_contents
                                                data["edges"] = ed_dict
                                                data["method_name"] = method_name_spl
                                                data["method_id"] = flagged_method

                                                # write the json data in jsonlines format
                                                writer.write(data)
                                                repr_ftgr = json.dumps(data)
                                                if counted_method % 100 == 0: 
                                                    print(f'Processed {counted_method}/10K+ methods.')
                                        
                                            # reset all previous jsonlines variables
                                            counted_method = counted_method + 1
                                            flagged_method = line[2]

                                            backbone_sequence  = []
                                            excluded_sequence  = []
                                            node_lbls_contents = []
                                            method_target_lbl  = "MLABEL"

                                            remap   = {}
                                            ed_dict = {}
                                            CH_list = []
                                            NT_list = []
                                            LU_list = []
                                            LW_list = []
                                            CF_list = []
                                            LL_list = []
                                            RT_list = []
                                            FA_list = []
                                            GB_list = []
                                            GN_list = []

                                        try: 
                                            projectId, classId, methodId, stmtId, nodeId, dotId, ntype, contents, stln, enln, stps, enps, allIncoming, allOutgoing, \
                                            CHIncoming, CHOutgoing, NTIncoming, NTOutgoing, LUIncoming, LUOutgoing, LWIncoming, LWOutgoing, CFIncoming, CFOutgoing, \
                                            LLIncoming, LLOutgoing, RTIncoming, RTOutgoing, FAIncoming, FAOutgoing, GBIncoming, GBOutgoing, GNIncoming, GNOutgoing = line       
                                        except Exception as e:
                                            print(str(e))
                                            print("ERROR in row:\n----\n", line)  
                                            sys.exit()    

                                        ## ***************************** ##
                                        if methodId == 'SET': continue   
                                        ## ***************************** ##
                                                        
                                        if NTIncoming.split(',') == [''] and NTOutgoing.split(',') == ['']:
                                            excluded_sequence.append(int(float(dotId)))
                                        
                                        backbone_sequence.append(int(float(dotId)))
                                        node_lbls_contents.append(str(contents))

                                        connections = [CHOutgoing, NTOutgoing, LUOutgoing, LWOutgoing, CFOutgoing, LLOutgoing, RTOutgoing, FAOutgoing, GBOutgoing, GNOutgoing]
                                        conn_lists  = [CH_list,    NT_list,    LU_list,    LW_list,    CF_list,    LL_list,    RT_list,    FA_list,    GB_list,    GN_list]

                                        for i in range(len(connections)):
                                            str_outgoing = connections[i].split(',')

                                            for outId in str_outgoing: 
                                                if not outId == '':
                                                    conn_lists[i].append((int(float(dotId)), int(float(outId))))

                                        # ******************************************
                                        # class CodeGraph2Seq(TypedDict):
                                        #     backbone_sequence: List[int]  ---> dotId
                                        #     node_labels: List[str]        ---> contents
                                        #     edges: Dict[str, List[Tuple[int, int]]]
                                        #     method_name: List[str]        ---> getMethodId(flagged_method)
                                        # ******************************************

                    else:
                        print()
                        print("********************************")
                        print("make_proto() FAILED!")
                        print(".proto path does not exist for: ", class_path)
                        print("********************************")
                    # NOTE: Next run 2) create_jsonl_db.py and 3) _reindexing_jsonl_files_CREATE_JSONL_DATASET.py >>>> in /Volumes/AJDRV/
                    # ******************************************

            
            if repr_ftgr == "" or repr_ftgr is None:
                print("********************************")
                print("feature-graph generation FAILED!")
                print("method_id:", method_id)
                print("********************************") 
            if repr_ftgr:
                csv_writer_reprftgr.writerow([method_id, str(repr_ftgr)])           

            print("********************************")
            #print(repr_tkna, "\n\n", repr_tknb, repr_c2vc, repr_c2sq, repr_ftgr)
    print(time.perf_counter() - s, "Seconds.")

