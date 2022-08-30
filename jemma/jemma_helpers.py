import os 
import sys
import csv
import time
import uuid
import subprocess
import concurrent.futures
import codeprep.api.text as cp

from graph_pb2 import Graph
from graph_pb2 import FeatureNode
from graph_pb2 import FeatureEdge

def getMethods(pId, cId, mId, df_mount):
    mrows = df_mount[df_mount['method_id'] == mId]
    return mrows

def is_valid(mid):
    try:
        uuid.UUID(str(mid))
        return True
    except ValueError:
        return False    

def runcmd(cmd):
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        process.wait()
    except Exception as e:
        print("_ERROR_", str(e))    

def get_all_class_paths(project_path):
    fullclasspath = ''
    for root, dirs, files in os.walk(project_path):
        for edir in dirs:
            fullclasspath = fullclasspath + ':' + os.path.join(root, edir) + '/'
    return fullclasspath

def get_all_jar_paths(project_path):
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
    runcmd(command)


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
