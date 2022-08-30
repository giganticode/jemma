"""
jemma_utils

author: @anjandash
license: MIT
"""

import sys
import uuid
import pandas as pd 
import jemma_helpers as jh

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


projects_csv = "./jemma_datasets/metatdata/Jemma_Metadata_Projects.csv"
packages_csv = "./jemma_datasets/metatdata/Jemma_Metadata_Packages.csv"
classes_csv  = "./jemma_datasets/metatdata/Jemma_Metadata_Classes.csv"
methods_csv  = "./jemma_datasets/metatdata/Jemma_Metadata_Methods.csv"

properties = { 
    "RSLK": "./jemma_datasets/properties/Jemma_Properties_Methods_RSLK.csv",
    "NLDF": "./jemma_datasets/properties/Jemma_Properties_Methods_NLDF.csv",
    "NMLC": "./jemma_datasets/properties/Jemma_Properties_Methods_NMLC.csv",
    "NMNC": "./jemma_datasets/properties/Jemma_Properties_Methods_NMNC.csv",
    "NUCC": "./jemma_datasets/properties/Jemma_Properties_Methods_NUCC.csv",
    "NUPC": "./jemma_datasets/properties/Jemma_Properties_Methods_NUPC.csv",
    "CMPX": "./jemma_datasets/properties/Jemma_Properties_Methods_CMPX.csv",
    "MXIN": "./jemma_datasets/properties/Jemma_Properties_Methods_MXIN.csv",
    "NAME": "./jemma_datasets/properties/Jemma_Properties_Methods_NAME.csv",
    "NMLT": "./jemma_datasets/properties/Jemma_Properties_Methods_NMLT.csv",
    "NMOP": "./jemma_datasets/properties/Jemma_Properties_Methods_NMOP.csv",
    "NMPR": "./jemma_datasets/properties/Jemma_Properties_Methods_NMPR.csv",
    "NMRT": "./jemma_datasets/properties/Jemma_Properties_Methods_NMRT.csv",
    "NMTK": "./jemma_datasets/properties/Jemma_Properties_Methods_NMTK.csv",
    "NTID": "./jemma_datasets/properties/Jemma_Properties_Methods_NTID.csv",
    "NUID": "./jemma_datasets/properties/Jemma_Properties_Methods_NUID.csv",
    "SLOC": "./jemma_datasets/properties/Jemma_Properties_Methods_SLOC.csv",
    "TLOC": "./jemma_datasets/properties/Jemma_Properties_Methods_TLOC.csv",    
}

representations = {
    "TEXT": "./jemma_datasets/representations/Jemma_Representations_Methods_TEXT.csv",
    "TKNA": "./jemma_datasets/representations/Jemma_Representations_Methods_TKNA.csv",
    "TKNB": "./jemma_datasets/representations/Jemma_Representations_Methods_TKNB.csv",    
    "C2VC": "./jemma_datasets/representations/Jemma_Representations_Methods_C2VC.csv",
    "C2SQ": "./jemma_datasets/representations/Jemma_Representations_Methods_C2SQ.csv",
    "FTGR": "./jemma_datasets/representations/Jemma_Representations_Methods_FTGR.csv"
}

properties_label = {
    "RSLK": "resource_leak",
    "NLDF": "null_dereference",
    "NMLC": "num_local_calls",
    "NMNC": "num_non_local_calls",
    "NUCC": "num_unique_callees",
    "NUPC": "num_unique_callers",
    "CMPX": "cyclomatic_complexity",
    "MXIN": "max_indent",
    "NAME": "method_name",
    "NMLT": "num_literals",
    "NMOP": "num_operators",
    "NMPR": "num_parameters",
    "NMRT": "num_returns",
    "NMTK": "num_tokens",
    "NTID": "num_identifiers",
    "NUID": "num_unique_identifiers",
    "SLOC": "source_lines_of_code",
    "TLOC": "total_lines_of_code",    
}


# *************** #
#  get functions  #
# *************** #


# *************** #
#    projects     #
# *************** #


def get_project_id(project_name):
    """
    Returns the project id of the project (queried by project name).

    Parameters:
    * project_name: (str) - name of the project

    Returns:
    * Returns a str uuid of the corresponding project (project_id)
    * Returns None if no such project_id was found
    * Returns None if multiple projects were found with the same name
    """

    df = pd.read_csv(projects_csv, header=0)
    df = df[df["project_name"] == project_name.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]["project_id"]
    return None


def get_project_id_by_path(project_path):
    """
    Returns the project id of the project (queried with project path).

    Parameters:
    * project_path: (str) - path of the project defined in jemma

    Returns:
    * Returns a str uuid of the corresponding project (project_id)
    * Returns None if no such project_path was found
    * Returns None if multiple projects were found with the same path
    """

    df = pd.read_csv(projects_csv, header=0)
    df = df[df["project_path"] == project_path.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]["project_id"]
    return None

def get_project_id_class_id(class_id):
    """
    Returns the project id of the project (queried with class id).

    Parameters:
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a str uuid of the corresponding project (project_id)
    * Returns None if no such project_id was found 
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df['class_id'] == class_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]['project_id']
    return None


def get_project_id_by_method_id(method_id):
    """
    Returns the project id of the project (queried with method id).

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a str uuid of the corresponding project (project_id)
    * Returns None if no such project_id was found
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]['project_id']
    return None

def get_project_name(project_id):
    """
    Returns the project name of the project.

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a str of the corresponding project name
    * Returns None if no such project_id is defined in jemma
    """

    df = pd.read_csv(projects_csv, header=0)
    df = df[df["project_id"] == project_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]["project_name"]
    return None


def get_project_path(project_id):
    """
    Returns the project path of the project.

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a str of the corresponding project path
    * Returns None if no such project_id is defined in jemma
    """

    df = pd.read_csv(projects_csv, header=0)
    df = df[df["project_id"] == project_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]["project_path"]
    return None

def get_project_size_by_classes(project_id):
    """
    Returns the size of a project, by the number of classes. 

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a str of the corresponding project size, by the number of classes
    * Returns None if no such project_id is defined in jemma
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df['project_id'] == project_id.strip()]
    return df.shape[0]

def get_project_size_by_methods(project_id):
    """
    Returns the size of a project, by the number of methods. 

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a str of the corresponding project size, by the number of methods
    * Returns None if no such project_id is defined in jemma
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['project_id'] == project_id.strip()]
    return df.shape[0]

def get_project_class_ids(project_id):    
    """
    Returns all class ids defined within the project.

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a (List[str]) corresponding to all class ids in the project   
    * Returns an empty List if no classes are found
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df['project_id'] == project_id.strip()]
    return df['class_id'].tolist()

def get_project_method_ids(project_id):
    """
    Returns all method ids defined within the project.

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a (List[str]) corresponding to all method ids in the project   
    * Returns an empty List if no methods are found
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['project_id'] == project_id.strip()]
    return df['method_id'].tolist()

def get_project_class_names(project_id):    
    """
    Returns all class names defined within the project.

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a (List[str]) corresponding to all class names in the project   
    * Returns an empty List if no classes are found
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df['project_id'] == project_id.strip()]
    return df['class_name'].tolist()

def get_project_method_names(project_id):
    """
    Returns all method names defined within the project.

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a (List[str]) corresponding to all method names in the project   
    * Returns an empty List if no methods are found
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['project_id'] == project_id.strip()]
    return df['method_name'].tolist()
                
def get_project_metadata(project_id):
    """
    Returns all metadata related to a particular project.

    Parameters: 
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a dictionary of project metadata values
    * Returns None if no such project_id is defined in jemma
    """

    df = pd.read_csv(projects_csv, header=0)
    df = df[df['project_id'] == project_id]

    if df.shape[0] == 1:
        return df.iloc[0].to_dict()
    return None    


# *************** #
#     classes     #
# *************** #


def get_class_id(project_id, class_name):
    """
    Returns the class id of a class in project (queried by class name).

    Parameters:
    * project_id: (str) - project_id of a project
    * class_name: (str) - class name of a class within the project

    Returns:
    * Returns a str uuid of the corresponding class (class_id)
    * Returns None if no such project_id or class_name was found
    * Returns None if multiple classes were found with the same name (use: get_class_id_by_path)
    """
    
    df = pd.read_csv(classes_csv, header=0)
    df = df[(df['project_id'] == project_id.strip()) & (df['class_name'] == class_name.strip())]

    if df.shape[0] == 1:
        return df.iloc[0]['class_id']
    return None


def get_class_id_by_path(class_path):
    """
    Returns the class id of a class (queried with class path).

    Parameters:
    * class_path: (str) - path of the class defined in jemma

    Returns:
    * Returns a str uuid of the corresponding class (class_id)
    * Returns None if no such class_path was found
    * Returns None if multiple classes were found with the same path
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df["class_path"] == class_path.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]['class_id']
    return None


def get_class_id_by_method_id(method_id):
    """
    Returns the class id of a class (queried with method id).

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a str uuid of the corresponding class (class_id)
    * Returns None if no such class_id was found
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]['class_id']
    return None    


def get_class_name(class_id):
    """
    Returns the class name of a particular class.

    Parameters:
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a str of the corresponding class name
    * Returns None if no such class_id is defined in jemma
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df["class_id"] == class_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]['class_name']
    return None


def get_class_path(class_id):
    """
    Returns the class path of a particular class.

    Parameters:
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a str of the corresponding class path
    * Returns None if no such class_id is defined in jemma
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df["class_id"] == class_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]['class_path']
    return None


def get_class_size_by_methods(class_id):
    """
    Returns the size of a class, by the number of methods. 

    Parameters:
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a str of the corresponding class size, by the number of methods
    * Returns None if no such class_id is defined in jemma
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['class_id'] == class_id.strip()]
    return df.shape[0]        

def get_class_method_ids(class_id):
    """
    Returns all method ids defined within a particular class.

    Parameters:
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a (List[str]) corresponding to all method ids in the class   
    * Returns an empty List if no methods are found
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['class_id'] == class_id.strip()]
    return df['method_id'].tolist()    

def get_class_method_names(class_id):
    """
    Returns all method names within a particular class.

    Parameters:
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a (List[str]) corresponding to all method names in the class   
    * Returns an empty List if no methods are found
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['class_id'] == class_id.strip()]
    return df['method_name'].tolist()    

def get_class_metadata(class_id):
    """
    Returns all metadata related to a particular class.

    Parameters: 
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a dictionary of class metadata values
    * Returns None if no such class_id is defined in jemma
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df['class_id'] == class_id]

    if df.shape[0] == 1:
        return df.iloc[0].to_dict()
    return None    

# *************** #
#     methods     #
# *************** #


def get_method_id(class_id, method_name):
    """
    Returns the method id of a method in a class (queried by method name).

    Parameters:
    * class_id: (str) - any class_id defined within jemma
    * method_name: (str) - method name of a method within the class

    Returns:
    * Returns a str uuid of the corresponding method (method_id)
    * Returns None if no such class_id or method_name was found
    * Returns None if multiple methods were found with the same name (use: get_method_id_stln_enln)
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[(df['class_id'] == class_id.strip()) & (df['method_name'] == method_name.strip())]

    if df.shape[0] == 1:
        return df.iloc[0]['method_id']
    return None

def get_method_id_by_stln_enln(class_id, method_name, stln, enln):
    """
    Returns the method id of a method in a class (queried by method name, start line, and end line).

    Parameters:
    * class_id: (str) - any class_id defined within jemma
    * method_name: (str) - method name of a method within the class
    * stln: (str) - start line of the method within the class
    * enln: (str) - end line of the method within the class

    Returns:
    * Returns a str uuid of the corresponding method (method_id)
    * Returns None if no such class_id or method_name was found
    """    

    df = pd.read_csv(methods_csv, header=0)
    df = df[(df['class_id'] == class_id.strip()) & (df['method_name'] == method_name.strip())]
    df = df[(df['start_line'] == stln.strip()) & (df['end_line'] == enln.strip())]

    if df.shape[0] == 1:
        return df.iloc[0]['method_id']
    return None


def get_method_path(method_id):
    """
    Returns the class path of the parent class of a method.

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a str of the corresponding class path
    * Returns None if no such method_id is defined in jemma
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        return df.iloc[0]['method_path']
    return None    

def get_start_line(method_id):
    """
    Returns the start line of a particular method.

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a str of the corresponding start line of the method
    * Returns None if no such method_id is defined in jemma
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        return df.iloc[0]['start_line']
    return None    

def get_end_line(method_id):
    """
    Returns the end line of a particular method.

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a str of the corresponding end line of the method
    * Returns None if no such method_id is defined in jemma
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        return df.iloc[0]['end_line']
    return None

def get_method_metadata(method_id):
    """
    Returns all metadata related to a particular method.

    Parameters: 
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a dictionary of method metadata values
    * Returns None if no such method_id is defined in jemma
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        return df.iloc[0].to_dict()
    return None    


def get_stps_enps(method_id):
    pass



# *************** #
#      utils      #
# *************** #


def get_properties(property, methods):
    """
    Get property values for a list of methods.

    Parameters:
    * property : (str) - property code
    * methods : (list[str]) - list of unique methods ids

    Returns:
    * pandas Dataframe object (with method_id, property) of the passed list of methods
    """

    df_m = pd.DataFrame({'method_id': methods})
    df_p = pd.read_csv(properties.get(property, None), header=0)
    df_f = pd.merge(df_p, df_m, on="method_id")

    return df_f

def get_balanced_properties(property, methods):
    """
    Get balanced property values for a list of methods.

    Parameters:
    * property : (str) - property code
    * methods : (list[str]) - list of unique methods ids [OPTIONAL]

    Returns:
    * pandas Dataframe object (with method_id, property) of the passed list of methods
    """

    df_p = pd.read_csv(properties.get(property, None), header=0)

    if methods:
        df_p = df_p[df_p["method_id"].isin(methods)]

    lbls = list(set(df_p[properties_label.get(property, None)].tolist()))
    minc = min([len(df_p[properties_label.get(property, None) == lbl].tolist()) for lbl in lbls])

    df_l = [df_p[properties_label.get(property, None) == lbl].tolist().head(minc) for lbl in lbls]
    df_f = pd.concat(df_l, ignore_index=True)

    return df_f


def get_representations(representation, methods):
    """
    Get representation values of a list of methods.

    Parameters:
    * representation : (str) - representation code
    * methods : (list[str]) - list of unique methods ids

    Returns:
    * pandas Dataframe object (with method_id, representation) of the passed list of methods
    """

    df_m = pd.DataFrame({'method_id': methods})
    df_r = pd.read_csv(properties.get(representation, None), header=0)
    df_f = pd.merge(df_r, df_m, on="method_id")

    return df_f



def get_callees(method_id):
    """
    Get a list of method ids for direct callees of a particular method.

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a (List[str]) of method ids for direct callees
    * Returns an empty List if no such method_id exists
    """

    project_id = get_project_id_by_method_id(method_id)
    project_cg = sys.path[0] + "/jemma_datasets/callgraphs/" + project_id + ".csv"

    df = pd.read_csv(project_cg, header=0)
    df = df[df["caller_method_id"] == method_id]
    
    callees = df["callee_method_id"].tolist()
    return list(set(callees))

def get_callers(method_id):
    """
    Get a list of method ids for direct callers of a particular method.

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a (List[str]) of method ids for direct callers
    * Returns an empty List if no such method_id exists
    """

    project_id = get_project_id_by_method_id(method_id)
    project_cg = sys.path[0] + "/jemma_datasets/callgraphs/" + project_id + ".csv"

    df = pd.read_csv(project_cg, header=0)
    df = df[df["callee_method_id"] == method_id]
    
    callers = df["caller_method_id"].tolist()
    return list(set(callers))


def get_caller_context(method_id, n_neighborhood, df):    
    """
    Get all caller method ids from n-hop neighborhood for a particular method.

    Parameters:
    * method_id: (str) - method_id for which callers are to be determined
    * n_neighborhood: (int) - size of n-hop neighborhood callers that are to be considered
    * df: (pandas Dataframe) - pandas Dataframe containing the caller-callee data for the project

    Returns:
    * Returns a (List[str]) of caller method ids 
    * Returns an empty List if no callers could be found for method_id
    * Returns an empty List if n_neighborhood is 0    
    """
    if n_neighborhood == 0:
        return []

    return_val = []
    df = df[df["callee_method_id"] == method_id]
    callers = df["caller_method_id"].tolist()
    callers = [item for item in callers if jh.is_valid(item)]
    callers = list(set(callers))    

    if method_id in callers: 
        callers = list(filter(lambda x: x != method_id, callers))    

    return_val.extend(callers)
    n_size = n_size - 1
    for caller in callers:
        return_val.extend(get_caller_context(caller, n_size, df))
          
    return return_val

def get_callee_context(method_id, n_neighborhood, df):
    """
    Get all callee method ids from n-hop neighborhood for a particular method.

    Parameters:
    * method_id: (str) - method_id for which callees are to be determined
    * n_neighborhood: (int) - size of n-hop neighborhood callees that are to be considered
    * df: (pandas Dataframe) - pandas Dataframe containing the caller-callee data for the project

    Returns:
    * Returns a (List[str]) of callee method ids 
    * Returns an empty List if no callees could be found for method_id
    * Returns an empty List if n_neighborhood is 0    
    """
    if n_neighborhood == 0:
        return []

    return_val = []
    df = df[df["caller_method_id"] == method_id]
    callees = df["callee_method_id"].tolist()
    callees = [item for item in callees if jh.is_valid(item)]
    callees = list(set(callees))    

    if method_id in callees: 
        callees = list(filter(lambda x: x != method_id, callees))    

    return_val.extend(callees)
    n_size = n_size - 1
    for callee in callees:
        return_val.extend(get_caller_context(callee, n_size, df))
          
    return return_val


# ************************************************************************************

# def is_valid(mid):
#     try:
#         uuid.UUID(str(mid))
#         return True
#     except ValueError:
#         return False    

# def get_caller_methods(pid, methods_list, n_size, df):
#     if n_size == 0 or len(methods_list) == 0:
#         return []

#     return_val = []
#     for mid in methods_list: 
#         df = df[df["callee_method_id"] == mid]
#         callers = df["caller_method_id"].tolist()
#         callers = [item for item in callers if is_valid(item)]
#         callers = list(set(callers))

#         if mid in callers: 
#             callers = list(filter(lambda x: x != mid, callers))
#         #print(f"For mid: {mid} the callers are: {callers}")

#         n_size = n_size - 1
#         return_val.append([callers, get_caller_methods(pid, callers, n_size, df)])
#     return return_val

# def get_caller_methods(pid, methods_list, n_size, df):
#     if n_size == 0 or len(methods_list) == 0:
#         return []

#     return_val = []
#     for mid in methods_list: 
#         df = df[df["callee_method_id"] == mid]
#         callers = df["caller_method_id"].tolist()
#         callers = [item for item in callers if is_valid(item)]
#         callers = list(set(callers))

#         if mid in callers: 
#             callers = list(filter(lambda x: x != mid, callers))
#         #print(f"For mid: {mid} the callers are: {callers}")

#         n_size = n_size - 1
#         return_val.append([callers, get_caller_methods(pid, callers, n_size, df)])
#     return return_val
