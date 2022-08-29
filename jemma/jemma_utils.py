"""
jemma_utils

author: @anjandash
license: MIT
"""

import sys
import uuid
import pandas as pd 


projects_csv = "./jemma_datasets/metatdata/Jemma_Metadata_Projects.csv"
packages_csv = "./jemma_datasets/metatdata/Jemma_Metadata_Packages.csv"
classes_csv  = "./jemma_datasets/metatdata/Jemma_Metadata_Classes.csv"
methods_csv  = "./jemma_datasets/metatdata/Jemma_Metadata_Methods.csv"

properties = { 
    "CMPX": "./jemma_datasets/properties/Jemma_Properties_Methods_CMPX.csv",
    "SLOC": "./jemma_datasets/properties/Jemma_Properties_Methods_SLOC.csv",
    "MXIN": "./jemma_datasets/properties/Jemma_Properties_Methods_MXIN.csv",
}

representations = {
    "TEXT": "./data/Giganticode_50PLUS_DB_representations_TEXT_CENTOS.csv",
    "TOKN": "./data/Giganticode_50PLUS_DB_representations_TOKN_CENTOS.csv",
    "C2VC": "./data/Giganticode_50PLUS_DB_representations_C2VC_CENTOS.csv",
    "C2SQ": "./data/Giganticode_50PLUS_DB_representations_C2SQ_CENTOS.csv",
}

properties_label = {
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
    Returns the project_id of the project.

    Parameters:
    * project_name: (str) - name of the project

    Returns:
    * Returns a str uuid of the corresponding project (project_id)
    * Returns None if no project_id was found
    * Returns None if multiple projects were found with the same name
    """

    df = pd.read_csv(projects_csv, header=0)
    df = df[df["project_name"] == project_name.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]["project_id"]
    return None


def get_project_id_by_path(project_path):
    """
    Returns the project_id of the project (queried with project_path).

    Parameters:
    * project_path: (str) - path of the project defined in jemma

    Returns:
    * Returns a str uuid of the corresponding project (project_id)
    * Returns None if no project_id was found
    * Returns None if multiple projects were found with the same name
    """

    df = pd.read_csv(projects_csv, header=0)
    df = df[df["project_path"] == project_path.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]["project_id"]
    return None

def get_project_id_class_id(class_id):
    """
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df['class_id'] == class_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]['project_id']
    return None


def get_project_id_by_method_id(method_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]['project_id']
    return None

def get_project_name(project_id):
    """
    """

    df = pd.read_csv(projects_csv, header=0)
    df = df[df["project_id"] == project_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]["project_name"]
    return None


def get_project_path(project_id):
    """
    """

    df = pd.read_csv(projects_csv, header=0)
    df = df[df["project_id"] == project_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]["project_path"]
    return None

def get_project_size_by_classes(project_id):
    """
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df['project_id'] == project_id.strip()]
    return df.shape[0]

def get_project_size_by_methods(project_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['project_id'] == project_id.strip()]
    return df.shape[0]

def get_project_class_ids(project_id):    
    """
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df['project_id'] == project_id.strip()]
    return df['class_id'].tolist()

def get_project_method_ids(project_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['project_id'] == project_id.strip()]
    return df['method_id'].tolist()

def get_project_class_names(project_id):    
    """
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df['project_id'] == project_id.strip()]
    return df['class_name'].tolist()

def get_project_method_names(project_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['project_id'] == project_id.strip()]
    return df['method_name'].tolist()
                

# *************** #
#     classes     #
# *************** #


def get_class_id(project_id, class_name):
    """
    """
    
    df = pd.read_csv(classes_csv, header=0)
    df = df[(df['project_id'] == project_id.strip()) & (df['class_name'] == class_name.strip())]

    if df.shape[0] == 1:
        return df.iloc[0]['class_id']
    return None


def get_class_id_by_path(class_path):
    """
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df["class_path"] == class_path.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]['class_id']
    return None


def get_class_id_by_method_id(method_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]['class_id']
    return None    


def get_class_name(class_id):
    """
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df["class_id"] == class_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]['class_name']
    return None


def get_class_path(class_id):
    """
    """

    df = pd.read_csv(classes_csv, header=0)
    df = df[df["class_id"] == class_id.strip()]

    if df.shape[0] == 1:
        return df.iloc[0]['class_path']
    return None


def get_class_size_by_methods(class_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['class_id'] == class_id.strip()]
    return df.shape[0]        

def get_class_method_ids(class_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['class_id'] == class_id.strip()]
    return df['method_id'].tolist()    

def get_class_method_names(class_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['class_id'] == class_id.strip()]
    return df['method_name'].tolist()    




# *************** #
#     methods     #
# *************** #


def get_method_id(class_id, method_name):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[(df['class_id'] == class_id.strip()) & (df['method_name'] == method_name.strip())]

    if df.shape[0] == 1:
        return df.iloc[0]['method_id']
    return None


def get_method_path(method_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        return df.iloc[0]['method_path']
    return None    

def get_start_line(method_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        return df.iloc[0]['start_line']
    return None    

def get_end_line(method_id):
    """
    """

    df = pd.read_csv(methods_csv, header=0)
    df = df[df['method_id'] == method_id]

    if df.shape[0] == 1:
        return df.iloc[0]['end_line']
    return None

# *************** #
#      utils      #
# *************** #


def get_properties(property, methods):
    """
    Get property values of a list of methods 

    Parameters:
    * property : (str) - property code
    * methods : (list[str]) - list of unique methods ids

    Returns:
    * pd.Dataframe object (with method_id, property) of the passed list of methods
    """

    df_m = pd.DataFrame({'method_id': methods})
    df_p = pd.read_csv(properties.get(property, None), header=0)
    df_f = pd.merge(df_p, df_m, on="method_id")

    return df_f

def get_balanced_properties(property):
    df_p = pd.read_csv(properties.get(property, None), header=0)

    lbls = list(set(df_p[properties_label.get(property, None)].tolist()))
    minc = min([len(df_p[properties_label.get(property, None) == lbl].tolist()) for lbl in lbls])

    df_l = [df_p[properties_label.get(property, None) == lbl].tolist().head(minc) for lbl in lbls]
    df_f = pd.concat(df_l, ignore_index=True)

    return df_f


def get_representations(representation, methods):
    """
    Get representation values of a list of methods

    Parameters:
    * representation : (str) - representation code
    * methods : (list[str]) - list of unique methods ids

    Returns:
    * pd.Dataframe object (with method_id, representation) of the passed list of methods
    """

    df_m = pd.DataFrame({'method_id': methods})
    df_r = pd.read_csv(properties.get(representation, None), header=0)
    df_f = pd.merge(df_r, df_m, on="method_id")

    return df_f



def get_callees(method_id):
    """
    """

    project_id = get_project_id_by_method_id(method_id)
    project_cg = sys.path[0] + "/jemma_datasets/callgraphs/" + project_id + ".csv"

    df = pd.read_csv(project_cg, header=0)
    df = df[df["caller_method_id"] == method_id]
    
    callees = df["callee_method_id"].tolist()
    return list(set(callees))

def get_callers(method_id):
    """
    """

    project_id = get_project_id_by_method_id(method_id)
    project_cg = sys.path[0] + "/jemma_datasets/callgraphs/" + project_id + ".csv"

    df = pd.read_csv(project_cg, header=0)
    df = df[df["callee_method_id"] == method_id]
    
    callers = df["caller_method_id"].tolist()
    return list(set(callers))

def get_caller_context(method_id, n_neighborhood):
    pass 

def get_callee_context(method_id, n_neighborhood):
    pass




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