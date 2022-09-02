import os,sys,csv

metadata = {
"5807578":  "Jemma_Metadata_Projects.csv",
"5807586":  "Jemma_Metadata_Packages.csv",
"5808902":  "Jemma_Metadata_Classes.csv",
"5813089":  "Jemma_Metadata_Methods.csv",
}

properties = {
"1096082": "Jemma_Properties_Methods_RSLK.csv",
"1096080": "Jemma_Properties_Methods_NLDF.csv",
"7020084": "Jemma_Properties_Methods_NMLC.csv",
"7019960": "Jemma_Properties_Methods_NMNC.csv",
"7019176": "Jemma_Properties_Methods_NUCC.csv",
"7019128": "Jemma_Properties_Methods_NUPC.csv", 
"5813084": "Jemma_Properties_Methods_CMPX.csv", 
"5813081": "Jemma_Properties_Methods_MXIN.csv", 
"5813308": "Jemma_Properties_Methods_NAME.csv", 
"5813054": "Jemma_Properties_Methods_NMLT.csv", 
"5813055": "Jemma_Properties_Methods_NMOP.csv", 
"5813053": "Jemma_Properties_Methods_NMPR.csv", 
"5813034": "Jemma_Properties_Methods_NMRT.csv", 
"5813032": "Jemma_Properties_Methods_NMTK.csv", 
"5813029": "Jemma_Properties_Methods_NTID.csv", 
"5813028": "Jemma_Properties_Methods_NUID.csv",
"5813094": "Jemma_Properties_Methods_SLOC.csv", 
"5813102": "Jemma_Properties_Methods_TLOC.csv"
}

representations = {
"5813705": "Jemma_Representations_Methods_TEXT.csv",
"5813717": "Jemma_Representations_Methods_TKNA.csv",
"5813730": "Jemma_Representations_Methods_TKNB.csv",    
"5813993": "Jemma_Representations_Methods_C2VC.csv",
"5814059": "Jemma_Representations_Methods_C2SQ.csv",
"5813933": "Jemma_Representations_Methods_FTGR.csv",
}

callgraphs = {
"6758937": "jemma_projects_callgraphs.zip"
}



ERROR = "\x1b[6;30;41m  ERROR  \x1b[0m"
dst_base = sys.path[0] + "/../jemma_datasets/"
dst_dats = ["metadata", "properties", "representations", "callgraphs"]

for i, data in enumerate([metadata, properties, representations, callgraphs]):
    for project_identifier, filename in data.items():

        dst_path = os.path.join(dst_base, dst_dats[i])
        if not os.path.exists(dst_path):
            print()
            print(ERROR, dst_path, "does not exist! \nAll data files may not have been downloaded!")