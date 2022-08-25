import os
import sys
import csv
import time 
import uuid 
import javalang
import argparse



contrib_path = "/Users/anjandash/Desktop/CONTRIB_PROJECTS/"
ERROR_STATUS = "\x1b[6;30;41m ERROR \x1b[0m"

def get_method_start_end(method_node, tree):
    startpos  = None
    endpos    = None
    startline = None
    endline   = None
    for path, node in tree:
        if startpos is not None and method_node not in path:
            endpos = node.position
            endline = node.position.line if node.position is not None else None
            break
        if startpos is None and node == method_node:
            startpos = node.position
            startline = node.position.line if node.position is not None else None
    return startpos, endpos, startline, endline


def get_method_text(startpos, endpos, startline, endline, last_endline_index, codelines):
    if startpos is None:
        return "", None, None, None
    else:
        startline_index = startline - 1 
        endline_index = endline - 1 if endpos is not None else None 

        # 1. check for and fetch annotations
        if last_endline_index is not None:
            for line in codelines[(last_endline_index + 1):(startline_index)]:
                if "@" in line: 
                    startline_index = startline_index - 1
        meth_text = "<ST>".join(codelines[startline_index:endline_index])
        meth_text = meth_text[:meth_text.rfind("}") + 1] 

        # 2. remove trailing rbrace for last methods & any external content/comments
        # if endpos is None and 
        if not abs(meth_text.count("}") - meth_text.count("{")) == 0:
            # imbalanced braces
            brace_diff = abs(meth_text.count("}") - meth_text.count("{"))

            for _ in range(brace_diff):
                meth_text  = meth_text[:meth_text.rfind("}")]    
                meth_text  = meth_text[:meth_text.rfind("}") + 1]     

        meth_lines = meth_text.split("<ST>")  
        meth_text  = "".join(meth_lines)                   
        last_endline_index = startline_index + (len(meth_lines) - 1) 

        return meth_text, (startline_index + 1), (last_endline_index + 1), last_endline_index


def main():
    with open(sys.path[0] + "/contrib/meta_projects.csv", "w+") as w_projects, \
         open(sys.path[0] + "/contrib/meta_packages.csv", "w+") as w_packages, \
         open(sys.path[0] + "/contrib/meta_classes.csv", "w+") as w_classes, \
         open(sys.path[0] + "/contrib/meta_methods.csv", "w+") as w_methods, \
         open(sys.path[0] + "/contrib/representations_text.csv", "w+") as w_reprtext:

        csv_writer_projects = csv.writer(w_projects)
        csv_writer_packages = csv.writer(w_packages)
        csv_writer_classes  = csv.writer(w_classes)
        csv_writer_methods  = csv.writer(w_methods)
        csv_writer_reprtext = csv.writer(w_reprtext)    

        csv_writer_projects.writerow(['project_id', 'project_path', 'project_name'])
        csv_writer_packages.writerow(['project_id', 'package_id', 'package_path'])
        csv_writer_classes.writerow(['project_id', 'package_id', 'class_id', 'class_path', 'class_name'])
        csv_writer_methods.writerow(['project_id', 'package_id', 'class_id', 'method_id', 'method_path', 'method_name', 'start_line', 'end_line', 'method_signature'])
        csv_writer_reprtext.writerow(['method_id', 'method_text'])

        # generate meta data --- uuid
        for project in os.listdir(contrib_path):
            if project.startswith("."):
                continue

            project_id   = str(uuid.uuid4())
            project_path = "/projects/" + project.strip() + "/"
            project_name = project

            csv_writer_projects.writerow([project_id, project_path, project_name]) ### NOTE
            proj_path = os.path.join(contrib_path, project)

            for root, dirs, files in os.walk(proj_path):
                for dir in dirs:
                    pack_path = os.path.join(root, dir)
                    package_catalogued = False

                    for file in (os.listdir(pack_path)):
                        if file.startswith("."):
                            continue
                        if file.endswith(".java"):

                            if not package_catalogued:
                                package_id = str(uuid.uuid4())
                                package_path = "/projects/" + project.strip() + pack_path.split(project)[-1] + "/"
                                package_catalogued = True
                                csv_writer_packages.writerow([project_id, package_id, package_path]) ### NOTE

                            class_id   = str(uuid.uuid4())
                            class_path = "/projects/" + project.strip() + pack_path.split(project)[-1] + "/" + file
                            class_name = file.split(".")[0]
                            csv_writer_classes.writerow([project_id, package_id, class_id, class_path, class_name]) ### NOTE

                            # generate method details --- parse class code_text
                            fl_path = os.path.join(root, dir, file)
                            with open(fl_path, 'r') as r:
                                codelines = r.readlines()
                                code_text = ''.join(codelines)

                            lex = None
                            tree = javalang.parse.parse(code_text)    
                            for _, method_node in tree.filter(javalang.tree.MethodDeclaration):
                                startpos, endpos, startline, endline = get_method_start_end(method_node, tree)
                                method_text, startline, endline, lex = get_method_text(startpos, endpos, startline, endline, lex, codelines)

                                method_code = "public class G{ \n" + method_text + "\n }"
                                method_code = '\n'.join(method_code.split('\n'))

                                try:
                                    mtree = javalang.parse.parse(method_code)    
                                except Exception as e:
                                    print("------------------------------------------------------------")
                                    print(f"{ERROR_STATUS} Something went wrong while parsing method!")
                                    print(f"{ERROR_STATUS} CSVs with __UNSET__ values will not be accepted, \nplease correct them or remove them first.")
                                    print("------------------------------------------------------------")
                                    print(method_text)
                                    print("------------------------------------------------------------")
                                    
                                    method_id   = str(uuid.uuid4())
                                    method_path = class_path
                                    start_line  = startline 
                                    end_line    = endline 
                                    method_name = "__UNSET__"
                                    method_signature = "__UNSET__"
                                    csv_writer_methods.writerow([project_id, package_id, class_id, method_id, method_path, method_name, start_line, end_line, method_signature]) ### NOTE
                                    csv_writer_reprtext.writerow([method_id, "__UNSET__"]) ### NOTE                         
                                    continue

                                guess_name = ""
                                for klass in mtree.types:
                                    for m in klass.methods:
                                        guess_name = m.name

                                    for c in klass.constructors:
                                        guess_name = c.name

                                method_id   = str(uuid.uuid4())
                                method_path = class_path
                                start_line  = startline 
                                end_line    = endline 
                                method_name = guess_name
                                method_signature = method_text.split("{")[0]+ ";"

                                print("------------------------------------------------------------")
                                print(method_id)
                                print(method_path)
                                print(start_line, end_line)
                                print(method_name, end="\n\n")
                                print("------------------------------------------------------------")                                
                                print(method_text)
                                print("------------------------------------------------------------", end="\n\n")


                                if method_name != "": # skip interface methods ( e.g., public int square(int a); )
                                    csv_writer_methods.writerow([project_id, package_id, class_id, method_id, method_path, method_name, start_line, end_line, method_signature])  ### NOTE
                                    csv_writer_reprtext.writerow([method_id, method_text])                                                                                        ### NOTE


if __name__ == "__main__":
    s = time.perf_counter()
    main()
    print(time.perf_counter() - s, "seconds.")



            





