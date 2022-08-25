import os, sys, csv, time
import pandas as pd
import subprocess


def run_cmd(cmd):
    jarout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = jarout.communicate()
    print(out, err)
    jarout.wait()
    

def main():
    project_path = "/Users/anjandash/Desktop/JEMMA_compiled_project_example_JEMMA/MatthewJWalls-anotherGame/"
    project_splt = project_path.split("/")
    project_name = project_splt[-2] if project_splt[-1] == "" else project_splt[-1]    
    project_jar  = ""

    # generating jar
    if not os.path.exists(project_jar) or project_jar == "":
        print("No jar file found.")
        print("Please make sure the project is compiled i.e. all .java files have corresponding .class files.")

        project_jar = sys.path[0] + "/contrib/tmp/" + project_name + ".jar"
        print("Generating jar file for project ... ")
        print(project_path)
        print(project_jar)
        
        jar_cmd = f"cd {project_path} && jar cf {project_jar} *"
        run_cmd(jar_cmd)

    # generating cg
    if project_jar != "":
        if os.path.exists(project_jar):
            cgjarpath = sys.path[0] + "/dependencies/callgraph/java-callgraph-master/target/javacg-0.1-SNAPSHOT-static.jar"
            cgtxtpath = sys.path[0] + "/contrib/tmp/" + project_name + ".cg.txt"

            cg_cmd = f"java -jar {cgjarpath} {project_jar} > {cgtxtpath}"
            print(cg_cmd)
        else:
            print("Callgraph not generated!")
            print("Because no jar path was given.")
            print("Because the jar creation process failed.")
    else:
        print("Callgraph not generated!")
        print("Because no jar path was given.")
        print("Or because the jar creation process failed.")


    # rename cg
    if os.path.exists(cgtxtpath):
        pass
        #### Copy from: /Users/anjandash/Desktop/NEWCODE_PY/callgraph_renamer_IRONCORE_MULTIPROC_FINAL_OPTIM_LATEST.py
        

if __name__ == "__main__":
    s = time.perf_counter()
    main()
    print(time.perf_counter() - s, "seconds.")







