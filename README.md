<!-- README -->

![](./assets/img/jemma-cover-main.png)


#### This is the official documentation for the JEMMA project

*JEMMA is an Extensible Java dataset for Many ML4Code Applications.* It is primarily a dataset of Java code entities, their properties, and representations. 
To help users interact and work with the data seamlessly, we have added Workbench capabilities to it as well. 

This repository hosts the Workbench part of JEMMA, while the raw data is hosted on Zenodo which can be downloaded at any moment while using the Workbench. The following sections provide more details.   


---
#### Contents


- [Setup Instructions](#setup-instructions)
- [Getting to know JEMMA Datasets](#getting-to-know-jemma-datasets)
    - [JEMMA Metadata](#jemma-metadata)
    - [JEMMA Representations](#jemma-representations)
    - [JEMMA Properties](#jemma-properties)
    - [JEMMA Callgraphs](#jemma-callgraphs)

- [Working with JEMMA Workbench](#working-with-jemma-workbench)
    - [List of API Calls](#list-of-api-calls)
        - [projects](#projects)       
        - [classes](#classes)         
        - [methods](#methods)         
        - [basic utils](#basic-utils) 
        - [task utils](#task-utils)   
    - [Use-Case Tutorials](#use-case-tutorials)


---

<a id="setup-instructions"></a>

#### Setup Instructions 

<!-- > Getting started with jemma -->


> First steps: Install jemma locally
```
1. $ git clone https://github.com/giganticode/jemma.git 
2. $ cd jemma/ 
3. $ pip install -r requirements.txt
4. $ pip install -e .
```

> Next steps: Downloading all the datasets <br>
> Sign-up to Zenodo.org and generate an API num_token [IMPORTANT!]

```
5. $ cd jemma/download/ 
6. $ nano config.ini (& replace the dummy `access_token` with your API key)
7. $ python3 download.py 
8. $ python3 sanity_checks.py
```



---


### Getting to know JEMMA Datasets

#### JEMMA Metadata

| Link to metadata     | columns      |
|:---------|:----------   |
| [*projects*](https://doi.org/10.5281/zenodo.5807578/) | *project_id* |
|  | *project_path* |
|  | *project_name* |
|  |  ||
| [*packages*](https://doi.org/10.5281/zenodo.5807586/) | *project_id* | 
|  | *package_id*   |
|  | *package_path* | 
|  | *package_name* |
|  |  ||
| [*classes*](https://doi.org/10.5281/zenodo.5808902/) | *project_id* |
|  | *package_id* |
|  | *class_id*   |
|  | *class_path* |
|  | *class_name* |
|  |  ||
| [*methods*](https://doi.org/10.5281/zenodo.5813089/) | *project_id* |
|  | *package_id*  |
|  | *class_id*    |
|  | *method_id*   |
|  | *method_name* |
|  | *start_line*  |
|  | *end_line*    |


---

#### JEMMA Representations

| Representation Code           | Representation Name   | Link to dataset   |
|:------------------------------|:----------------------|:------------------|
| TEXT                          | raw_source_code       | https://doi.org/10.5281/zenodo.5813705 |
| TKNA                          | code_tokens (spaced)  | https://doi.org/10.5281/zenodo.5813717 |
| TKNB                          | code_tokens (comma)   | https://doi.org/10.5281/zenodo.5813730 |   
| C2VC                          | code2vec*              | https://doi.org/10.5281/zenodo.5813993 |
| C2SQ                          | code2seq*              | https://doi.org/10.5281/zenodo.5814059 |
| FTGR                          | feature_graph*         | https://doi.org/10.5281/zenodo.5813933 |




---

#### JEMMA Properties

| Property Code $~~~~~~~~~$     | Property Name $~~~~~~~~~$        | Link to dataset   |
|:------------------------------|:---------------------------------|:------------------|
| RSLK                          | resource_leak | https://doi.org/10.5281/zenodo.1096082 |
| NLDF                          | null_dereference | https://doi.org/10.5281/zenodo.1096080 |
| NMLC                          | num_local_calls | https://doi.org/10.5281/zenodo.7020084 |
| NMNC                          | num_non_local_calls | https://doi.org/10.5281/zenodo.7019960 |
| NUCC                          | num_unique_callees | https://doi.org/10.5281/zenodo.7019176 |
| NUPC                          | num_unique_callers | https://doi.org/10.5281/zenodo.7019128 |
| CMPX                          | cyclomatic_complexity | https://doi.org/10.5281/zenodo.5813084 |
| MXIN                          | max_indent | https://doi.org/10.5281/zenodo.5813081 |
| NAME                          | method_name | https://doi.org/10.5281/zenodo.5813308 |
| NMLT                          | num_literals | https://doi.org/10.5281/zenodo.5813054 |
| NMOP                          | num_operators |https://doi.org/10.5281/zenodo.5813055 |
| NMPR                          | num_parameters | https://doi.org/10.5281/zenodo.5813053 |
| NMRT                          | num_returns | https://doi.org/10.5281/zenodo.5813034 |
| NMTK                          | num_tokens | https://doi.org/10.5281/zenodo.5813032 |
| NTID                          | num_identifiers | https://doi.org/10.5281/zenodo.5813029 |
| NUID                          | num_unique_identifiers | https://doi.org/10.5281/zenodo.5813028 |
| SLOC                          | source_lines_of_code | https://doi.org/10.5281/zenodo.5813094 |
| TLOC                          | total_lines_of_code | https://doi.org/10.5281/zenodo.5813102 |    



<!-- \textit{Properties:} \texttt{[TLOC]}  & \url{} & 335.5 MB\Tstrut{}\\
\textit{Properties:} \texttt{[SLOC]}  & \url{} & 335.0 MB  \\

\textit{Properties:} \texttt{[NUID]}  & \url{} & 335.6 MB  \\
\textit{Properties:} \texttt{[NTID]}  & \url{} & 336.7 MB  \\ 
\textit{Properties:} \texttt{[NMTK]}  & \url{} & 342.5 MB  \\
\textit{Properties:} \texttt{[NMRT]}  & \url{} & 333.3 MB  \\
\textit{Properties:} \texttt{[NMPR]}  & \url{} & 333.3 MB  \\
\textit{Properties:} \texttt{[NMOP]}  & \url{} & 334.5 MB  \\ 
\textit{Properties:} \texttt{[NMLT]}  & \url{} & 333.4 MB  \\
\textit{Properties:} \texttt{[NAME]}  & \url{} & 432.0 MB  \\
\textit{Properties:} \texttt{[MXIN]}  & \url{} & 267.0 MB  \\
\textit{Properties:} \texttt{[CMPX]}  & \url{} & 267.1 MB\Bstrut{}\\ 


\textit{Properties:} \texttt{[NUPC]}  & \url{} & 333.3 MB  \\
\textit{Properties:} \texttt{[NUCC]}  & \url{} & 333.6 MB  \\

\textit{Properties:} \texttt{[NMNC]}  & \url{} & 334.0 MB  \\
\textit{Properties:} \texttt{[NMLC]}  & \url{} & 333.2 MB  \\


% \textit{Properties:} \texttt{[NMTC]}  & \url{https://doi.org/10.5281/zenodo.7019246} & 334.0 MB\Bstrut{}\\ 


\textit{Properties:} \texttt{[NLDF]}  & \url{} & 333.6 MB  \\
\textit{Properties:} \texttt{[RSLK]}  & \url{} & 334.0 MB\Bstrut{}\\  -->


---

#### JEMMA Callgraphs

| Link to callgraphs data | columns |
|:----------------------- |:------- |
| [Callgraphs](https://doi.org/10.5281/zenodo.6758937) | *caller_project_id* |
| | *caller_class_id* |
| | *caller_method_id* |
| | *call_direction* |
| | *callee_project_id* |
| | *callee_class_id* |
| | *callee_method_id* |
<!-- | | *call_type* | -->


---

### Working with JEMMA Workbench

#### List of API calls
---



## *projects* 

- *get_project_id* 
    
    ```Returns the project_id of the project (queried by project name).```

    Parameters:
    * project_name: (str) - name of the project

    Returns:
    * Returns a str uuid of the corresponding project (project_id)
    * Returns None if no such project_id was found
    * Returns None if multiple projects were found with the same name
    
---

- *get_project_id_by_path*
    
    ```Returns the project id of the project (queried with project path).```

    Parameters:
    * project_path: (str) - path of the project defined in jemma

    Returns:
    * Returns a str uuid of the corresponding project (project_id)
    * Returns None if no such project_path was found
    * Returns None if multiple projects were found with the same path
    

---

- *get_project_id_class_id*
    
    ```Returns the project id of the project (queried with class id)```

    Parameters:
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a str uuid of the corresponding project (project_id)
    * Returns None if no such project_id was found
    

---

- *get_project_id_by_method_id*
    
    ```Returns the project id of the project (queried with method id)```

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a str uuid of the corresponding project (project_id)
    * Returns None if no such project_id was found
    

--- 

- *get_project_name*
    
    ```Returns the project name of the project.```

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a str of the corresponding project name
    * Returns None if no such project_id is defined in jemma
    

---

- *get_project_path*
    
    ```Returns the project path of the project.```

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a str of the corresponding project path
    * Returns None if no such project_id is defined in jemma
    

---

- *get_project_size_by_classes*
    
    ```Returns the size of a project, by the number of classes.``` 

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a str of the corresponding project size, by the number of classes
    * Returns None if no such project_id is defined in jemma
    

---

- *get_project_size_by_methods*
    
    ```Returns the size of a project, by the number of methods.```

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a str of the corresponding project size, by the number of methods
    * Returns None if no such project_id is defined in jemma
    

---

- *get_project_class_ids*
    
    ```Returns all class ids defined within the project.```

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a (List[str]) corresponding to all class ids in the project   
    * Returns an empty List if no classes are found
    

---

- *get_project_method_ids*
    
    ```Returns all method ids defined within the project.```

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a (List[str]) corresponding to all method ids in the project   
    * Returns an empty List if no methods are found
    

---


- *get_project_class_names*
    
    ```Returns all class names defined within the project.```

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a (List[str]) corresponding to all class names in the project   
    * Returns an empty List if no classes are found
    

---

- *get_project_method_names*
    
    ```Returns all method names defined within the project.```

    Parameters:
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a (List[str]) corresponding to all method names in the project   
    * Returns an empty List if no methods are found
    
---

- *get_project_metadata*
    
    ```Returns all metadata related to a particular project.```

    Parameters: 
    * project_id: (str) - any project_id defined within jemma

    Returns:
    * Returns a dictionary of project metadata values
    * Returns None if no such project_id is defined in jemma
    


---


## *classes*

- *get_class_id*
    
    ```Returns the class id of a class in project (queried by class name).```

    Parameters:
    * project_id: (str) - project_id of a project
    * class_name: (str) - class name of a class within the project

    Returns:
    * Returns a str uuid of the corresponding class (class_id)
    * Returns None if no such project_id or class_name was found
    * Returns None if multiple classes were found with the same name (use: get_class_id_by_path)


---

- *get_class_id_by_path*
    
    ```Returns the class id of a class (queried with class path).```

    Parameters:
    * class_path: (str) - path of the class defined in jemma

    Returns:
    * Returns a str uuid of the corresponding class (class_id)
    * Returns None if no such class_path was found
    * Returns None if multiple classes were found with the same path
    
---

- *get_class_id_by_method_id*
    
    ```Returns the class id of a class (queried with method id)```

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a str uuid of the corresponding class (class_id)
    * Returns None if no such class_id was found
    

---

- *get_class_name*
    
    ```Returns the class name of a particular class.```

    Parameters:
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a str of the corresponding class name
    * Returns None if no such class_id is defined in jemma
    
---

- *get_class_path*
    
    ```Returns the class path of a particular class.```

    Parameters:
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a str of the corresponding class path
    * Returns None if no such class_id is defined in jemma
    

---

- *get_class_size_by_methods*
    
    ```Returns the size of a class, by the number of methods.```

    Parameters:
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a str of the corresponding class size, by the number of methods
    * Returns None if no such class_id is defined in jemma
    

---

- *get_class_method_ids*
    
    ```Returns all method ids defined within a particular class.```

    Parameters:
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a (List[str]) corresponding to all method ids in the class   
    * Returns an empty List if no methods are found
    

---

- *get_class_method_names*
    
    ```Returns all method names within a particular class.```

    Parameters:
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a (List[str]) corresponding to all method names in the class   
    * Returns an empty List if no methods are found
    

---

- *get_class_metadata*
    
    ```Returns all metadata related to a particular class.```

    Parameters: 
    * class_id: (str) - any class_id defined within jemma

    Returns:
    * Returns a dictionary of class metadata values
    * Returns None if no such class_id is defined in jemma
    

---


## *methods*

- *get_method_id*
    
    ```Returns the method id of a method in a class (queried by method name).```

    Parameters:
    * class_id: (str) - any class_id defined within jemma
    * method_name: (str) - method name of a method within the class

    Returns:
    * Returns a str uuid of the corresponding method (method_id)
    * Returns None if no such class_id or method_name was found
    * Returns None if multiple methods were found with the same name (use: get_method_id_stln_enln)
    

---

- *get_method_id_by_stln_enln*
    
    ```Returns the method id of a method in a class (queried by method name, start line, and end line).```

    Parameters:
    * class_id: (str) - any class_id defined within jemma
    * method_name: (str) - method name of a method within the class
    * stln: (str) - start line of the method within the class
    * enln: (str) - end line of the method within the class

    Returns:
    * Returns a str uuid of the corresponding method (method_id)
    * Returns None if no such class_id or method_name was found
    

---

- *get_method_path*
    
    ```Returns the class path of the parent class of a method.```

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a str of the corresponding class path
    * Returns None if no such method_id is defined in jemma
    

---

- *get_start_line*
    
    ```Returns the start line of a particular method```

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a str of the corresponding start line of the method
    * Returns None if no such method_id is defined in jemma
    

---

- *get_end_line*
    
    ```Returns the end line of a particular method```

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a str of the corresponding end line of the method
    * Returns None if no such method_id is defined in jemma
    

---

- *get_method_metadata*
    
    ```Returns all metadata related to a particular method.```

    Parameters: 
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a dictionary of method metadata values
    * Returns None if no such method_id is defined in jemma
    

---

## *basic utils*

- *get_properties*
    
    ```Get property values for a list of methods.```

    Parameters:
    * property : (str) - property code
    * methods : (list[str]) - list of unique methods ids

    Returns:
    * pandas Dataframe object (with method_id, property) of the passed list of methods
    
---

- *get_balanced_properties*
    
    ```Get balanced property values for a list of methods.```

    Parameters:
    * property : (str) - property code
    * methods : (list[str]) - list of unique methods ids [OPTIONAL]

    Returns:
    * pandas Dataframe object (with method_id, property) of the passed list of methods
    
---

- *get_representations*
    
    ```Get representation values of a list of methods.```

    Parameters:
    * representation : (str) - representation code
    * methods : (list[str]) - list of unique methods ids

    Returns:
    * pandas Dataframe object (with method_id, representation) of the passed list of methods
    
---

- *get_callees*
    
    ```Get a list of method ids for direct callees of a particular method.```

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a (List[str]) of method ids for direct callees
    * Returns an empty List if no such method_id exists
    

---

- *get_callers*
    
    ```Get a list of method ids for direct callers of a particular method.```

    Parameters:
    * method_id: (str) - any method_id defined within jemma

    Returns:
    * Returns a (List[str]) of method ids for direct callers
    * Returns an empty List if no such method_id exists

---

- *get_caller_context*
    
    ```Get all caller method ids from n-hop neighborhood for a particular method.```

    Parameters:
    * method_id: (str) - method_id for which callers are to be determined
    * n_neighborhood: (int) - size of n-hop neighborhood callers that are to be considered
    * df: (pandas Dataframe) - pandas Dataframe containing the caller-callee data for the project

    Returns:
    * Returns a (List[str]) of caller method ids 
    * Returns an empty List if no callers could be found for method_id
    * Returns an empty List if n_neighborhood is 0    
    

---

- *get_callee_context*
    
    ```Get all callee method ids from n-hop neighborhood for a particular method.```

    Parameters:
    * method_id: (str) - method_id for which callees are to be determined
    * n_neighborhood: (int) - size of n-hop neighborhood callees that are to be considered
    * df: (pandas Dataframe) - pandas Dataframe containing the caller-callee data for the project

    Returns:
    * Returns a (List[str]) of callee method ids 
    * Returns an empty List if no callees could be found for method_id
    * Returns an empty List if n_neighborhood is 0    
    
---

## *task utils*    #<br/>

- *gen_TKNA_from_method_text*
    
    ```Process the method text of a method and returns the TKNA representation.```

    Parameters:
    * method_id: (str) - method_id for which TKNA representation is to be generated
    * method_text: (str) - corresponding method_text for the method_id 

    Returns:
    * Returns the TKNA representation of a method
    
---

- *gen_TKNB_from_method_text*
    
    ```Process the method text of a method and returns the TKNB representation.```

    Parameters:
    * method_id: (str) - method_id for which TKNB representation is to be generated
    * method_text: (str) - corresponding method_text for the method_id 

    Returns:
    * Returns the TKNB representation of a method
    
---

- *gen_C2VC_from_method_text*
    
    ```Process the method text of a method and returns the C2VC representation.```

    Parameters:
    * method_id: (str) - method_id for which C2VC representation is to be generated
    * method_text: (str) - corresponding method_text for the method_id 

    Returns:
    * Returns the C2VC representation of a method
    

---

- *gen_C2SQ_from_method_text*
    
    ```Process the method text of a method and returns the C2SQ representation.```

    Parameters:
    * method_id: (str) - method_id for which C2SQ representation is to be generated
    * method_text: (str) - corresponding method_text for the method_id 

    Returns:
    * Returns the C2SQ representation of a method
    

---

- *gen_FTGR_from_method_text*
    
    ```Process the method text of a method and returns the FTGR representation.```

    Parameters:
    * method_id: (str) - method_id for which FTGR representation is to be generated
    * method_text: (str) - corresponding method_text for the method_id 

    Returns:
    * Returns the FTGR representation of a method
    
---


- *gen_representation*
    
    ```Process the method text of a method and returns the selected representation.```

    Parameters:
    * representation: (str) - representation (code) which is to be generated
    * method_id: (str) - method_id for which the representation is to be generated
    * method_text: (str) - corresponding method_text for the method 

    Returns:
    * Returns the selected representation for the method
    
---

- *get_properties*
    
    ```Get property values for a list of methods.```

    Parameters:
    * property : (str) - property code
    * methods : (list[str]) - list of unique methods ids

    Returns:
    * pandas Dataframe object (with method_id, property) of the passed list of methods
    
---

- *run_models*
    
    ```Trains/finetunes a set of models for a given task and representation, from the specified data```

    Parameters:
    * property: (str) - property (code) which is to be used 
    * representation: (str) - representation (code) which is to be used 
    * train_methods: (List[str]) - list of methods (method_ids) to be considered as training samples
    * test_methods: (List[str]) - list of methods (method_ids) to be considered as test samples
    * models: (List[str]) - List of models (huggingface paths or codes) to train and evaluate

    Returns:
    * None: Prints the evaluation scores for each model
    
---

#### Use-Case Tutorials [COMING SOON]