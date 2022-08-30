# jemma


#### This is the official documentation for the JEMMA project

#### Contents


- [Setup Instructions](####Setup-Instructions)
- [Getting to know JEMMA Datasets](###Getting-to-know-JEMMA-Datasets)
    - JEMMA Metadata
    - JEMMA Representations
    - JEMMA Properties
    - JEMMA Callgraphs

- Working with JEMMA Workbench
    - List of API Calls
        - [projects](##projects)
        - [classes](##classes)
        - [methods](##methods) 
        - [basic utils](##basic-utils) 
        - [task utils](##task-utils)
    - Use-Case Tutorials


---


#### Setup Instructions

Getting started with jemma


---


### Getting to know JEMMA Datasets

#### JEMMA Metadata

---

#### JEMMA Representations

Table 

Representation Name --- Representation Code --- Link to dataset


---

#### JEMMA Properties

Table 

Property Name --- Property Code --- Link to dataset

---

#### JEMMA Callgraphs


---

### Working with JEMMA Workbench



#### List of JEMMA API calls
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
    """

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

---