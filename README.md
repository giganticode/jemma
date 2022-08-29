# jemma


#### This is the official documentation for the JEMMA project

#### Contents

- List of API Calls
- Tutorials



---
#### Exhaustive List of JEMMA API calls
---



##    *projects*    #<br/>

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


##    *classes*    #<br/>

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
    * Returns a str of the corresponding project path
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


##    *methods*    #<br/>

- *get_method_id*

---

- *get_method_path*

---

- *get_start_line*

---

- *get_end_line*

---

- *get_method_metadata*

---
