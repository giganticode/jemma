# jemma


#### This is the official documentation for the JEMMA project

#### Contents

- List of API Calls
- Tutorials



---
#### Exhaustive List of JEMMA API calls

---
- *get_project_id* 
    
    ```Returns the project_id of the project.```

    Parameters:
    * project_name: (str) - name of the project

    Returns:
    * Returns a str uuid of the corresponding project (project_id)
    * Returns None if no project_id was found
    * Returns None if multiple projects were found with the same name
    
---

- *get_project_id_by_path*
    
    ```Returns the project_id of the project (queried with project_path).```

    Parameters:
    * project_path: (str) - path of the project defined in jemma

    Returns:
    * Returns a str uuid of the corresponding project (project_id)
    * Returns None if no project_id was found
    * Returns None if multiple projects were found with the same name
    

---

get_project_id_class_id


---

get_project_id_by_method_id

--- 

get_project_name

---

get_project_path

---

get_project_size_by_classes

---

get_project_size_by_methods

---

get_project_class_ids

---

get_project_method_ids

---


get_project_class_names

---

get_project_method_names

---