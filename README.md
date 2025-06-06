# GenAI for SD - Project 3
# Jackson Taylor
# LAST EDITED 4-24-2025

**REPO INFORMATION** 

You find all the code used to automate the process in the src file. 

The file *models.py* contains all the relevant information for API calls to the four different models. 

The file *prompting_strategies.py* implements how to manage the messages and feed them to the models to properly make a prompting strategy. 

The file *main.py* reads from task files in the "data/input" end point (given as numbers in cmd-line arguments) and loops until the file is over, taking the first two arguments as the Model and Strategy, and the next arguments (number depending on the strategy) as the prompt text. The outputs of each task file is written to a file of the same name in the "data/output" folder, including the model, strategy, and response.

**HOW TO INSTALL GIT REPOSITORY**

    PS C:\path\to\directory > git clone https://github.com/jtaylor05/genai_proj_3

**HOW TO RUN AUTOMATION CODE**

    **MAKE A VENV**

    For a Powershell environment:

    PS C:\path\to\directory > python3 -m venv venv_name

    **OPEN VENV**

    For Powershell environment:

    PS C:\path\to\directory > .\venv_name\Scripts\activate

    **INSTALL DEPENDENCIES**

    For Powershell environment:

    PS C:\path\to\directory > python3 -m pip3 install -r requirements.txt

    **RUN MAIN.PY**

    For Powershell:

    PS C:\path\to\directory > python3 main.py [list int arguments of task files to be run]
    
**PURPOSE OF THE ASSIGNMENT**

I think the purpose of the assignment is 2-fold: First, it is to test the students understanding of the prompting strategies and their different effects on model responses and which strategies are better suited for which types of tasks, secondly, it is to see the students ability to automate their coding work with LLMs. 

On the first, I think this point could have been made with much fewer tasks. This leads to my reasoning behind the second: the sheer number of tasks suggests to me that much (if not all) of the pipeline was meant to be automated. 

I think I could have automated more, however, the week-long time limit left me a little strapped for time and confidence to properly automate and test code for all of the parts of this project. I instead opted to do many of the parts by hand which, while perhaps antithetical to the purpose of this class, was required based on the time I had. I am happy with my few short python scripts however I would have preferred to have more time so I could better utilise the models. I will most likely be playing with these scripts long after this course.