**Overview:**
The AI Resume Matchmaker project is designed to develop a cutting-edge AI model that efficiently matches resumes with job descriptions (JDs). By leveraging advanced natural language processing (NLP) and machine learning techniques, the system aims to automate and enhance the recruitment process, ensuring precise alignment between candidate profiles and job requirements.

**Objectives:**
The primary objective of this project is to create a highly accurate and transparent AI-driven system that can extract and analyze key information from resumes and JDs. The model focuses on identifying critical skills, experiences, and qualifications, providing recruiters with a streamlined tool to improve hiring decisions and enhance candidate sourcing.

**Video:**


https://github.com/user-attachments/assets/d22cf525-398f-4249-9599-a438888fba15



**Steps:**
Data extraction, cleaning, and normalization.
Integration of first approach i.e. OpenAI and Qdrant for AI-driven matching.
Integration of second approach i.e. Machine Learning approach.
Implementation of a scoring algorithm to evaluate resume and JD matches.
Development of a user-friendly UI with Streamlit to display results.

**Technology:**
Programming Languages: Python
Frameworks/Libraries: Streamlit, OpenAI API, ML Libraries
Databases: Qdrant
Tools/Platforms: Visual Studio, Git

**Deployment:**
**(NOTE: Openai Approach Not Uploaded only ML approach uploaded due to personal information)**
Using VS Code:
Navigate to the project directory.
Install the required libraries for open ai, ML, preprocessing
Install the openai and qdrant-client packages. 
pip install openai
pip install qdrant-client
Pull the Qdrant Docker image.
docker pull qdrant/qdrant
Run the Qdrant Docker container on port 6333.
docker run -p 6333:6333 qdrant/qdrant
Start the Streamlit application
streamlit run app.py 	(Step f for ML approach)

Using Qdrant Cloud or Google Colab: 
Directly Navigate through Qdrant Dashboard Components 
Get Qdrant API Key and paste in Colab
Directly Run Colab code cells for deployment (Step c for ML approach)

**Screenshots:**
Snippet of UI:
![UI single JD Matched - openai](https://github.com/user-attachments/assets/97690b21-40e8-400a-a020-6554fdd2d137)

Snippet of Multiple JD Approach:
![UI Multiple JD Match](https://github.com/user-attachments/assets/ae248307-2855-4dd5-95f9-37c7dc830c84)

Snippet of ML Approach:
![UI ML Approach - Resume Not Matched](https://github.com/user-attachments/assets/5899fdfd-fd6e-4e4a-a666-0dce63ae1006)





