import streamlit as st
import os
import tempfile
import zipfile
import docx2txt
import PyPDF2
import io
from typing import Dict, Any, List, Tuple
import re
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class ProjectTools:
    @staticmethod
    def clean_path(path: str) -> str:
        """Clean path string of invalid characters and normalize separators"""
        # Remove carriage returns, newlines, and extra spaces
        path = path.strip().replace('\r', '').replace('\n', '')
        # Replace invalid Windows characters
        path = re.sub(r'[<>:"|?*]', '', path)
        # Normalize path separators
        path = path.replace('/', os.sep).replace('\\', os.sep)
        return path

    @staticmethod
    def scan_directory(path: str, prefix: str = "") -> str:
        """Scan directory and return tree structure as string"""
        tree_str = []
        path = Path(path)
        
        # Get all items in directory
        items = sorted(path.glob("*"))
        dirs = [item for item in items if item.is_dir() and not item.name.startswith(('.', '__'))]
        files = [item for item in items if item.is_file() and not item.name.startswith('.')]
        
        # Process all directories
        for idx, dir_path in enumerate(dirs):
            is_last_dir = (idx == len(dirs) - 1 and not files)
            
            # Add directory to tree
            tree_str.append(f"{prefix}{'└── ' if is_last_dir else '├── '}{dir_path.name}/")
            
            # Recursively scan subdirectory
            sub_prefix = prefix + ('    ' if is_last_dir else '│   ')
            tree_str.append(ProjectTools.scan_directory(dir_path, sub_prefix))
            
        # Process all files
        for idx, file_path in enumerate(files):
            is_last = (idx == len(files) - 1)
            tree_str.append(f"{prefix}{'└── ' if is_last else '├── '}{file_path.name}")
            
        return "\n".join(filter(None, tree_str))

    @staticmethod
    def create_structure(tree_text: str, output_path: str):
        """Create directory structure from tree text"""
        current_path = []
        base_indent = None
        root_created = False
        
        # Split lines and filter out empty ones
        lines = [line for line in tree_text.split('\n') if line.strip()]
        
        for line in lines:
            # Calculate indent level
            indent = len(line) - len(line.lstrip('│├└── '))
            
            if base_indent is None:
                base_indent = indent
            
            # Get relative indent level
            level = (indent - base_indent) // 4 if base_indent is not None else 0
            
            # Clean up the name and remove trailing slashes
            name = ProjectTools.clean_path(line.lstrip('│├└── ').rstrip('/'))
            
            if not name:
                continue
                
            # Handle root directory specially
            if not root_created:
                root_created = True
                root_dir = os.path.join(output_path, name)
                os.makedirs(root_dir, exist_ok=True)
                current_path = [root_dir]
                continue
            
            # Adjust current path based on level
            current_path = current_path[:level + 1]
            
            # Create full path
            full_path = os.path.join(current_path[-1], name)
            
            # Create directory or file
            if line.rstrip().endswith('/'):
                os.makedirs(full_path, exist_ok=True)
                current_path.append(full_path)
            else:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'a') as f:
                    pass

class AIAnalyzer:
    """Handles AI-powered project structure analysis using Google's Gemini"""
    
    def __init__(self):
        # Initialize Gemini
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key not found in environment variables")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def analyze_structure(self, tree_text: str) -> Dict[str, Any]:
        """
        Analyze project structure using Gemini Pro
        Returns detailed analysis and recommendations
        """
        prompt = f"""
        As a senior software architect, analyze this project structure and provide:
        1. Project type identification (based on files and structure)
        2. Detailed structure analysis
        3. Key strengths of the structure
        4. Specific areas for improvement
        5. Best practices recommendations
        6. Security considerations
        7. Overall score out of 100

        Format the response as a JSON object with these exact keys:
        "project_type", "analysis", "strengths" (list), "improvements" (list), 
        "best_practices", "security", "score" (number)

        Project structure to analyze:
        {tree_text}
        """

        try:
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            json_str = response.text
            # Clean up the response if needed
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            # Parse the JSON response
            analysis_result = json.loads(json_str)
            return analysis_result

        except Exception as e:
            st.error(f"Error in AI analysis: {str(e)}")
            return None

def main():
    st.set_page_config(page_title="AI Project Structure Analyzer", layout="wide")
    
    st.title("AI Project Structure Analyzer")
    
    # API Key input in sidebar
    with st.sidebar:
        st.subheader("Google AI Configuration")
        api_key = st.text_input("Google API Key", type="password")
        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key

    tab1, tab2, tab3 = st.tabs(["Structure → Project", "Project → Structure", "AI Analysis"])
    
    with tab1:
        st.header("Generate Project from Structure")
        
        structure_file = st.file_uploader(
            "Upload structure file", 
            type=['txt', 'docx', 'pdf'],
            key="structure_upload",
            help="Upload a file containing your project structure in tree format"
        )
        
        if structure_file:
            try:
                if structure_file.type == "text/plain":
                    tree_text = structure_file.getvalue().decode('utf-8-sig')
                elif structure_file.type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(structure_file)
                    tree_text = "\n".join(page.extract_text() for page in pdf_reader.pages)
                else:  # docx
                    tree_text = docx2txt.process(structure_file)
                
                st.subheader("Detected Structure:")
                st.text(tree_text)
                
                if st.button("Generate Project", key="gen_project"):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        try:
                            ProjectTools.create_structure(tree_text, temp_dir)
                            
                            # Create zip file
                            memory_file = io.BytesIO()
                            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                for root, _, files in os.walk(temp_dir):
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        arcname = os.path.relpath(file_path, temp_dir)
                                        zipf.write(file_path, arcname)
                            
                            memory_file.seek(0)
                            
                            st.download_button(
                                label="Download Project",
                                data=memory_file,
                                file_name="project_structure.zip",
                                mime="application/zip"
                            )
                            
                            st.success("Project generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating project: {str(e)}")
            
            except Exception as e:
                st.error(f"Error processing structure file: {str(e)}")

    with tab2:
        st.header("Generate Structure from Project")
        
        project_file = st.file_uploader(
            "Upload project zip file", 
            type=['zip'],
            key="project_upload",
            help="Upload a zip file containing your project"
        )
        
        if project_file:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(project_file) as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    tree_structure = ProjectTools.scan_directory(temp_dir)
                    
                    st.subheader("Generated Structure:")
                    st.text(tree_structure)
                    
                    st.download_button(
                        label="Download Structure as TXT",
                        data=tree_structure,
                        file_name="project_structure.txt",
                        mime="text/plain"
                    )
                    
                    st.success("Structure generated successfully!")
            
            except Exception as e:
                st.error(f"Error processing project file: {str(e)}")
    
    with tab3:
        st.header("AI Structure Analysis")
        
        structure_input = st.text_area(
            "Enter or paste your project structure:",
            height=300,
            help="Paste your project structure in tree format"
        )
        
        if structure_input and st.button("Analyze with AI"):
            if not os.getenv('GOOGLE_API_KEY'):
                st.error("Please provide your Google API key in the sidebar")
                return
                
            with st.spinner("Analyzing project structure using Gemini AI..."):
                try:
                    analyzer = AIAnalyzer()
                    analysis = analyzer.analyze_structure(structure_input)
                    
                    if analysis:
                        # Display results in an organized layout
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("Project Analysis")
                            st.markdown(f"**Project Type**: {analysis['project_type']}")
                            st.markdown(analysis['analysis'])
                            
                            st.subheader("Strengths")
                            for strength in analysis['strengths']:
                                st.markdown(f"✓ {strength}")
                            
                            st.subheader("Suggested Improvements")
                            for improvement in analysis['improvements']:
                                st.markdown(f"• {improvement}")
                        
                        with col2:
                            # Create a color for the score
                            score = float(analysis['score'])
                            color = 'green' if score >= 80 else 'orange' if score >= 60 else 'red'
                            st.markdown(f"<h1 style='text-align: center; color: {color};'>{score}%</h1>", 
                                      unsafe_allow_html=True)
                            
                            st.subheader("Security Considerations")
                            st.markdown(analysis['security'])
                        
                        st.divider()
                        
                        # Best practices section
                        st.subheader("Best Practices Recommendations")
                        st.markdown(analysis['best_practices'])
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.error("Make sure your Google API key is valid and has access to Gemini Pro")

    # Add helpful tips in sidebar
    with st.sidebar:
        st.subheader("Tips")
        st.markdown("""
        1. Get a Google API key from Google AI Studio
        2. Enter your project structure in tree format
        3. Click 'Analyze with AI' for detailed insights
        4. Review recommendations and improve your structure
        """)
        
        st.subheader("Example Structure")
        st.code("""my_project/
├── src/
│   ├── main.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   └── test_main.py
└── README.md""")

if __name__ == "__main__":
    main()