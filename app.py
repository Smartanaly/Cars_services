import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import io
import re


class EnhancedCleaningServicesApp:
    def __init__(self):
        """Initialize the app with comprehensive configuration"""
        st.set_page_config(page_title="Advanced Car Services Management Tool", layout="wide")
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.endpoint = (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT_ID/locations/us-central1/publishers/google/models/gemma-7b:predict"
        )
        self.required_columns = [
            'No', 'Name', 'Service', 'Address', 'Pin Code', 'Phone',
            'Areas served', 'WebSite', 'time', 'Review', 'Add On', 'Source'
        ]
        self.df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()

        # Predefined Areas and Services Mapping
        self.area_keywords = {
            'pallavaram': ['pallavaram'],
            'karapakkam': ['karapakkam', 'karapakkam padur'],
            'sholinganallur': ['sholinganallur'],
            'guindy': ['guindy', 'west', 'jafferkhanpet'],
            'porur': ['porur'],
            'thoraipakkam': ['thoraipakkam', 'arumbakkam'],
            'medavakkam': ['medavakkam'],
            'kovur': ['kovur']
        }

        self.service_categories = {
            'cleaning': ['deep clean', 'general cleaning', 'car wash'],
            'maintenance': ['car service', 'repair', 'inspection'],
            'detailing': ['detailing', 'polish', 'interior cleaning']
        }

    def validate_columns(self):
        """Ensure the dataset has all required columns"""
        missing_columns = [col for col in self.required_columns if col not in self.df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns in the data: {', '.join(missing_columns)}")

    def load_data(self, file_path):
        """Enhanced data loading with comprehensive preprocessing"""
        try:
            self.df = pd.read_excel(file_path)
            self.validate_columns()
            self.preprocess_data()
            return True
        except Exception as e:
            st.error(f"Data Loading Error: {str(e)}")
            return False

    def preprocess_data(self):
        """Advanced data preprocessing"""
        self.df.columns = self.df.columns.str.strip()
        text_columns = ['No', 'Name', 'Service', 'Address', 'Areas served']

        for col in text_columns:
            if col in self.df:
                self.df[col] = self.df[col].astype(str).str.lower()

        # Enhanced preprocessing
        self.df['normalized_address'] = self.df['Address'].apply(self._normalize_address)
        self.df['normalized_service'] = self.df['Service'].apply(self._normalize_service)

    def _normalize_address(self, address):
        """Normalize address for better matching"""
        address = str(address).lower()
        for area, keywords in self.area_keywords.items():
            if any(keyword in address for keyword in keywords):
                return area
        return 'other'

    def _normalize_service(self, service):
        """Normalize service for better categorization"""
        service = str(service).lower()
        for category, keywords in self.service_categories.items():
            if any(keyword in service for keyword in keywords):
                return category
        return 'other'

    def generate_response(self, prompt):
        """Generate a response from the Gemma model using API requests"""
        if not self.api_key:
            return "API key not configured."
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "instances": [
                {
                    "content": prompt,
                }
            ],
            "parameters": {
                "temperature": 0.2,
                "maxOutputTokens": 256,
                "topP": 0.8,
                "topK": 40,
            },
        }
        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result.get("predictions", [{}])[0].get("content", "No response generated.")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error sending request: {str(e)}"

    def process_query(self, question):
        """Process natural language query"""
        # Create a prompt with improved context for better understanding
        prompt = f"""
        You are a highly knowledgeable AI assistant specializing in car services. Analyze this question about car services: "{question}"

        Available data:
        - Locations: {', '.join(self.df['Areas served'].unique())}
        - Services: {', '.join(self.df['Service'].unique())}

        Please provide:
        1. Understanding of the query
        2. Relevant search criteria based on areas and services
        3. Recommended services and providers based on the query
        """
        return self.generate_response(prompt)

    def filter_data(self, question):
        """Advanced data filtering with multiple matching strategies"""
        self.filtered_df = self.df.copy()

        # Normalize question
        normalized_question = question.lower()

        # Multiple filtering strategies
        area_matches = [area for area, keywords in self.area_keywords.items()
                        if any(keyword in normalized_question for keyword in keywords)]
        service_matches = [service for service, keywords in self.service_categories.items()
                           if any(keyword in normalized_question for keyword in keywords)]

        # Apply filters
        if area_matches:
            self.filtered_df = self.filtered_df[self.filtered_df['normalized_address'].isin(area_matches)]

        if service_matches:
            self.filtered_df = self.filtered_df[self.filtered_df['normalized_service'].isin(service_matches)]

    def download_excel(self):
        """Provide an option to download the filtered data as an Excel file"""

        @st.cache_data
        def convert_df(df):
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            buffer.seek(0)
            return buffer

        if not self.filtered_df.empty:
            excel_data = convert_df(self.filtered_df)
            st.download_button(
                label="Download Excel Data ",
                data=excel_data,
                file_name='car_services_filtered.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )

    def run(self, file_path):
        """Enhanced Streamlit application"""
        st.title("🚗 Advanced Car Services Assistant")

        if self.load_data(file_path):
            st.subheader("🔍 Intelligent Service Search")
            search_query = st.text_input("Ask about car services...")

            if search_query:
                # Process and display results
                self.filter_data(search_query)

                if not self.filtered_df.empty:
                    st.subheader("Matching Services")
                    # Display all columns
                    st.dataframe(self.filtered_df)

                    # Download option
                    self.download_excel()
                else:
                    st.info("No services match your query.")


if __name__ == "__main__":
    EXCEL_FILE_PATH = "DeepClean_R&D.xlsx"
    app = EnhancedCleaningServicesApp()
    app.run(EXCEL_FILE_PATH)
