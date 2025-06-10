import streamlit as st
import os
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
from openai import OpenAI
import time

# Page config
st.set_page_config(
    page_title="HubSpot AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stChat {
        height: 600px;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff7a45;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class HubSpotChatbot:
    """A chatbot that interprets natural language queries and fetches data from HubSpot API."""
    
    def __init__(self, hubspot_api_key: str, openai_api_key: str):
        """Initialize the chatbot with API keys."""
        self.hubspot_api_key = hubspot_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.base_url = "https://api.hubapi.com"
        
        # Define available HubSpot endpoints and their purposes
        self.available_endpoints = {
            "deals": {
                "endpoint": "/crm/v3/objects/deals",
                "description": "Fetch deals data",
                "properties": ["dealname", "amount", "dealstage", "closedate", "pipeline", "hubspot_owner_id"],
                "filters": ["dealstage", "pipeline", "amount"]
            },
            "contacts": {
                "endpoint": "/crm/v3/objects/contacts",
                "description": "Fetch contacts data",
                "properties": ["firstname", "lastname", "email", "phone", "company", "jobtitle"],
                "filters": ["email", "company"]
            },
            "companies": {
                "endpoint": "/crm/v3/objects/companies",
                "description": "Fetch companies data",
                "properties": ["name", "domain", "industry", "city", "state", "country", "numberofemployees"],
                "filters": ["industry", "city", "state"]
            }
        }
        
        # Deal stages mapping (customize based on your HubSpot setup)
        self.deal_stages = {
            "appointmentscheduled": "Appointment Scheduled",
            "qualifiedtobuy": "Qualified to Buy",
            "presentationscheduled": "Presentation Scheduled",
            "decisionmakerboughtin": "Decision Maker Bought-In",
            "contractsent": "Contract Sent",
            "closedwon": "Closed Won",
            "closedlost": "Closed Lost"
        }

    def interpret_query(self, user_query: str) -> Dict[str, Any]:
        """Use OpenAI to interpret the user's natural language query."""
        system_prompt = f"""You are a HubSpot API query interpreter. Analyze the user's question and extract:
        1. The object type they're asking about (deals, contacts, or companies)
        2. Any filters they want to apply
        3. What specific information they want to see
        
        Available deal stages: {', '.join(self.deal_stages.values())}
        
        Return a JSON object with:
        - object_type: "deals", "contacts", or "companies"
        - filters: object with filter criteria
        - properties: array of properties to return
        - limit: number of results (default 10)
        - summary_type: "list", "count", "total" (for aggregations)
        
        Examples:
        "What deals are in negotiation?" -> {{"object_type": "deals", "filters": {{"dealstage": "decisionmakerboughtin"}}, "properties": ["dealname", "amount", "closedate"], "limit": 10, "summary_type": "list"}}
        "How many contacts do we have?" -> {{"object_type": "contacts", "filters": {{}}, "properties": ["email"], "limit": 1000, "summary_type": "count"}}
        "Total value of deals in contract sent stage?" -> {{"object_type": "deals", "filters": {{"dealstage": "contractsent"}}, "properties": ["amount"], "limit": 1000, "summary_type": "total"}}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error interpreting query: {e}")
            return {
                "object_type": "deals",
                "filters": {},
                "properties": ["dealname", "amount"],
                "limit": 10,
                "summary_type": "list"
            }

    def build_api_request(self, interpretation: Dict[str, Any]) -> Dict[str, Any]:
        """Build the HubSpot API request based on the interpreted query."""
        object_type = interpretation.get("object_type", "deals")
        endpoint_info = self.available_endpoints.get(object_type, self.available_endpoints["deals"])
        
        # Build the API URL
        url = f"{self.base_url}{endpoint_info['endpoint']}"
        
        # Build query parameters
        params = {
            "limit": min(interpretation.get("limit", 10), 100),  # HubSpot API limit
            "properties": ",".join(interpretation.get("properties", endpoint_info["properties"][:3]))
        }
        
        # Add filters if any
        filters = interpretation.get("filters", {})
        if filters:
            filter_groups = []
            for key, value in filters.items():
                # Map deal stage names to internal values
                if key == "dealstage" and object_type == "deals":
                    # Find the internal stage key
                    for stage_key, stage_name in self.deal_stages.items():
                        if stage_name.lower() in value.lower() or value.lower() in stage_name.lower():
                            value = stage_key
                            break
                
                filter_groups.append({
                    "filters": [{
                        "propertyName": key,
                        "operator": "EQ",
                        "value": value
                    }]
                })
            
            if filter_groups:
                params["filterGroups"] = json.dumps(filter_groups)
        
        return {
            "url": url,
            "params": params,
            "headers": {
                "Authorization": f"Bearer {self.hubspot_api_key}",
                "Content-Type": "application/json"
            }
        }

    def fetch_hubspot_data(self, request_config: Dict[str, Any]) -> Dict[str, Any]:
        """Make the actual API call to HubSpot."""
        try:
            response = requests.get(
                request_config["url"],
                params=request_config["params"],
                headers=request_config["headers"]
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data from HubSpot: {e}")
            return {"results": [], "error": str(e)}

    def format_response(self, data: Dict[str, Any], interpretation: Dict[str, Any]) -> str:
        """Format the API response into natural language."""
        if "error" in data:
            return f"❌ Sorry, I encountered an error: {data['error']}"
        
        results = data.get("results", [])
        if not results:
            return "🔍 I couldn't find any matching records."
        
        object_type = interpretation.get("object_type", "records")
        summary_type = interpretation.get("summary_type", "list")
        
        # Handle different summary types
        if summary_type == "count":
            return f"📊 Total {object_type}: **{len(results)}**"
        
        elif summary_type == "total" and object_type == "deals":
            total = sum(float(r.get("properties", {}).get("amount", 0) or 0) for r in results)
            return f"💰 Total deal value: **${total:,.2f}** across {len(results)} deals"
        
        # Build the list response
        response_parts = [f"📋 I found **{len(results)} {object_type}**:\n"]
        
        for i, result in enumerate(results[:10], 1):  # Show first 10 results
            properties = result.get("properties", {})
            
            if object_type == "deals":
                name = properties.get("dealname", "Unnamed Deal")
                amount = properties.get("amount", "0")
                stage = properties.get("dealstage", "N/A")
                closedate = properties.get("closedate", "N/A")
                
                # Convert stage to readable name
                if stage in self.deal_stages:
                    stage = self.deal_stages[stage]
                
                # Format amount
                try:
                    amount_num = float(amount)
                    amount_str = f"${amount_num:,.2f}"
                except:
                    amount_str = "N/A"
                
                response_parts.append(f"**{i}.** 🏷️ {name}")
                response_parts.append(f"   - 💵 Amount: {amount_str}")
                response_parts.append(f"   - 📊 Stage: {stage}")
                if closedate != "N/A":
                    response_parts.append(f"   - 📅 Close Date: {closedate[:10]}")
                response_parts.append("")
                
            elif object_type == "contacts":
                first = properties.get("firstname", "")
                last = properties.get("lastname", "")
                email = properties.get("email", "N/A")
                company = properties.get("company", "N/A")
                name = f"{first} {last}".strip() or "Unnamed Contact"
                
                response_parts.append(f"**{i}.** 👤 {name}")
                response_parts.append(f"   - 📧 Email: {email}")
                if company != "N/A":
                    response_parts.append(f"   - 🏢 Company: {company}")
                response_parts.append("")
                
            elif object_type == "companies":
                name = properties.get("name", "Unnamed Company")
                industry = properties.get("industry", "N/A")
                domain = properties.get("domain", "N/A")
                
                response_parts.append(f"**{i}.** 🏢 {name}")
                if industry != "N/A":
                    response_parts.append(f"   - 🏭 Industry: {industry}")
                if domain != "N/A":
                    response_parts.append(f"   - 🌐 Domain: {domain}")
                response_parts.append("")
        
        if len(results) > 10:
            response_parts.append(f"\n_...and {len(results) - 10} more results._")
        
        return "\n".join(response_parts)

    def process_query(self, user_query: str) -> str:
        """Main method to process a user query end-to-end."""
        # Step 1: Interpret the query
        interpretation = self.interpret_query(user_query)
        
        # Step 2: Build API request
        request_config = self.build_api_request(interpretation)
        
        # Step 3: Fetch data from HubSpot
        data = self.fetch_hubspot_data(request_config)
        
        # Step 4: Format response
        response = self.format_response(data, interpretation)
        
        return response


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 HubSpot AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your HubSpot data in natural language</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key inputs
        hubspot_key = st.text_input(
            "HubSpot API Key",
            type="password",
            value=st.session_state.get("hubspot_key", ""),
            help="Your HubSpot private app access token"
        )
        
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get("openai_key", ""),
            help="Your OpenAI API key for natural language processing"
        )
        
        # Save keys to session state
        if hubspot_key:
            st.session_state.hubspot_key = hubspot_key
        if openai_key:
            st.session_state.openai_key = openai_key
        
        # Check if keys are provided
        keys_provided = bool(hubspot_key and openai_key)
        
        if keys_provided:
            st.success("✅ API keys configured")
        else:
            st.warning("⚠️ Please provide both API keys")
        
        st.divider()
        
        # Example queries
        st.header("💡 Example Queries")
        example_queries = [
            "What deals are in negotiation?",
            "Show me all deals worth more than $10,000",
            "List contacts from Acme Corporation",
            "How many contacts do we have?",
            "Total value of deals in contract sent stage?",
            "What companies are in the technology industry?",
            "Show me deals closing this month"
        ]
        
        for query in example_queries:
            if st.button(f"📝 {query}", key=f"example_{query}"):
                st.session_state.current_query = query
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about your HubSpot data..."):
            if not keys_provided:
                st.error("❌ Please provide both API keys in the sidebar first!")
            else:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Searching HubSpot data..."):
                        try:
                            # Initialize chatbot
                            chatbot = HubSpotChatbot(
                                st.session_state.hubspot_key,
                                st.session_state.openai_key
                            )
                            
                            # Process query
                            response = chatbot.process_query(prompt)
                            st.markdown(response)
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                        except Exception as e:
                            error_msg = f"❌ An error occurred: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Handle example query button clicks
        if "current_query" in st.session_state:
            query = st.session_state.current_query
            del st.session_state.current_query
            
            if keys_provided:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": query})
                
                # Generate response
                try:
                    chatbot = HubSpotChatbot(
                        st.session_state.hubspot_key,
                        st.session_state.openai_key
                    )
                    response = chatbot.process_query(query)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    error_msg = f"❌ An error occurred: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()
    
    with col2:
        st.header("📊 Quick Stats")
        
        if keys_provided:
            # Add some quick stats buttons
            st.markdown("### Quick Actions")
            
            if st.button("📈 Total Deals Count"):
                st.session_state.current_query = "How many deals do we have in total?"
            
            if st.button("💰 Pipeline Value"):
                st.session_state.current_query = "What's the total value of all deals?"
            
            if st.button("👥 Contact Count"):
                st.session_state.current_query = "How many contacts are in the system?"
            
            if st.button("🏢 Company Count"):
                st.session_state.current_query = "How many companies do we have?"
            
            st.divider()
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
        else:
            st.info("👈 Configure API keys in the sidebar to get started")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit, OpenAI, and HubSpot API</p>
        <p>💡 Tip: You can ask complex questions like "Show me all deals from technology companies closing this quarter"</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
