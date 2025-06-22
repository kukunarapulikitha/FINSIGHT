"""
LLM Evaluation Module for Financial Analysis System
This module evaluates and compares different LLMs for various financial analysis tasks
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
from typing import Dict, List, Any, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Evaluation metrics weights
METRIC_WEIGHTS = {
    "accuracy": 0.25,
    "response_time": 0.20,
    "cost_efficiency": 0.15,
    "context_understanding": 0.20,
    "output_quality": 0.20
}

# LLM characteristics and justifications
LLM_PROFILES = {
    "gpt-4": {
        "name": "GPT-4",
        "provider": "OpenAI",
        "strengths": ["Highest accuracy", "Best reasoning", "Excellent for complex analysis"],
        "weaknesses": ["Higher cost", "Slower response time"],
        "best_for": ["Insight generation", "Complex financial analysis", "Investment recommendations"],
        "cost_per_1k_tokens": 0.03,
        "avg_latency": 2.5,
        "accuracy_score": 0.95
    },
    "gemini-pro": {
        "name": "Gemini 1.5 Flash",
        "provider": "Google",
        "strengths": ["Fast processing", "Good accuracy", "Multimodal capabilities"],
        "weaknesses": ["Occasional inconsistencies", "Limited financial expertise"],
        "best_for": ["Company discovery", "Document parsing", "Quick analysis"],
        "cost_per_1k_tokens": 0.001,
        "avg_latency": 1.2,
        "accuracy_score": 0.85
    },
    "groq-mixtral": {
        "name": "Mixtral (Groq)",
        "provider": "Groq",
        "strengths": ["Ultra-fast inference", "Good balance of speed/quality", "Cost-effective"],
        "weaknesses": ["Limited context window", "Less sophisticated reasoning"],
        "best_for": ["Risk assessment", "Quick calculations", "Real-time responses"],
        "cost_per_1k_tokens": 0.0005,
        "avg_latency": 0.3,
        "accuracy_score": 0.82
    },
    "groq-llama3": {
        "name": "Llama 3 70B (Groq)",
        "provider": "Groq/Meta",
        "strengths": ["Very fast", "Open source", "Good general knowledge"],
        "weaknesses": ["Less financial expertise", "Smaller context window"],
        "best_for": ["KPI extraction", "Basic parsing", "Quick responses"],
        "cost_per_1k_tokens": 0.0007,
        "avg_latency": 0.4,
        "accuracy_score": 0.78
    },
    "claude-3": {
        "name": "Claude 3 Sonnet",
        "provider": "Anthropic",
        "strengths": ["Strong reasoning", "Good at nuanced analysis", "Ethical considerations"],
        "weaknesses": ["Higher cost", "Sometimes overly cautious"],
        "best_for": ["Risk assessment", "Detailed analysis", "Compliance checks"],
        "cost_per_1k_tokens": 0.015,
        "avg_latency": 2.0,
        "accuracy_score": 0.90
    }
}

# Test cases for evaluation
EVALUATION_TEST_CASES = [
    {
        "id": "company_discovery",
        "query": "Should I invest in Apple?",
        "expected_company": "Apple Inc.",
        "expected_ticker": "AAPL",
        "task_type": "company_discovery"
    },
    {
        "id": "financial_parsing",
        "query": "Extract key financial metrics",
        "test_data": {"revenue": 1000000000, "net_income": 100000000},
        "task_type": "parsing"
    },
    {
        "id": "kpi_calculation",
        "query": "Calculate financial ratios",
        "test_data": {"revenue": 1000000000, "net_income": 100000000, "total_assets": 5000000000},
        "expected_metrics": ["net_margin", "return_on_assets"],
        "task_type": "kpi"
    },
    {
        "id": "risk_assessment",
        "query": "Assess investment risk",
        "test_kpis": {"debt_to_equity": 2.5, "current_ratio": 0.8, "net_margin": -5},
        "expected_risk": "high",
        "task_type": "risk"
    },
    {
        "id": "investment_advice",
        "query": "Should I buy Tesla stock?",
        "context": {"price": 250, "pe_ratio": 45, "risk_score": 65},
        "task_type": "insight"
    }
]

class LLMEvaluator:
    """Evaluates and compares LLM performance for financial analysis tasks"""
    
    def __init__(self, llms: Dict):
        self.llms = llms
        self.evaluation_results = {}
        self.task_recommendations = {}
        
    def evaluate_llm_response(self, llm_name: str, response: str, test_case: Dict) -> Dict[str, float]:
        """Evaluate a single LLM response"""
        scores = {}
        
        # Accuracy score based on task type
        if test_case["task_type"] == "company_discovery":
            # Check if company and ticker are correctly identified
            accuracy = 0.0
            if test_case.get("expected_company", "").lower() in response.lower():
                accuracy += 0.5
            if test_case.get("expected_ticker", "").lower() in response.lower():
                accuracy += 0.5
            scores["accuracy"] = accuracy
            
        elif test_case["task_type"] == "parsing":
            # Check if data is properly structured
            try:
                # Check for JSON structure
                if "{" in response and "}" in response:
                    scores["accuracy"] = 0.8
                else:
                    scores["accuracy"] = 0.3
            except:
                scores["accuracy"] = 0.2
                
        elif test_case["task_type"] == "kpi":
            # Check if expected metrics are calculated
            accuracy = 0.0
            for metric in test_case.get("expected_metrics", []):
                if metric in response.lower():
                    accuracy += 0.5
            scores["accuracy"] = min(accuracy, 1.0)
            
        elif test_case["task_type"] == "risk":
            # Check risk assessment
            expected_risk = test_case.get("expected_risk", "")
            if expected_risk in response.lower():
                scores["accuracy"] = 0.9
            else:
                scores["accuracy"] = 0.4
                
        else:  # insight
            # Check for investment decision keywords
            decision_keywords = ["buy", "sell", "hold", "wait"]
            if any(keyword in response.lower() for keyword in decision_keywords):
                scores["accuracy"] = 0.8
            else:
                scores["accuracy"] = 0.5
        
        # Context understanding (based on response relevance)
        scores["context_understanding"] = min(len(response) / 500, 1.0) * 0.8 + 0.2
        
        # Output quality (based on structure and completeness)
        quality_score = 0.5
        if len(response) > 100:
            quality_score += 0.2
        if any(keyword in response.lower() for keyword in ["because", "therefore", "analysis", "based on"]):
            quality_score += 0.3
        scores["output_quality"] = min(quality_score, 1.0)
        
        return scores
    
    def run_evaluation_suite(self, progress_callback=None) -> Dict:
        """Run complete evaluation suite on all LLMs"""
        results = {}
        total_tests = len(EVALUATION_TEST_CASES) * len(self.llms)
        completed = 0
        
        for llm_name, llm in self.llms.items():
            results[llm_name] = {
                "test_results": [],
                "avg_scores": {},
                "response_times": [],
                "total_score": 0
            }
            
            for test_case in EVALUATION_TEST_CASES:
                start_time = time.time()
                
                try:
                    # Create appropriate prompt based on task type
                    prompt = self._create_test_prompt(test_case)
                    
                    # Get LLM response
                    from langchain_core.messages import HumanMessage
                    response = llm.invoke([HumanMessage(content=prompt)])
                    response_text = response.content
                    
                    # Measure response time
                    response_time = time.time() - start_time
                    results[llm_name]["response_times"].append(response_time)
                    
                    # Evaluate response
                    scores = self.evaluate_llm_response(llm_name, response_text, test_case)
                    
                    # Add time-based score
                    profile = LLM_PROFILES.get(llm_name, {})
                    expected_latency = profile.get("avg_latency", 2.0)
                    time_score = max(0, 1 - (response_time - expected_latency) / expected_latency)
                    scores["response_time"] = time_score
                    
                    # Add cost efficiency score
                    cost_per_token = profile.get("cost_per_1k_tokens", 0.01)
                    cost_score = 1 - (cost_per_token / 0.03)  # Normalized to GPT-4 cost
                    scores["cost_efficiency"] = max(0, cost_score)
                    
                    # Store results
                    results[llm_name]["test_results"].append({
                        "test_id": test_case["id"],
                        "task_type": test_case["task_type"],
                        "scores": scores,
                        "response_time": response_time,
                        "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text
                    })
                    
                except Exception as e:
                    # Handle errors gracefully
                    results[llm_name]["test_results"].append({
                        "test_id": test_case["id"],
                        "task_type": test_case["task_type"],
                        "scores": {metric: 0 for metric in METRIC_WEIGHTS.keys()},
                        "response_time": 999,
                        "error": str(e)
                    })
                
                completed += 1
                if progress_callback:
                    progress_callback(completed / total_tests)
        
        # Calculate average scores and recommendations
        self._calculate_final_scores(results)
        self._generate_task_recommendations(results)
        
        self.evaluation_results = results
        return results
    
    def _create_test_prompt(self, test_case: Dict) -> str:
        """Create appropriate prompt for test case"""
        task_type = test_case["task_type"]
        
        if task_type == "company_discovery":
            return test_case["query"]
        elif task_type == "parsing":
            return f"Parse this financial data and return structured JSON: Revenue: $1B, Net Income: $100M, Assets: $5B"
        elif task_type == "kpi":
            return f"Calculate key financial ratios from: {json.dumps(test_case['test_data'])}"
        elif task_type == "risk":
            return f"Assess investment risk given these KPIs: {json.dumps(test_case['test_kpis'])}"
        else:  # insight
            return f"{test_case['query']} Context: {json.dumps(test_case['context'])}"
    
    def _calculate_final_scores(self, results: Dict):
        """Calculate weighted final scores for each LLM"""
        for llm_name, llm_results in results.items():
            # Calculate average scores across all tests
            avg_scores = {metric: 0 for metric in METRIC_WEIGHTS.keys()}
            
            for test_result in llm_results["test_results"]:
                for metric, score in test_result["scores"].items():
                    avg_scores[metric] += score
            
            # Average the scores
            num_tests = len(llm_results["test_results"])
            for metric in avg_scores:
                avg_scores[metric] /= num_tests
            
            # Calculate weighted total score
            total_score = sum(avg_scores[metric] * METRIC_WEIGHTS[metric] 
                            for metric in METRIC_WEIGHTS.keys())
            
            results[llm_name]["avg_scores"] = avg_scores
            results[llm_name]["total_score"] = total_score
    
    def _generate_task_recommendations(self, results: Dict):
        """Generate recommendations for which LLM to use for each task"""
        task_types = ["company_discovery", "parsing", "kpi", "risk", "insight"]
        
        for task_type in task_types:
            task_scores = {}
            
            for llm_name, llm_results in results.items():
                # Get average score for this task type
                task_tests = [r for r in llm_results["test_results"] 
                            if r["task_type"] == task_type]
                
                if task_tests:
                    avg_accuracy = np.mean([t["scores"]["accuracy"] for t in task_tests])
                    avg_time = np.mean([t["response_time"] for t in task_tests])
                    
                    # Combined score considering accuracy and speed
                    combined_score = avg_accuracy * 0.7 + (1 - min(avg_time / 5, 1)) * 0.3
                    task_scores[llm_name] = combined_score
            
            # Sort by score
            sorted_llms = sorted(task_scores.items(), key=lambda x: x[1], reverse=True)
            self.task_recommendations[task_type] = sorted_llms

def create_evaluation_dashboard(evaluator: LLMEvaluator):
    """Create Streamlit dashboard for LLM evaluation results"""
    st.header("🤖 LLM Performance Evaluation")
    
    if not evaluator.evaluation_results:
        st.warning("No evaluation results available. Run evaluation first.")
        return
    
    # Overall performance comparison
    st.subheader("📊 Overall Performance Comparison")
    
    # Create DataFrame for overall scores
    overall_data = []
    for llm_name, results in evaluator.evaluation_results.items():
        profile = LLM_PROFILES.get(llm_name, {})
        overall_data.append({
            "LLM": profile.get("name", llm_name),
            "Provider": profile.get("provider", "Unknown"),
            "Total Score": results["total_score"],
            "Accuracy": results["avg_scores"]["accuracy"],
            "Speed": results["avg_scores"]["response_time"],
            "Cost Efficiency": results["avg_scores"]["cost_efficiency"],
            "Avg Response Time": np.mean(results["response_times"])
        })
    
    overall_df = pd.DataFrame(overall_data)
    overall_df = overall_df.sort_values("Total Score", ascending=False)
    
    # Display rankings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🥇 Best Overall", 
                 overall_df.iloc[0]["LLM"],
                 f"Score: {overall_df.iloc[0]['Total Score']:.2f}")
    
    with col2:
        st.metric("⚡ Fastest", 
                 overall_df.nsmallest(1, "Avg Response Time").iloc[0]["LLM"],
                 f"{overall_df.nsmallest(1, 'Avg Response Time').iloc[0]['Avg Response Time']:.2f}s")
    
    with col3:
        st.metric("💰 Most Cost-Effective",
                 overall_df.nlargest(1, "Cost Efficiency").iloc[0]["LLM"],
                 f"Score: {overall_df.nlargest(1, 'Cost Efficiency').iloc[0]['Cost Efficiency']:.2f}")
    
    # Radar chart for multi-metric comparison
    st.subheader("🎯 Multi-Metric Comparison")
    
    metrics = ["Accuracy", "Speed", "Cost Efficiency", "Context Understanding", "Output Quality"]
    
    fig = go.Figure()
    
    for llm_name, results in evaluator.evaluation_results.items():
        profile = LLM_PROFILES.get(llm_name, {})
        scores = results["avg_scores"]
        
        values = [
            scores["accuracy"],
            scores["response_time"],
            scores["cost_efficiency"],
            scores["context_understanding"],
            scores["output_quality"]
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=profile.get("name", llm_name)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Task-specific recommendations
    st.subheader("🎯 Task-Specific Recommendations")
    
    task_names = {
        "company_discovery": "Company Discovery",
        "parsing": "Document Parsing",
        "kpi": "KPI Calculation",
        "risk": "Risk Assessment",
        "insight": "Insight Generation"
    }
    
    rec_data = []
    for task_type, recommendations in evaluator.task_recommendations.items():
        if recommendations:
            best_llm = recommendations[0][0]
            profile = LLM_PROFILES.get(best_llm, {})
            rec_data.append({
                "Task": task_names.get(task_type, task_type),
                "Recommended LLM": profile.get("name", best_llm),
                "Score": f"{recommendations[0][1]:.2f}",
                "Reason": profile.get("best_for", ["General tasks"])[0]
            })
    
    rec_df = pd.DataFrame(rec_data)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)
    
    # Detailed metrics table
    st.subheader("📈 Detailed Performance Metrics")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Cost Analysis", "Speed Analysis"])
    
    with tab1:
        # Performance heatmap
        perf_data = []
        for llm_name, results in evaluator.evaluation_results.items():
            profile = LLM_PROFILES.get(llm_name, {})
            row = {"LLM": profile.get("name", llm_name)}
            for metric, score in results["avg_scores"].items():
                row[metric.replace("_", " ").title()] = score
            perf_data.append(row)
        
        perf_df = pd.DataFrame(perf_data)
        perf_df = perf_df.set_index("LLM")
        
        fig_heatmap = px.imshow(perf_df.T, 
                                color_continuous_scale="RdYlGn",
                                aspect="auto",
                                title="Performance Heatmap")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        # Cost analysis
        cost_data = []
        for llm_name in evaluator.evaluation_results.keys():
            profile = LLM_PROFILES.get(llm_name, {})
            cost_data.append({
                "LLM": profile.get("name", llm_name),
                "Cost per 1K tokens": f"${profile.get('cost_per_1k_tokens', 0):.3f}",
                "Relative Cost": profile.get('cost_per_1k_tokens', 0) / 0.03 * 100,  # Relative to GPT-4
                "Cost Efficiency Score": evaluator.evaluation_results[llm_name]["avg_scores"]["cost_efficiency"]
            })
        
        cost_df = pd.DataFrame(cost_data)
        
        # Bar chart for cost comparison
        fig_cost = px.bar(cost_df, x="LLM", y="Relative Cost",
                         title="Cost Comparison (Relative to GPT-4 = 100%)",
                         color="Cost Efficiency Score",
                         color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_cost, use_container_width=True)
        
        st.dataframe(cost_df, use_container_width=True, hide_index=True)
    
    with tab3:
        # Speed analysis
        speed_data = []
        for llm_name, results in evaluator.evaluation_results.items():
            profile = LLM_PROFILES.get(llm_name, {})
            speed_data.append({
                "LLM": profile.get("name", llm_name),
                "Avg Response Time": f"{np.mean(results['response_times']):.2f}s",
                "Min Time": f"{np.min(results['response_times']):.2f}s",
                "Max Time": f"{np.max(results['response_times']):.2f}s",
                "Expected Latency": f"{profile.get('avg_latency', 0):.1f}s"
            })
        
        speed_df = pd.DataFrame(speed_data)
        st.dataframe(speed_df, use_container_width=True, hide_index=True)
        
        # Box plot for response times
        fig_box = go.Figure()
        for llm_name, results in evaluator.evaluation_results.items():
            profile = LLM_PROFILES.get(llm_name, {})
            fig_box.add_trace(go.Box(
                y=results['response_times'],
                name=profile.get("name", llm_name),
                boxmean=True
            ))
        
        fig_box.update_layout(
            title="Response Time Distribution",
            yaxis_title="Response Time (seconds)",
            showlegend=True
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # LLM Profiles and Justifications
    st.subheader("🔍 LLM Profiles & Justifications")
    
    with st.expander("View detailed LLM profiles and use case justifications"):
        for llm_name, profile in LLM_PROFILES.items():
            if llm_name in evaluator.evaluation_results:
                st.markdown(f"### {profile['name']} ({profile['provider']})")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Strengths:**")
                    for strength in profile['strengths']:
                        st.markdown(f"• {strength}")
                    
                    st.markdown("**Best Used For:**")
                    for use_case in profile['best_for']:
                        st.markdown(f"• {use_case}")
                
                with col2:
                    st.markdown("**Weaknesses:**")
                    for weakness in profile['weaknesses']:
                        st.markdown(f"• {weakness}")
                    
                    st.markdown("**Performance Stats:**")
                    results = evaluator.evaluation_results[llm_name]
                    st.markdown(f"• Overall Score: {results['total_score']:.2f}")
                    st.markdown(f"• Avg Response Time: {np.mean(results['response_times']):.2f}s")
                
                st.divider()

def run_live_evaluation(llms: Dict, progress_placeholder=None):
    """Run evaluation and return results"""
    evaluator = LLMEvaluator(llms)
    
    # Run evaluation with progress callback
    def update_progress(progress):
        if progress_placeholder:
            progress_placeholder.progress(progress)
    
    results = evaluator.run_evaluation_suite(progress_callback=update_progress)
    
    return evaluator

# Integration function for main app
def show_llm_evaluation_page(llms: Dict):
    """Main function to be called from the main app"""
    st.header("🔬 LLM Evaluation & Comparison")
    
    # Check if evaluation has been run
    if "llm_evaluation_results" not in st.session_state:
        st.info("📊 Evaluate different LLMs to see which performs best for each financial analysis task.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Run LLM Evaluation", type="primary", use_container_width=True):
                progress_placeholder = st.empty()
                
                with st.spinner("Running comprehensive LLM evaluation..."):
                    evaluator = run_live_evaluation(llms, progress_placeholder)
                    st.session_state.llm_evaluation_results = evaluator
                
                progress_placeholder.empty()
                st.success("✅ Evaluation completed!")
                st.rerun()
        
        with col2:
            st.metric("Available LLMs", len(llms))
            st.metric("Test Cases", len(EVALUATION_TEST_CASES))
        
        # Show what will be tested
        with st.expander("📋 What will be tested?"):
            st.markdown("""
            **Evaluation Criteria:**
            - **Accuracy**: How well the LLM performs the task
            - **Speed**: Response time for queries
            - **Cost Efficiency**: Token cost optimization
            - **Context Understanding**: Comprehension of financial context
            - **Output Quality**: Structure and completeness of responses
            
            **Test Tasks:**
            1. **Company Discovery**: Identifying companies from natural language
            2. **Document Parsing**: Extracting financial data
            3. **KPI Calculation**: Computing financial ratios
            4. **Risk Assessment**: Evaluating investment risks
            5. **Insight Generation**: Providing investment recommendations
            """)
    
    else:
        # Show evaluation results
        evaluator = st.session_state.llm_evaluation_results
        
        # Add refresh button
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Re-run Evaluation"):
                del st.session_state.llm_evaluation_results
                st.rerun()
        
        # Create dashboard
        create_evaluation_dashboard(evaluator)
        
        # Show current LLM assignments
        st.subheader("⚙️ Current LLM Assignments in Your Workflow")
        
        # Map display names to actual LLM keys used in your workflow
        current_assignments = {
            "Company Discovery": ("gemini-pro", "Gemini 1.5 Flash"),
            "Document Parsing": ("groq-llama3", "Llama 3 70B (Groq)"),
            "KPI Extraction": ("groq-llama3", "Llama 3 70B (Groq)"),
            "Risk Assessment": ("groq-mixtral", "Mixtral (Groq)"),
            "Insight Generation": ("gpt-4", "GPT-4")
        }
        
        # Map task names to evaluation task types
        task_mapping = {
            "Company Discovery": "company_discovery",
            "Document Parsing": "parsing",
            "KPI Extraction": "kpi",
            "Risk Assessment": "risk",
            "Insight Generation": "insight"
        }
        
        assign_data = []
        for task, (llm_key, llm_display) in current_assignments.items():
            # Get the corresponding task type
            task_type = task_mapping.get(task, task.lower().replace(" ", "_"))
            
            # Check if it matches recommendation
            recommended_llm = ""
            status = "⚠️ Consider changing"
            
            if task_type in evaluator.task_recommendations:
                # Get the top recommended LLM for this task
                if evaluator.task_recommendations[task_type]:
                    recommended_llm_key = evaluator.task_recommendations[task_type][0][0]
                    recommended_score = evaluator.task_recommendations[task_type][0][1]
                    
                    # Check if current assignment matches recommendation
                    if recommended_llm_key == llm_key:
                        status = "✅ Optimal"
                    else:
                        # Check if current is in top 2 recommendations
                        top_2_llms = [rec[0] for rec in evaluator.task_recommendations[task_type][:2]]
                        if llm_key in top_2_llms:
                            status = "👍 Good choice"
                        
                        # Add recommendation
                        recommended_profile = LLM_PROFILES.get(recommended_llm_key, {})
                        recommended_llm = recommended_profile.get("name", recommended_llm_key)
            
            assign_data.append({
                "Task": task,
                "Current LLM": llm_display,
                "Status": status,
                "Recommended": recommended_llm if status.startswith("⚠️") else "—"
            })
        
        assign_df = pd.DataFrame(assign_data)
        st.dataframe(assign_df, use_container_width=True, hide_index=True)
        
        # Add explanation
        with st.expander("💡 Understanding the Status"):
            st.markdown("""
            - **✅ Optimal**: You're using the best LLM for this task based on evaluation
            - **👍 Good choice**: Your choice is in the top 2 recommendations
            - **⚠️ Consider changing**: A different LLM performed better in testing
            
            The recommendations are based on balancing accuracy, speed, and cost for each specific task.
            """)
            
            # Show detailed recommendations if any changes suggested
            changes_suggested = [item for item in assign_data if item["Status"].startswith("⚠️")]
            if changes_suggested:
                st.markdown("### 🔄 Suggested Changes:")
                for item in changes_suggested:
                    st.markdown(f"**{item['Task']}**: Consider switching from {item['Current LLM']} to {item['Recommended']}")
                    
                    # Show why
                    task_type = task_mapping.get(item['Task'])
                    if task_type and task_type in evaluator.task_recommendations:
                        current_llm_key = next((k for k, (_, display) in current_assignments.items() if display == item['Current LLM']), None)
                        if current_llm_key:
                            # Find current LLM's score
                            for llm, score in evaluator.task_recommendations[task_type]:
                                if llm == current_assignments[item['Task']][0]:
                                    st.caption(f"Current score: {score:.2f}")
                                    break
                            # Show recommended score
                            if evaluator.task_recommendations[task_type]:
                                st.caption(f"Recommended score: {evaluator.task_recommendations[task_type][0][1]:.2f}")