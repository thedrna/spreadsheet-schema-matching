#!/usr/bin/env python3
"""
Test script for PandasAI Question Answering on CSV data.
This script tests PandasAI's ability to answer questions about the merged_comments.csv file
and compares the answers with actual Python/Pandas calculations.
"""

import os
import pandas as pd
import requests
import json
import re
from datetime import datetime
from pandasai import SmartDataframe
from pandasai.llm.base import LLM
import time
import numpy as np



class OllamaLLM(LLM):
    """Custom LLM class for Ollama integration with PandasAI"""
    
    def __init__(self, model_name, api_base="http://127.0.0.1:11500"):
        self.model_name = model_name
        self.api_base = api_base.rstrip('/')
        self._temperature = 0.1
        self._max_tokens = 1000  # Increased for longer code responses
        
    @property
    def type(self) -> str:
        return "ollama"
    
    def call(self, instruction: str, context: str = None, **kwargs) -> str:
        """Generate response from Ollama"""
        # Convert instruction to string if it's a Prompt object
        if hasattr(instruction, 'to_string'):
            prompt = instruction.to_string()
        elif hasattr(instruction, '__str__'):
            prompt = str(instruction)
        else:
            prompt = instruction
            
        # Combine with context if provided
        if context:
            if hasattr(context, 'to_string'):
                context_str = context.to_string()
            elif hasattr(context, '__str__'):
                context_str = str(context)
            else:
                context_str = context
            prompt = f"{context_str}\n\n{prompt}"
            
        url = f"{self.api_base}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
                "num_gpu": 999
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            answer = result.get('response', '')
            
            # Debug: Print the raw response
            print(f"\n[DEBUG] Raw LLM Response:\n{answer[:500]}...\n")
            
            return answer
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            raise


def verify_answer(question, pandasai_answer, actual_answer, tolerance=0.01):
    """Compare PandasAI answer with actual calculated answer.

    Important: Always print the *raw* PandasAI answer for traceability.
    """

    def _extract_first_number(x):
        """Return the first numeric value found in a string, or None."""
        if x is None:
            return None
        s = str(x)
        # Remove thousands separators to make parsing easier
        s = s.replace(",", "")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    def _norm_text(x):
        """Normalize text for string comparison."""
        if x is None:
            return ""
        return str(x).strip().lower()

    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"{'-'*80}")
    # Keep raw answer visible no matter what
    print(f"PandasAI Answer (RAW): {pandasai_answer}")
    print(f"Actual Answer:         {actual_answer}")

    # 1) Numeric comparison (best-effort): extract first number from both sides
    pa_num = _extract_first_number(pandasai_answer)
    actual_num = _extract_first_number(actual_answer)

    if pa_num is not None and actual_num is not None:
        diff = abs(pa_num - actual_num)
        print(f"\n[DEBUG] Parsed numeric -> PandasAI: {pa_num} | Actual: {actual_num}")
        if diff <= tolerance:
            print(f"✓ MATCH (numeric within tolerance {tolerance})")
            return True
        else:
            print(f"✗ MISMATCH (numeric difference: {diff})")
            return False

    # 2) String comparison (normalized)
    pa_txt = _norm_text(pandasai_answer)
    actual_txt = _norm_text(actual_answer)
    print(f"\n[DEBUG] Normalized text -> PandasAI: '{pa_txt}' | Actual: '{actual_txt}'")

    if pa_txt == actual_txt:
        print("✓ MATCH (string)")
        return True

    # 3) Soft match fallback for common cases:
    #    - PandasAI answers like: "The most common agency is EMLI."
    #    - Actual is: "EMLI"
    if actual_txt and actual_txt in pa_txt:
        print("✓ MATCH (actual is substring of PandasAI answer)")
        return True

    print("✗ MISMATCH")
    return False


def main():
    """Main test function"""
    
    # Configuration
    input_file = os.getenv('DF_QA_INPUT_CSV', 'merged_comments.csv')
    model_name = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
    ollama_host = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11500')
    run_dir = os.getenv('RUN_DIR', './outputs/test_df_qa_pandasai')
    
    print("="*80)
    print("PandasAI Question Answering Test")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Model: {model_name}")
    print(f"Ollama host: {ollama_host}")
    print(f"Output directory: {run_dir}")
    print("="*80)
    
    # Create output directory
    os.makedirs(run_dir, exist_ok=True)
    
    # Load the CSV file
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    
    # Prepare evaluation columns (dates + derived response time)
    df_eval = df.copy()
    for col in ["date_received", "date_responded"]:
        if col in df_eval.columns:
            df_eval[col] = pd.to_datetime(df_eval[col], errors="coerce")

    # Prefer existing response_time_days if present; otherwise derive from dates
    if "response_time_days" in df_eval.columns:
        df_eval["response_time_days_calc"] = pd.to_numeric(df_eval["response_time_days"], errors="coerce")
    elif "date_received" in df_eval.columns and "date_responded" in df_eval.columns:
        df_eval["response_time_days_calc"] = (
            (df_eval["date_responded"] - df_eval["date_received"]).dt.total_seconds() / 86400.0
        )
    else:
        df_eval["response_time_days_calc"] = pd.NA

    
    # testing subset
    print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"\nColumns: {', '.join(df.columns.tolist())}")
    print(f"\nFirst few rows:")
    print(df.head(3))
    
    # Initialize Ollama LLM
    print(f"\n\nInitializing Ollama LLM with model: {model_name}...")
    llm = OllamaLLM(model_name=model_name, api_base=ollama_host)
    
    # Create PandasAI Agent
    print("Creating PandasAI Agent...")
    from pandasai import Agent
    
    # SECURITY NOTE: This is a controlled environment because:
    # 1. Running on isolated compute node (SLURM job)
    # 2. Only processing known CSV file (merged_comments.csv)
    # 3. No network access from generated code
    # 4. Limited to read-only operations on data
    # 5. Output files written to restricted directory
    
    # For production, consider:
    # - Keep enforce_privacy=True (default)
    # - Use custom_whitelisted_dependencies to limit imports
    # - Set save_charts=False to prevent file writes
    # - Review generated code before execution with verbose=True
    
    agent = Agent(df, config={
        "llm": llm,
        "verbose": True,  # Shows generated code before execution
        "enable_cache": False,
        "enforce_privacy": False,  # Disabled for testing; re-enable for production
        "enable_logging": True,
        "max_retries": 3,
        "save_charts": False,  # Prevent chart file writes
        "open_charts": False,  # Prevent opening files
        "custom_whitelisted_dependencies": ["pandas", "numpy"]  # Limit allowed imports
    })
    print("✓ Agent created (running in controlled SLURM environment)")
    
    test_cases = [
                # =========================
        # EASY (basic grouped stats)
        # =========================
        {
            "question": "How many comments are in round 1?",
            "actual": lambda d: int((d[d["round"] == 1]).shape[0])
        },
        {
            "question": "What is the most common agency?",
            "actual": lambda d: (d["agency"].mode().iloc[0] if not d["agency"].mode().empty else "N/A")
        },
        {
            "question": "Which round has the highest number of comments?",
            "actual": lambda d: int(d["round"].value_counts().idxmax())
        },
        {
            "question": "How many comments are from the most common agency?",
            "actual": lambda d: int(d["agency"].value_counts().iloc[0])
        },
        {
            "question": "How many unanswered comments are there (missing date_responded)?",
            "actual": lambda d: int(d["date_responded"].isna().sum())
        },

        # =========================
        # MEDIUM (response-time + rates)
        # =========================
        {
            "question": "What is the median response time in days for responded comments?",
            "actual": lambda d: round(float(pd.to_numeric(d["response_time_days_calc"], errors="coerce").dropna().median()), 2)
        },
        {
            "question": "Which agency has the lowest median response time (days)?",
            "actual": lambda d: (
                d.assign(_rt=pd.to_numeric(d["response_time_days_calc"], errors="coerce"))
                 .dropna(subset=["_rt"])
                 .groupby("agency")["_rt"].median()
                 .sort_values()
                 .index[0]
            )
        },
        {
            "question": "What percentage of responded comments were answered within 30 days?",
            "actual": lambda d: (
                lambda tmp: round(float((tmp["_rt"] <= 30).mean() * 100), 2) if len(tmp) else 0.0
            )(
                d.assign(_rt=pd.to_numeric(d["response_time_days_calc"], errors="coerce"))
                 .dropna(subset=["_rt"])
            )
        },
        {
            "question": "In round 1, which agency has the most comments?",
            "actual": lambda d: (
                d[d["round"] == 1]
                 .groupby("agency").size()
                 .sort_values(ascending=False)
                 .index[0]
            )
        },
        {
            "question": "Which agency has the highest unanswered rate (share of rows with missing date_responded)?",
            "actual": lambda d: (
                d.assign(_unanswered=d["date_responded"].isna())
                 .groupby("agency")["_unanswered"].mean()
                 .sort_values(ascending=False)
                 .index[0]
            )
        },

        # =========================
        # HARD (comparisons + tail risk + dominance)
        # =========================
        {
            "question": "Which agency has the highest 90th percentile response time (days)?",
            "actual": lambda d: (
                d.assign(_rt=pd.to_numeric(d["response_time_days_calc"], errors="coerce"))
                 .dropna(subset=["_rt"])
                 .groupby("agency")["_rt"].quantile(0.9)
                 .sort_values(ascending=False)
                 .index[0]
            )
        },
        {
            "question": "How many comments received in the last 90 days are still unanswered?",
            "actual": lambda d: (
                lambda cutoff: int(
                    d[(d["date_received"].notna()) & (d["date_received"] >= cutoff)]["date_responded"].isna().sum()
                )
            )(d["date_received"].max() - pd.Timedelta(days=90))
        },
        {
            "question": "Which project has the slowest median response time (days)?",
            "actual": lambda d: (
                d.assign(_rt=pd.to_numeric(d["response_time_days_calc"], errors="coerce"))
                 .dropna(subset=["_rt"])
                 .groupby("project")["_rt"].median()
                 .sort_values(ascending=False)
                 .index[0]
            )
        },
        {
            "question": "Which agency's mean response time increases the most from round 1 to round 3+?",
            "actual": lambda d: (
                lambda tmp: (
                    (tmp.groupby(["agency", "_grp"])["_rt"].mean().unstack()["r3plus"]
                     - tmp.groupby(["agency", "_grp"])["_rt"].mean().unstack()["r1"]) 
                    .dropna()
                    .sort_values(ascending=False)
                    .index[0]
                )
            )(
                d.assign(
                    _rt=pd.to_numeric(d["response_time_days_calc"], errors="coerce"),
                    _grp=np.where(d["round"] == 1, "r1", np.where(d["round"] >= 3, "r3plus", "other"))
                )
                 .dropna(subset=["_rt"])
                 .query("_grp in ['r1','r3plus']")
            )
        },
        {
            "question": "Which (project, agency) pair has the highest share of comments within that project?",
            "actual": lambda d: (
                lambda shares: f"{shares.iloc[0]['project']} | {shares.iloc[0]['agency']} | share={round(float(shares.iloc[0]['share']), 4)}"
            )(
                (
                    (d.groupby(["project", "agency"]).size() / d.groupby("project").size())
                    .rename("share")
                    .reset_index()
                    .sort_values("share", ascending=False)
                )
            )
        },
        
        
    ]
    
    # Define test questions with their actual answers
    # test_cases = [
    #     {
    #         "question": "How many total rows are in this dataset?",
    #         "actual": len(df)
    #     },
    #     {
    #         "question": "How many unique projects are there?",
    #         "actual": df['project'].nunique()
    #     },
    #     {
    #         "question": "How many unique agencies are there?",
    #         "actual": df['agency'].nunique()
    #     },
    #     {
    #         "question": "What is the most common agency?",
    #         "actual": df['agency'].mode()[0] if not df['agency'].mode().empty else "N/A"
    #     },
    #     {
    #         "question": "How many comments are from EMLI agency?",
    #         "actual": len(df[df['agency'] == 'EMLI'])
    #     },
    #     {
    #         "question": "What is the average response time in days (excluding null values)?",
    #         "actual": round(df['response_time_days'].dropna().mean(), 2)
    #     },
    #     {
    #         "question": "What is the maximum response time in days?",
    #         "actual": df['response_time_days'].max()
    #     },
    #     {
    #         "question": "How many comments have a response time greater than 30 days?",
    #         "actual": len(df[df['response_time_days'] > 30])
    #     },
    #     {
    #         "question": "How many unique comment IDs are there?",
    #         "actual": df['comment_id'].nunique()
    #     },
    #     {
    #         "question": "How many comments are in round 1?",
    #         "actual": len(df[df['round'] == 1.0])
    #     }
    # ]
    
    # Run tests
    results = []
    print("\n\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n\nTest {i}/{len(test_cases)}")
        question = test['question']
        # actual_answer = test['actual']
        actual_fn_or_value = test["actual"]
        actual_answer = actual_fn_or_value(df_eval) if callable(actual_fn_or_value) else actual_fn_or_value
        
        try:
            # Get PandasAI answer
            print(f"Asking PandasAI: {question}")
            start_time = time.time()
            
            try:
                pandasai_answer = agent.chat(question)
            except Exception as chat_error:
                print(f"Chat error: {chat_error}")
                # Try to extract any useful information
                pandasai_answer = f"CHAT_ERROR: {str(chat_error)}"
            
            elapsed_time = time.time() - start_time
            print(f"Response time: {elapsed_time:.2f} seconds")
            
            # Verify answer
            match = verify_answer(question, pandasai_answer, actual_answer)
            
            results.append({
                'test_number': i,
                'question': question,
                'pandasai_answer': pandasai_answer,
                'actual_answer': actual_answer,
                'match': match,
                'response_time_seconds': round(elapsed_time, 2)
            })
            
        except Exception as e:
            import traceback
            print(f"✗ ERROR: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            results.append({
                'test_number': i,
                'question': question,
                'pandasai_answer': f"ERROR: {str(e)}",
                'actual_answer': actual_answer,
                'match': False,
                'response_time_seconds': 0
            })
        
        # Add a small delay between questions
        time.sleep(1)
    
    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['match'])
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_file = os.path.join(run_dir, f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Save detailed log
    log_file = os.path.join(run_dir, f'test_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(log_file, 'w') as f:
        f.write("PandasAI Question Answering Test Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Passed: {passed_tests}\n")
        f.write(f"Failed: {failed_tests}\n")
        f.write(f"Success rate: {success_rate:.1f}%\n\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            f.write(f"Test {result['test_number']}: {result['question']}\n")
            f.write(f"  PandasAI: {result['pandasai_answer']}\n")
            f.write(f"  Actual:   {result['actual_answer']}\n")
            f.write(f"  Match:    {'✓' if result['match'] else '✗'}\n")
            f.write(f"  Time:     {result['response_time_seconds']}s\n")
            f.write("\n")
    
    print(f"✓ Detailed log saved to: {log_file}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    # Exit with appropriate code
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)




















# #!/usr/bin/env python3
# """
# Test script for PandasAI Question Answering on CSV data.
# This script tests PandasAI's ability to answer questions about the merged_comments.csv file
# and compares the answers with actual Python/Pandas calculations.
# """

# import os
# import pandas as pd
# import requests
# import json
# from datetime import datetime
# from pandasai import SmartDataframe
# from pandasai.llm.base import LLM
# import time


# class OllamaLLM(LLM):
#     """Custom LLM class for Ollama integration with PandasAI"""
    
#     def __init__(self, model_name, api_base="http://127.0.0.1:11500"):
#         self.model_name = model_name
#         self.api_base = api_base.rstrip('/')
#         self._temperature = 0.1
#         self._max_tokens = 1000  # Increased for longer code responses
        
#     @property
#     def type(self) -> str:
#         return "ollama"
    
#     def call(self, instruction: str, context: str = None, **kwargs) -> str:
#         """Generate response from Ollama"""
#         # Convert instruction to string if it's a Prompt object
#         if hasattr(instruction, 'to_string'):
#             prompt = instruction.to_string()
#         elif hasattr(instruction, '__str__'):
#             prompt = str(instruction)
#         else:
#             prompt = instruction
            
#         # Combine with context if provided
#         if context:
#             if hasattr(context, 'to_string'):
#                 context_str = context.to_string()
#             elif hasattr(context, '__str__'):
#                 context_str = str(context)
#             else:
#                 context_str = context
#             prompt = f"{context_str}\n\n{prompt}"
            
#         url = f"{self.api_base}/api/generate"
#         payload = {
#             "model": self.model_name,
#             "prompt": prompt,
#             "stream": False,
#             "options": {
#                 "temperature": self._temperature,
#                 "num_predict": self._max_tokens,
#                 "num_gpu": 999
#             }
#         }
        
#         try:
#             response = requests.post(url, json=payload, timeout=300)
#             response.raise_for_status()
#             result = response.json()
#             answer = result.get('response', '')
            
#             # Debug: Print the raw response
#             print(f"\n[DEBUG] Raw LLM Response:\n{answer[:500]}...\n")
            
#             return answer
#         except Exception as e:
#             print(f"Error calling Ollama API: {e}")
#             raise


# def verify_answer(question, pandasai_answer, actual_answer, tolerance=0.01):
#     """Compare PandasAI answer with actual calculated answer"""
#     print(f"\n{'='*80}")
#     print(f"Question: {question}")
#     print(f"{'-'*80}")
#     print(f"PandasAI Answer: {pandasai_answer}")
#     print(f"Actual Answer:   {actual_answer}")
    
#     # Try to compare numerically if both are numbers
#     try:
#         pa_num = float(str(pandasai_answer).replace(',', ''))
#         actual_num = float(str(actual_answer).replace(',', ''))
        
#         if abs(pa_num - actual_num) <= tolerance:
#             print(f"✓ MATCH (within tolerance {tolerance})")
#             return True
#         else:
#             print(f"✗ MISMATCH (difference: {abs(pa_num - actual_num)})")
#             return False
#     except (ValueError, TypeError):
#         # String comparison
#         if str(pandasai_answer).strip().lower() == str(actual_answer).strip().lower():
#             print(f"✓ MATCH")
#             return True
#         else:
#             print(f"✗ MISMATCH")
#             return False


# def main():
#     """Main test function"""
    
#     # Configuration
#     input_file = os.getenv('DF_QA_INPUT_CSV', 'merged_comments.csv')
#     model_name = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
#     ollama_host = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11500')
#     run_dir = os.getenv('RUN_DIR', './outputs/test_df_qa_pandasai')
    
#     print("="*80)
#     print("PandasAI Question Answering Test")
#     print("="*80)
#     print(f"Input file: {input_file}")
#     print(f"Model: {model_name}")
#     print(f"Ollama host: {ollama_host}")
#     print(f"Output directory: {run_dir}")
#     print("="*80)
    
#     # Create output directory
#     os.makedirs(run_dir, exist_ok=True)
    
#     # Load the CSV file
#     print(f"\nLoading data from {input_file}...")
#     df = pd.read_csv(input_file)
#     # testing subset
#     print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
#     print(f"\nColumns: {', '.join(df.columns.tolist())}")
#     print(f"\nFirst few rows:")
#     print(df.head(3))
    
#     # Initialize Ollama LLM
#     print(f"\n\nInitializing Ollama LLM with model: {model_name}...")
#     llm = OllamaLLM(model_name=model_name, api_base=ollama_host)
    
#     # Create PandasAI Agent
#     print("Creating PandasAI Agent...")
#     from pandasai import Agent
    
#     # SECURITY NOTE: This is a controlled environment because:
#     # 1. Running on isolated compute node (SLURM job)
#     # 2. Only processing known CSV file (merged_comments.csv)
#     # 3. No network access from generated code
#     # 4. Limited to read-only operations on data
#     # 5. Output files written to restricted directory
    
#     # For production, consider:
#     # - Keep enforce_privacy=True (default)
#     # - Use custom_whitelisted_dependencies to limit imports
#     # - Set save_charts=False to prevent file writes
#     # - Review generated code before execution with verbose=True
    
#     agent = Agent(df, config={
#         "llm": llm,
#         "verbose": True,  # Shows generated code before execution
#         "enable_cache": False,
#         "enforce_privacy": False,  # Disabled for testing; re-enable for production
#         "enable_logging": True,
#         "max_retries": 3,
#         "save_charts": False,  # Prevent chart file writes
#         "open_charts": False,  # Prevent opening files
#         "custom_whitelisted_dependencies": ["pandas", "numpy"]  # Limit allowed imports
#     })
#     print("✓ Agent created (running in controlled SLURM environment)")
    
#     # Define test questions with their actual answers
#     test_cases = [
#         {
#             "question": "How many total rows are in this dataset?",
#             "actual": len(df)
#         },
#         {
#             "question": "How many unique projects are there?",
#             "actual": df['project'].nunique()
#         },
#         {
#             "question": "How many unique agencies are there?",
#             "actual": df['agency'].nunique()
#         },
#         {
#             "question": "What is the most common agency?",
#             "actual": df['agency'].mode()[0] if not df['agency'].mode().empty else "N/A"
#         },
#         {
#             "question": "How many comments are from EMLI agency?",
#             "actual": len(df[df['agency'] == 'EMLI'])
#         },
#         {
#             "question": "What is the average response time in days (excluding null values)?",
#             "actual": round(df['response_time_days'].dropna().mean(), 2)
#         },
#         {
#             "question": "What is the maximum response time in days?",
#             "actual": df['response_time_days'].max()
#         },
#         {
#             "question": "How many comments have a response time greater than 30 days?",
#             "actual": len(df[df['response_time_days'] > 30])
#         },
#         {
#             "question": "How many unique comment IDs are there?",
#             "actual": df['comment_id'].nunique()
#         },
#         {
#             "question": "How many comments are in round 1?",
#             "actual": len(df[df['round'] == 1.0])
#         }
#     ]
    
#     # Run tests
#     results = []
#     print("\n\n" + "="*80)
#     print("RUNNING TESTS")
#     print("="*80)
    
#     for i, test in enumerate(test_cases, 1):
#         print(f"\n\nTest {i}/{len(test_cases)}")
#         question = test['question']
#         actual_answer = test['actual']
        
#         try:
#             # Get PandasAI answer
#             print(f"Asking PandasAI: {question}")
#             start_time = time.time()
            
#             try:
#                 pandasai_answer = agent.chat(question)
#             except Exception as chat_error:
#                 print(f"Chat error: {chat_error}")
#                 # Try to extract any useful information
#                 pandasai_answer = f"CHAT_ERROR: {str(chat_error)}"
            
#             elapsed_time = time.time() - start_time
#             print(f"Response time: {elapsed_time:.2f} seconds")
            
#             # Verify answer
#             match = verify_answer(question, pandasai_answer, actual_answer)
            
#             results.append({
#                 'test_number': i,
#                 'question': question,
#                 'pandasai_answer': pandasai_answer,
#                 'actual_answer': actual_answer,
#                 'match': match,
#                 'response_time_seconds': round(elapsed_time, 2)
#             })
            
#         except Exception as e:
#             import traceback
#             print(f"✗ ERROR: {e}")
#             print(f"Traceback: {traceback.format_exc()}")
#             results.append({
#                 'test_number': i,
#                 'question': question,
#                 'pandasai_answer': f"ERROR: {str(e)}",
#                 'actual_answer': actual_answer,
#                 'match': False,
#                 'response_time_seconds': 0
#             })
        
#         # Add a small delay between questions
#         time.sleep(1)
    
#     # Summary
#     print("\n\n" + "="*80)
#     print("TEST SUMMARY")
#     print("="*80)
    
#     total_tests = len(results)
#     passed_tests = sum(1 for r in results if r['match'])
#     failed_tests = total_tests - passed_tests
#     success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
#     print(f"Total tests: {total_tests}")
#     print(f"Passed: {passed_tests}")
#     print(f"Failed: {failed_tests}")
#     print(f"Success rate: {success_rate:.1f}%")
    
#     # Save results to CSV
#     results_df = pd.DataFrame(results)
#     output_file = os.path.join(run_dir, f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
#     results_df.to_csv(output_file, index=False)
#     print(f"\n✓ Results saved to: {output_file}")
    
#     # Save detailed log
#     log_file = os.path.join(run_dir, f'test_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
#     with open(log_file, 'w') as f:
#         f.write("PandasAI Question Answering Test Results\n")
#         f.write("="*80 + "\n\n")
#         f.write(f"Input file: {input_file}\n")
#         f.write(f"Model: {model_name}\n")
#         f.write(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"Total tests: {total_tests}\n")
#         f.write(f"Passed: {passed_tests}\n")
#         f.write(f"Failed: {failed_tests}\n")
#         f.write(f"Success rate: {success_rate:.1f}%\n\n")
#         f.write("="*80 + "\n\n")
        
#         for result in results:
#             f.write(f"Test {result['test_number']}: {result['question']}\n")
#             f.write(f"  PandasAI: {result['pandasai_answer']}\n")
#             f.write(f"  Actual:   {result['actual_answer']}\n")
#             f.write(f"  Match:    {'✓' if result['match'] else '✗'}\n")
#             f.write(f"  Time:     {result['response_time_seconds']}s\n")
#             f.write("\n")
    
#     print(f"✓ Detailed log saved to: {log_file}")
    
#     print("\n" + "="*80)
#     print("TEST COMPLETE")
#     print("="*80)
    
#     # Exit with appropriate code
#     return 0 if failed_tests == 0 else 1


# if __name__ == "__main__":
#     exit_code = main()
#     exit(exit_code)
