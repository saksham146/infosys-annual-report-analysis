"""Manual end-to-end smoke check (requires a built index and Gemini API key)."""

from src.rag_pipeline import ask_annual_report_stream


def run_test():
    question = "What were consolidated revenue, total income, and profit for FY 2024?"
    answer_stream, source_pages = ask_annual_report_stream(question, top_k=12)
    print("\n".join(answer_stream))
    print(f"\nReport pages: {source_pages}")


if __name__ == "__main__":
    run_test()
