"""Simple manual test for answer_with_rag (Feature 4).

Run inside virtual environment:
  python test.py
"""
from pprint import pprint
from tools import answer_with_rag

if __name__ == "__main__":
  print("Product docs question:")
  result1 = answer_with_rag("How do I connect Snowflake?")
  pprint(result1)
  print("\nDeveloper docs question:")
  result2 = answer_with_rag("How do I authenticate with the Atlan API using a token?")
  pprint(result2)
