from pathlib import Path

import requests

test_solidity = Path(__file__).with_name("test.sol").read_text(encoding="utf-8")
payload = {
    "code": test_solidity,
}
response = requests.post("http://127.0.0.1:8000/analyze", json=payload, timeout=120)
print(response.json())