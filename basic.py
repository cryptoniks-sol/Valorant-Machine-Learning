import requests
import json

def test_api():
    base_url = "http://localhost:5000/api/v1"
    
    # Test results endpoint
    print("Testing results endpoint...")
    try:
        response = requests.get(f"{base_url}/results", params={"page": 1})
        response.raise_for_status()
        data = response.json()
        print(f"Response status: {response.status_code}")
        print(f"Data structure: {json.dumps(data.keys(), indent=2)}")
        print(f"Number of matches: {len(data.get('data', []))}")
        
        # Print sample match structure
        if data.get('data'):
            print("\nSample match structure:")
            sample_match = data['data'][0]
            print(json.dumps(sample_match, indent=2))
    except Exception as e:
        print(f"Error: {e}")
    
    # Add tests for other endpoints as needed
    # Test teams endpoint
    print("\nTesting teams endpoint with sample team...")
    try:
        # Try with the first team from a match
        sample_team_name = data['data'][0]['teams'][0]['name']
        response = requests.get(f"{base_url}/teams/{sample_team_name}")
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            team_data = response.json()
            print(f"Team data structure: {json.dumps(team_data, indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()