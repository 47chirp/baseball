# exporting baseball data to csv file

import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the page you want to scrape
url = 'https://baseballsavant.mlb.com/gamefeed?date=3/30/2024&gamePk=746168&chartType=pitch&legendType=pitchName&playerType=pitcher&inning=&count=&pitchHand=&batSide=&descFilter=&ptFilter=&resultFilter=&hf=pitchVelocity&sportId=1#746168'

# Send an HTTP GET request to fetch the page
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the page content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Initialize a list to hold scraped data
    pitches_data = []

    # Extract relevant information
    # Here, you'll need to identify the relevant HTML elements using the browser's Developer Tools to inspect
    # Adjust the parsing logic based on the actual structure of the page
    # Example structure (replace these tags/classes with actual ones)
    for pitch in soup.select('.pitch-row'):
        pitcher = pitch.select_one('.pitcher-name').get_text(strip=True)
        batter = pitch.select_one('.batter-name').get_text(strip=True)
        pitch_number = pitch.select_one('.pitch-number').get_text(strip=True)
        pitch_result = pitch.select_one('.pitch-result').get_text(strip=True)
        pitch_type = pitch.select_one('.pitch-type').get_text(strip=True)
        velocity = pitch.select_one('.pitch-velocity').get_text(strip=True)
        spin_rate = pitch.select_one('.pitch-spin-rate').get_text(strip=True)

        pitches_data.append({
            "Pitcher": pitcher,
            "Batter": batter,
            "Pitch Number": pitch_number,
            "Pitch Result": pitch_result,
            "Pitch Type": pitch_type,
            "Speed (mph)": velocity,
            "Spin Rate (rpm)": spin_rate
        })

    # Create a DataFrame from the scraped data
    df = pd.DataFrame(pitches_data)

    # Export to CSV
    df.to_csv('baseball_pitches.csv', index=False)

    print("CSV file has been exported.")

else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")

