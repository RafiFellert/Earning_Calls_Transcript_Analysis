import os
import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

BASE_URL = "https://stockanalysis.com"
# Add the ticker name you are looking for
TICKER = "CSCO"

INDEX_URL = (
    f"{BASE_URL}/stocks/{TICKER.lower()}/transcripts/"
)

OUTPUT_DIR = os.path.join("transcripts", TICKER.upper())

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Start browser
# -------------------------------------------------------

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)

# -------------------------------------------------------
# Get transcript list
# -------------------------------------------------------

print("Loading transcript page...")

driver.get(INDEX_URL)

time.sleep(5)

soup = BeautifulSoup(driver.page_source, "html.parser")

links = []

for a in soup.find_all("a", href=True):

    href = a["href"]

    if (
        f"/stocks/{TICKER.lower()}/transcripts/" in href
        and href != f"/stocks/{TICKER.lower()}/transcripts/"
    ):
        links.append(href)

# Remove duplicates

links = sorted(set(links))

links = [
    x for x in links
    if re.search(
        r'q[1-4]-(202[1-6])',
        x,
        re.IGNORECASE
    )
]

print(f"Found {len(links)} earnings calls for TICKER={TICKER}")

# -------------------------------------------------------
# Download each transcript
# -------------------------------------------------------

for i, link in enumerate(links, 1):

    url = BASE_URL + link

    print(
        f"[{i}/{len(links)}] {url}"
    )

    driver.get(url)

    time.sleep(3)

    soup = BeautifulSoup(
        driver.page_source,
        "html.parser"
    )

    # -----------------------------
    # Title
    # -----------------------------

    title_tag = soup.find("h1")

    title = (
        title_tag.get_text(strip=True)
        if title_tag
        else "Unknown"
    )

    # -----------------------------
    # Transcript text
    # -----------------------------

    article = soup.find(
        "div",
        class_="transcript-body"
    )

    if article:

        transcript = article.get_text(
            "\n",
            strip=True
        )

    else:

        transcript = soup.get_text(
            "\n",
            strip=True
        )

    # -----------------------------
    # Create filename
    # Format: YYYY-Mon-DD-TICKER.txt
    # -----------------------------

    date_pattern = r'([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{4})'

    # 1. Try the title
    date_match = re.search(date_pattern, title)

    # 2. If not found, look for the line immediately
    #    following "Earnings Call:"
    if not date_match:

        page_text = soup.get_text("\n", strip=True)

        earnings_match = re.search(
            r'Earnings Call:.*?\n([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})',
            page_text,
            re.DOTALL
        )

        if earnings_match:
            date_match = re.search(
                date_pattern,
                earnings_match.group(1)
            )

    # 3. Create filename
    if date_match:

        month, day, year = date_match.groups()

        filename = (
            f"{year}-{month}-{int(day):02d}-{TICKER.upper()}.txt"
        )

    else:
        print("didn't find date_match")

    # -----------------------------
    # Save transcript
    # -----------------------------

    file_path = os.path.join(OUTPUT_DIR, filename)

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(transcript)

    print(f"Saved: {file_path}")

    time.sleep(2)

driver.quit()

print("Done!")
