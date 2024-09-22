import os
import time
from datetime import datetime,timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException

FILE_FRESHNESS_THRESHOLD = timedelta(hours=4)

def download_historical_forex_data(period):
    base_url = "https://data.forexsb.com"
    data_url = "/data-app"

    # Set up the download directory
    download_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(download_dir, exist_ok=True)

    # Set up the Selenium WebDriver with the specified download directory
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run headless
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(base_url + data_url)

    # Wait until the dropdowns are present
    wait = WebDriverWait(driver, 30)

    # Select the currency pair dropdown
    currency_pair_dropdown = wait.until(EC.presence_of_element_located((By.ID, "select-symbol")))
    currency_pair_dropdown.click()
    time.sleep(1)

    # Select EURUSD from the dropdown
    currency_pair_option = wait.until(EC.element_to_be_clickable((By.XPATH, "//option[contains(text(), 'EURUSD')]")))
    currency_pair_option.click()

    # Select the format (MetaTrader)
    format_dropdown = wait.until(EC.presence_of_element_located((By.ID, "select-format")))
    format_dropdown.click()
    time.sleep(1)
    format_option = wait.until(EC.element_to_be_clickable((By.XPATH, "//option[contains(text(), 'MetaTrader (CSV)')]")))
    format_option.click()

    # Click the "Load data" button
    load_data_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load data')]")))
    load_data_button.click()

    # Wait for the data to load
    try:
        wait.until(EC.presence_of_element_located((By.ID, "table-acquisition")))
        print("Data loaded successfully.")
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#table-acquisition tbody tr")))
    except Exception as e:
        print(f"Error waiting for data to load: {e}")

    # Find the table containing the data
    try:
        table = driver.find_element(By.ID, "table-acquisition")

        # Iterate through each row in the table
        for row in table.find_elements(By.TAG_NAME, "tr"):
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 6:
                symbol_name = cols[0].text.strip()
                period_name = cols[1].text.strip()
                download_link = cols[5].find_element(By.TAG_NAME, "a")

                if symbol_name == "EURUSD" and period_name == period and download_link:
                    # Check if an older version of the file exists
                    file_name = f"{symbol_name}_{period_name}.csv"
                    file_path = os.path.join(download_dir, file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Deleted older version: {file_name}")

                    # Click the download link directly (the text of the link)
                    download_link.click()
                    print(f"Successfully downloaded: {file_name}")
                    time.sleep(2)
                    break
    except (NoSuchElementException, StaleElementReferenceException) as e:
        print(f"Error finding the table or downloading files: {e}")

    # Close the browser
    driver.quit()
    print("Finished downloading all files.")



def is_file_stale(file_path):
    """Check if the file is older than the freshness threshold."""
    if os.path.exists(file_path):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        return datetime.now() - file_mod_time > FILE_FRESHNESS_THRESHOLD
    return True  # If the file doesn't exist, consider it stale


def periodic_download():
    # Define paths for each period's data
    data_files = {
        "M30": os.path.join(os.getcwd(), "datasets", "EURUSD_M30.csv"),
        "H1": os.path.join(os.getcwd(), "datasets", "EURUSD_H1.csv"),
        "H4": os.path.join(os.getcwd(), "datasets", "EURUSD_H4.csv"),
        "D1": os.path.join(os.getcwd(), "datasets", "EURUSD_D1.csv"),
    }

    while True:
        now = datetime.now()
        weekday = now.weekday()

        if weekday < 7:  # Check if it's a weekday
            # Check each period and download if necessary
            for period in data_files.keys():
                if is_file_stale(data_files[period]):
                    print(f"Downloading {period}...")
                    download_historical_forex_data(period)
                else:
                    print(f"{period} data is up to date.")

            # Sleep for a longer duration to avoid frequent checks
            time.sleep(3600)  # Check every hour after processing all downloads
        else:
            print("It's the weekend. Skipping downloads.")
            time.sleep(86400)  # Sleep for a day if it's the weekend

if __name__ == "__main__":
    periodic_download()