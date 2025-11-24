import os
import time
import requests
import base64
import io
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys

# ==========================================
# CONFIGURATION
# ==========================================
DOWNLOAD_PATH = "datasets/cathay_eaten_raw"
MAX_IMAGES_PER_QUERY = 50  # Keep low to avoid bans, increase if needed

# Specific keywords to find "Eaten/Waste" specifically for Cathay
SEARCH_QUERIES = [
    "CX economy meal eaten tray",
    "CX economy meal empty tray",
    "CX economy food waste",
    "CX inflight meal leftovers",
    "Cathay Pacific economy meal eaten tray",
    "Cathay Pacific economy meal empty tray",
    "Cathay Pacific economy food waste",
    "Cathay Pacific inflight meal leftovers",
    "國泰航空 CX 飛機餐 食剩",   
    "國泰航空 CX 經濟艙 吃完",   
    "國泰航空 CX 飛機餐 空盤",
    "國泰航空 CX 飛機餐 空食",
    "國泰航空 CX 經濟艙 食完",
    "國泰航空 CX 經濟艙 難食", 
    "國泰航空 CX 經濟艙 唔好食", 
    "國泰航空 CX 經濟艙 搞唔掂",      
    "airline meal eaten tray economy" 
]
# ==========================================

def setup_driver():
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless") # Run in background (Comment out to see the browser working)
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def scroll_to_bottom(driver):
    """Scrolls down to load more images."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for page to load
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                # Try clicking "Show more results" button if it exists
                load_more = driver.find_element(By.CSS_SELECTOR, ".mye4qd")
                if load_more.is_displayed():
                    load_more.click()
                    time.sleep(2)
                else:
                    break
            except:
                break
        last_height = new_height

def download_images(driver, query):
    print(f"--- Searching for: {query} ---")
    driver.get("https://www.google.com/imghp?hl=en")
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys(query)
    search_box.send_keys(Keys.ENTER)
    
    time.sleep(2)
    scroll_to_bottom(driver)
    
    # Find image elements
    img_results = driver.find_elements(By.CSS_SELECTOR, "img.YQ4gaf")
    
    count = 0
    for img in img_results:
        if count >= MAX_IMAGES_PER_QUERY:
            break
            
        try:
            # Get image source (src)
            src = img.get_attribute("src")
            
            if src and "http" in src:
                # Setup folder
                query_folder = query.replace(" ", "_")
                save_dir = os.path.join(DOWNLOAD_PATH, query_folder)
                os.makedirs(save_dir, exist_ok=True)
                
                # Download
                try:
                    img_data = requests.get(src).content
                    file_path = os.path.join(save_dir, f"{count}.jpg")
                    with open(file_path, "wb") as f:
                        f.write(img_data)
                    print(f"Saved: {file_path}")
                    count += 1
                except Exception as e:
                    print(f"Failed to download image: {e}")

            elif src and "data:image" in src:
                # Handle Base64 images (Google often uses these for thumbnails)
                try:
                    query_folder = query.replace(" ", "_")
                    save_dir = os.path.join(DOWNLOAD_PATH, query_folder)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    header, encoded = src.split(",", 1)
                    data = base64.b64decode(encoded)
                    file_path = os.path.join(save_dir, f"base64_{count}.jpg")
                    with open(file_path, "wb") as f:
                        f.write(data)
                    print(f"Saved (Base64): {file_path}")
                    count += 1
                except Exception as e:
                    print(f"Failed to save base64 image: {e}")
                    
        except Exception as e:
            continue

def main():
    driver = setup_driver()
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)
        
    for query in SEARCH_QUERIES:
        download_images(driver, query)
        time.sleep(5) # Be polite to Google servers
        
    driver.quit()
    print("\nDone! Now you must manually filter the images.")

if __name__ == "__main__":
    main()
