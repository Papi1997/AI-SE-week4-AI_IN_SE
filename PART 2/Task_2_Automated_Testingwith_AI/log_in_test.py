from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# --- Configuration ---
LOGIN_URL = "https://practicetestautomation.com/practice-test-login/"
VALID_USERNAME = "student"
VALID_PASSWORD = "Password123"
INVALID_USERNAME = "wronguser" # Or any other invalid username you prefer
INVALID_PASSWORD = "wrongpassword" # Or any other invalid password you prefer
# --- Helper Function to Initialize Driver ---
def get_driver():
    """Initializes and returns a Chrome WebDriver."""
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless") # Uncomment to run in headless mode (no browser UI)
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    return driver

# --- Test Case 1: Valid Login ---
def test_valid_login():
    driver = get_driver()
    try:
        driver.get(LOGIN_URL)
        print(f"Navigated to: {LOGIN_URL}")

        # Wait for elements to be present (robustness)
        # Inside def test_valid_login():

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "username"))
        )
        username_field = driver.find_element(By.ID, "username")
        password_field = driver.find_element(By.ID, "password") 
        login_button = driver.find_element(By.ID, "submit") 

        username_field.send_keys(VALID_USERNAME)
        password_field.send_keys(VALID_PASSWORD)
        login_button.click()

        
        # On the practice site, successful login redirects to a specific URL
        try:
            WebDriverWait(driver, 10).until(
                EC.url_contains("/logged-in-successfully/") # Waits for the URL to contain this specific path
            )
            # If the above line passes without error, then the URL was found
            if "/logged-in-successfully/" in driver.current_url:
                print(f"Valid Login Test: PASS - Successfully logged in. Current URL: {driver.current_url}")
                return "PASS"
            else: # This 'else' usually means the WebDriverWait failed, but good to have
                print("Valid Login Test: FAIL - Did not reach the success page.")
                return "FAIL"
        except Exception as e:
            # If a TimeoutException occurs, it means the URL was not found within 10 seconds
            print(f"Valid Login Test: FAIL - Did not reach the success page within 10 seconds. Error: {e}")
            return "FAIL"
    finally:
        driver.quit() # Close the browser



# --- Test Case 2: Invalid Login ---
def test_invalid_login():
    driver = get_driver()
    try:
        driver.get(LOGIN_URL)
        print(f"Navigated to: {LOGIN_URL}")

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "username"))
        )
        username_field = driver.find_element(By.ID, "username")
        password_field = driver.find_element(By.ID, "password")
        login_button = driver.find_element(By.ID, "submit")

        username_field.send_keys(INVALID_USERNAME)
        password_field.send_keys(INVALID_PASSWORD)
        login_button.click()



        
        # --- Validation for Invalid Login ---
        # On the practice site, an error message appears with ID "error"
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "error")) # This is the ID of the error message on the practice site
        )
        error_message_element = driver.find_element(By.ID, "error") # Finds the error message element
        error_text = error_message_element.text # Gets the text from the error message
        if "Your username is invalid!" in error_text or "Your password is invalid!" in error_text: # Checks for the specific error text
            print(f"Invalid Login Test: PASS - Error message found: '{error_text}'")
            return "PASS"
        else:
            print(f"Invalid Login Test: FAIL - Expected error message not found. Message: '{error_text}'")
            return "FAIL"

    except Exception as e:
        print(f"Invalid Login Test: ERROR - {e}")
        return "ERROR"
    finally:
        driver.quit() # Close the browser




# --- Run Tests ---
if __name__ == "__main__":
    print("\n--- Running Login Tests ---")
    results = {
        "valid_login": test_valid_login(),
        "invalid_login": test_invalid_login()
    }

    print("\n--- Test Results ---")
    for test_name, status in results.items():
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    pass_count = list(results.values()).count("PASS")
    total_tests = len(results)
    success_rate = (pass_count / total_tests) * 100 if total_tests > 0 else 0

    print(f"\nOverall Success Rate: {success_rate:.2f}% ({pass_count}/{total_tests} tests passed)")