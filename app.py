import requests
import os

def upload_image(file_path):
    url = 'https://legendary-happiness-jj56v9q7vw7w3prwq-8000.app.github.dev/api/process_image/?fbclid=IwZXh0bgNhZW0CMTAAAR1n5H1McoAzsMJiy67_W2u_oHL4KlI7312iPs6A07LCY_3M7pSm0ZIugLg_aem__4S-cohPv2XVpQPS4hBrcw'  # Update this URL to your Django API endpoint'  # Update with your API endpoint
    with open(file_path, 'rb') as image_file:
        files = {'image': image_file}
        response = requests.post(url, files=files)
        if response.status_code == 200:
            print('Success:', response.json())
        else:
            print('Error:', response.json())

if __name__ == "__main__":
    file_path = input("Enter the path of the image to upload: ")
    if os.path.isfile(file_path):
        upload_image(file_path)
    else:
        print("File not found. Please check the path.")
