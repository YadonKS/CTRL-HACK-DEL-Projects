import sys
import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from base64 import b64encode, b64decode
import os
import openai

import requests

# Replace with your OpenAI API key
openai.api_key = 'your-api-key'

def aes_encrypt(data, key):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    return iv + encryptor.update(data) + encryptor.finalize()

def aes_decrypt(ciphertext, key):
    iv = ciphertext[:16]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext[16:]) + decryptor.finalize()

def triple_des_encrypt(data, key):
    iv = os.urandom(8)
    cipher = Cipher(algorithms.TripleDES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    return iv + encryptor.update(data) + encryptor.finalize()

def triple_des_decrypt(ciphertext, key):
    iv = ciphertext[:8]
    cipher = Cipher(algorithms.TripleDES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext[8:]) + decryptor.finalize()

def blowfish_encrypt(data, key):
    iv = os.urandom(8)
    cipher = Cipher(algorithms.Blowfish(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    return iv + encryptor.update(data) + encryptor.finalize()

def blowfish_decrypt(ciphertext, key):
    iv = ciphertext[:8]
    cipher = Cipher(algorithms.Blowfish(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext[8:]) + decryptor.finalize()

def chacha20_encrypt(data, key):
    nonce = os.urandom(16)
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    return nonce + encryptor.update(data)

def chacha20_decrypt(ciphertext, key):
    nonce = ciphertext[:16]
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext[16:])

def rsa_encrypt(data, public_key):
    return public_key.encrypt(
        data,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    )

def rsa_decrypt(ciphertext, private_key):
    return private_key.decrypt(
        ciphertext,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    )

def encrypt_message(message, aes_key, des_key, blowfish_key, chacha_key, rsa_public_key):
    data = message.encode()
    
    data = aes_encrypt(data, aes_key)
    print("AES Encrypted:", b64encode(data).decode())

    data = triple_des_encrypt(data, des_key)
    print("Triple DES Encrypted:", b64encode(data).decode())

    data = blowfish_encrypt(data, blowfish_key)
    print("Blowfish Encrypted:", b64encode(data).decode())

    data = chacha20_encrypt(data, chacha_key)
    print("ChaCha20 Encrypted:", b64encode(data).decode())

    data = rsa_encrypt(data, rsa_public_key)
    print("RSA Encrypted:", b64encode(data).decode())

    return data

# Layered decryption
def decrypt_message(data, aes_key, des_key, blowfish_key, chacha_key, rsa_private_key):
    # Step 1: RSA Decryption
    data = rsa_decrypt(data, rsa_private_key)
    print("RSA Decrypted:", b64encode(data).decode())

    # Step 2: ChaCha20 Decryption
    data = chacha20_decrypt(data, chacha_key)
    print("ChaCha20 Decrypted:", b64encode(data).decode())

    # Step 3: Blowfish Decryption
    data = blowfish_decrypt(data, blowfish_key)
    print("Blowfish Decrypted:", b64encode(data).decode())

    # Step 4: Triple DES Decryption
    data = triple_des_decrypt(data, des_key)
    print("Triple DES Decrypted:", b64encode(data).decode())

    # Step 5: AES Decryption
    data = aes_decrypt(data, aes_key)
    print("AES Decrypted:", data.decode())

    return data.decode()

aes_key = os.urandom(32) 
des_key = os.urandom(24)   
blowfish_key = os.urandom(16)
chacha_key = os.urandom(32)
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
public_key = private_key.public_key()

# Encrypt and decrypt message
message = "Hello, World!"
print("Original Message:", message)

# Encrypt
encrypted_message = encrypt_message(message, aes_key, des_key, blowfish_key, chacha_key, public_key)
# Format the prompt to clarify the question
prompt = f"Is the following input a string? Input: {encrypted_message}"

# Call the OpenAI API with the formatted prompt
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # Use gpt-3.5-turbo or gpt-4 if available
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
)

# Extract the response content
answer = response['choices'][0]['message']['content'].strip().lower()
if answer == "yes":
# Decrypt
    decrypted_message = decrypt_message(encrypted_message, aes_key, des_key, blowfish_key, chacha_key, private_key)
    print("Decrypted Message:", decrypted_message)

message = encrypted_message
print("In final Python script with encrypted message:", message)

# Your API key (replace with your actual API key)
api_key = 'your-api-key'

# API endpoint for sending emails
url = 'https://smtp.maileroo.com/send'

# Email data
data = {
    'from': 'helloworld@0815d524ef2c565a.maileroo.org',  # Sender's email address
    'to': 'krishnayearbook@gmail.com',  # Recipient email address (comma-separated for multiple recipients)
    'subject': 'Hello World!',  # Email subject
    'html': '<p>This is a test email in <strong>HTML</strong> format.</p>',  # HTML body content (required)
    'plain': 'This is a test email in plain text format.',  # Plain text body content (optional)
    'attachments': [],  # Optional: Array of file attachments
    'inline_attachments': [],  # Optional: Array of inline file attachments
    'reference_id': '',  # Optional: Reference ID (24-character hexadecimal string)
    'tags': {'category': 'test_email'},  # Optional: Tags for the email (JSON object)
    'tracking': 'no'  # Optional: Enable email tracking (yes or no)
}

# Headers with API Key for authentication
headers = {
    'X-API-Key': api_key
}

# Send the POST request
response = requests.post(url, data=data, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    print('Email sent successfully!')
else:
    print(f'Failed to send email. Status code: {response.status_code}, Error: {response.text}')