# Cryptograpy (COL759)

In this assighnment we are implementing the following :
* Hill Cipher encryption and decryption.
* Cryptanalysis of Hill Cipher.

## Hill cipher Encryption and Decryption
* For encryption use the following command 
```
python3 encrypt.py <key file> <plain-text file> 

python3 encrypt.py key1.txt plain_text.txt
```
The above command will encrypt the plain-text file and the output will be saved in "cipher_text_encrypt.txt" file in the current working directory. Plain-text file will contain English characters without any special character (except space). 

Using the "debug" (True / False) varible in "encrypt.py" we can see the intermediate results.


* For decryption use the following command
```
python3 decrypt.py <key file> <cipher-text file>

python3 decrypt.py key1.txt cipher_text_encrypt.txt
```
The above command will decrypt the cipher-text file and the output will be saved in "plain_text_decrypt.txt" file in the current working directory. Cipher-text file will contain English characters without any special character or space.

Using the "debug" (True / False) varible in "decrypt.py" we can see the intermediate results.

## Cryptanalysis of Hill Cipher


## Contributers
* Rajat Singh
* Sahil Vijay Dahake
