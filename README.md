# Cryptograpy (COL759) - Assignment 1 (Hill Cipher)

In this assighnment we are implementing the following :
* Hill Cipher encryption and decryption.
* Cryptanalysis of Hill Cipher.

## Hill cipher Encryption and Decryption
* For encryption use the following command 
```
python3 encrypt.py <key file> <plain-text file> 

Example:
python3 encrypt.py key1.txt plain_text.txt
```
The above command will encrypt the plain-text file and the output will be saved in "cipher_text_encrypt.txt" file in the current working directory. Plain-text file will contain English characters without any special character (except space). 

Using the "debug" (True / False) varible in "encrypt.py" we can see the intermediate results.


* For decryption use the following command
```
python3 decrypt.py <key file> <cipher-text file>

Example :
python3 decrypt.py key1.txt cipher_text_encrypt.txt
```
The above command will decrypt the cipher-text file and the output will be saved in "plain_text_decrypt.txt" file in the current working directory. Cipher-text file will contain English characters without any special character or space.

Using the "debug" (True / False) varible in "decrypt.py" we can see the intermediate results.

## Cryptanalysis of Hill Cipher
In this part we have done the cryptanalysis of Hill cipher. We will give a cipher-text file and a plain-text file as the input and get key as the output of the code. cipher-text file should be larger than plain-text file for IoC computation. For cryptanalysis of the Hill Cipher, use the following command
```
python3 partb.py <cipher-text file> <plain-text file>

Example:
python3 partb.py input3.txt input2.txt
```
The above command will analyse the cipher-text and the plain-text, and find the length of the key and the key by using which the palin text is being encryped using hill cipher algorithm. As a result it will print the key length, key, the IoC deviation, IoC and the plain text of the cipher text. 

Using the "debug" (True / False) varible in "partb.py" we can see the intermediate results.

Dummy output
```
Key size = 3
Key : [[2, 4, 5], [9, 2, 1], [3, 17, 7]]
IC_deviation from 0.065 : 0.0032577421976103804
IC value : 0.06825774219761038
Decrypted Text :
THESEARESHORTFAMOUSTEXTSINENGLISHFROMCLASSICSOURCESLIKETHEBIBLEORSHAKESPEARESOMETEXTSHAVEWORDDEFINITIONSANDEXPLANATIONSTOHELPYOUSOMEOFTHESETEXTSAREWRITTENINANOLDSTYLEOFENGLISHTRYTOUNDERSTANDTHEMBECAUSETHEENGLISHTHATWESPEAKTODAYISBASEDONWHATOURGREATGREATGREATGREATGRANDPARENTSSPOKEBEFOREOFCOURSENOTALLTHESETEXTSWEREORIGINALLYWRITTENINENGLISHTHEBIBLEFOREXAMPLEISATRANSLATIONBUTTHEYAREALLWELLKNOWNINENGLISHTODAYANDMANYOFTHEMEXPRESSBEAUTIFULTHOUGHTSXX

```

## Contributers
* Rajat Singh (CSZ208507)
* Sahil Vijay Dahake (CS5170488)
